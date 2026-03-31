import numpy as np
import cv2
from typing import List, Dict, Tuple
from models.model_loader import ModelLoader
from services.image_service import preprocess_image
from services.text_service import preprocess_text, read_file_content
import tensorflow as tf


def convert_scores_to_names(scores, label_map):
    """Convert dictionary từ index-based sang name-based"""
    return {label_map.get(int(k), 'Unknown'): round(v, 3) for k, v in scores.items()}


def convert_top_predictions(top_predictions, label_map):
    """Convert top predictions sang dạng có tên nhãn"""
    converted = []
    for pred in top_predictions:
        label_idx = int(pred[0])
        converted.append({
            "label_id": label_idx,
            "label_name": label_map.get(label_idx, 'Unknown'),
            "score": round(float(pred[1]), 3)
        })
    return converted


async def process_voting_inputs(image_file, text_input, text_file):

    # Xử lý ảnh
    image_content = await image_file.read()
    image = np.frombuffer(image_content, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Tiền xử lý ảnh cho DenseNet
    processed_image, error = preprocess_image(image, model_name="DenseNet121")

    # Xử lý text
    file_text = await read_file_content(text_file)
    final_text = file_text or text_input or ""
    processed_text = preprocess_text(final_text)

    return processed_image, processed_text


async def predict_voting(image_file, text_input, text_file):
    try:
        # Load các model cần thiết
        image_models = [
            ModelLoader.load_image_model("DenseNet121"),
            ModelLoader.load_image_model("DenseNet169")
        ]

        text_models = [
            ModelLoader.load_text_model("FastText"),
            ModelLoader.load_text_model("Electra")
        ]

        # Validate models
        if None in image_models or None in text_models:
            missing = []
            for idx, model in enumerate(image_models):
                if model is None:
                    missing.append("DenseNet121" if idx ==
                                   0 else "DenseNet169")
            for idx, model in enumerate(text_models):
                if model is None:
                    missing.append("FastText" if idx == 0 else "Electra")

            if missing:
                return {"error": f"Thiếu model: {', '.join(missing)}"}

        # Xử lý đầu vào
        image_array, processed_text = await process_voting_inputs(image_file, text_input, text_file)

        # Dự đoán ảnh
        image_probs = []
        for model in image_models:
            if model is not None:
                pred = model.predict(image_array.astype(
                    np.float32))  # Đảm bảo dtype
                prob = tf.nn.softmax(pred).numpy()[0]
                image_probs.append(prob)

        # Dự đoán text
        text_probs = []
        # FastText
        if text_models[0]:
            try:
                ft_prob = predict_text_fasttext(text_models[0], processed_text)
                text_probs.append(ft_prob)
            except Exception as e:
                print(f"Lỗi dự đoán FastText: {str(e)}")
                # Thêm mảng zeros nếu có lỗi
                label_map = ModelLoader.get_label_mapping('text')
                text_probs.append(np.zeros(len(label_map)))

        # Electra
        if text_models[1] and isinstance(text_models[1], dict) and "pipeline" in text_models[1]:
            try:
                # Thêm dòng khai báo label_map trước khi sử dụng
                label_map = ModelLoader.get_label_mapping(
                    'text')  # <-- THÊM DÒNG NÀY

                # Lấy pipeline từ model dict
                electra_pipeline = text_models[1]["pipeline"]
                # Dự đoán và chuyển đổi kết quả
                elec_result = electra_pipeline(processed_text)
                prob_array = np.zeros(len(label_map))

                # Xử lý kết quả pipeline
                if elec_result and isinstance(elec_result, list):
                    for item in elec_result[0]:  # Lấy list đầu tiên
                        label_idx = int(item['label'].split('_')[-1])
                        prob_array[label_idx] = item['score']

                text_probs.append(prob_array)
            except Exception as e:
                print(f"Lỗi dự đoán Electra: {str(e)}")
                # Đảm bảo đã có label_map ở đây
                text_probs.append(np.zeros(len(label_map)))

        # Gọi logic voting
        final_label, top_1, top_3, image_scores, text_scores, combined = weighted_voting_with_scoring(
            image_probs,
            text_probs
        )

        # Lấy label mapping
        label_map = ModelLoader.get_label_mapping('text')

        # Chuẩn bị dữ liệu trả về
        return {
            "final_prediction": {
                "label_id": int(final_label),
                "label_name": label_map.get(final_label, 'Unknown')
            },
            "top_predictions": {
                "top_1": {
                    "label_id": int(top_1[0][0]),
                    "label_name": label_map.get(int(top_1[0][0]), 'Unknown'),
                    "score": round(top_1[0][1], 3)
                },
                "top_3": convert_top_predictions(top_3, label_map)
            },
            "model_scores": {
                "image_models": [convert_scores_to_names(s, label_map) for s in image_scores],
                "text_models": [convert_scores_to_names(s, label_map) for s in text_scores],
                "combined": convert_scores_to_names(combined, label_map)
            },
            "label_mapping": label_map
        }

    except Exception as e:
        return {"error": f"Lỗi dự đoán: {str(e)}"}


def predict_text_fasttext(model, text: str):
    """Dự đoán bằng FastText và trả về mảng xác suất"""
    try:
        # Lấy số lượng class từ label mapping
        label_map = ModelLoader.get_label_mapping('text')
        num_classes = len(label_map)

        # Khởi tạo mảng xác suất với giá trị mặc định
        prob_array = np.zeros(num_classes)

        # Dự đoán từ model FastText
        labels, probs = model.predict(text, k=num_classes)

        # Tạo reverse map từ label text sang index
        reverse_map = {v.lower(): k for k, v in label_map.items()}

        for label, prob in zip(labels, probs):
            clean_label = label.replace('__label__', '').lower()
            if clean_label in reverse_map:
                idx = reverse_map[clean_label]
                prob_array[idx] = prob

        # Chuẩn hóa xác suất về tổng bằng 1
        if prob_array.sum() > 0:
            prob_array = prob_array / prob_array.sum()

        return prob_array
    except Exception as e:
        print(f"Lỗi trong predict_text_fasttext: {str(e)}")
        return np.zeros(num_classes)


def predict_text_electra(pipeline, text: str, num_classes: int):
    """Xử lý dự đoán cho Electra với đầu ra đầy đủ"""
    try:
        if not pipeline or not text:
            return np.zeros(num_classes)

        # Dự đoán và nhận tất cả các điểm số
        results = pipeline(text)

        # Khởi tạo mảng xác suất
        prob_array = np.zeros(num_classes)

        # Điền giá trị vào mảng
        if results and isinstance(results, list):
            for item in results[0]:  # Lấy batch đầu tiên
                label_idx = int(item['label'].split('_')[-1])
                if 0 <= label_idx < num_classes:
                    prob_array[label_idx] = item['score']

        return prob_array

    except Exception as e:
        print(f"Electra prediction failed: {str(e)}")
        return np.zeros(num_classes)


def calculate_model_scores(predictions, model_type='image'):
    """
    predictions: List of top 3 predicted labels
    model_type: 'image' or 'text' to determine initial point allocation
    """
    # Khởi tạo điểm ban đầu
    initial_points = {
        'image': [3, 2, 1],
        'text': [3, 2, 1]
    }

    def calculate_label_scores(top_labels, probabilities):

        label_scores = {}

        # Lấy điểm khởi tạo
        initial = initial_points[model_type]

        print("\n=== Bắt đầu tính điểm cho nhãn ===")
        print(f"Top labels: {top_labels}")
        print(f"Top probabilities: {probabilities}")

        # Chỉ xử lý nếu có ít nhất 3 nhãn
        if len(top_labels) >= 3:
            p1, p2, p3 = probabilities[:3]  # Xác suất
            labels = top_labels[:3]  # Nhãn

            # Xét chênh lệch tương đối so với trung bình
            relative_avg_12 = abs(p1 - p2) / ((p1 + p2) / 2)
            relative_avg_13 = abs(p1 - p3) / ((p1 + p3) / 2)
            relative_avg_23 = abs(p2 - p3) / ((p2 + p3) / 2)
            # Kiểm tra từng cặp
            cond_12 = abs(p1 - p2) / ((p1 + p2) / 2) <= 0.1
            cond_13 = abs(p1 - p3) / ((p1 + p3) / 2) <= 0.1
            cond_23 = abs(p2 - p3) / ((p2 + p3) / 2) <= 0.1

            print(
                f"Điều kiện (1-2): {cond_12}, (1-3): {cond_13}, (2-3): {cond_23}")

            # Xác định cặp có khoảng cách nhỏ nhất
            distances = {"12": relative_avg_12,
                         "13": relative_avg_13, "23": relative_avg_23}
            # Lấy cặp có khoảng cách nhỏ nhất
            closest_pair = min(distances, key=distances.get)
            print(f"Các khoảng cách: {distances}")
            print(f"Cặp gần nhất: {closest_pair}")

            if cond_12 and cond_13 and cond_23:
                # Nếu cả 3 cặp đều thỏa mãn, chia đều điểm
                avg_score = sum(initial[:3]) / 3
                for label in labels:
                    label_scores[label] = avg_score

                print(f"Chia đều điểm: {label_scores}")

            elif distances[closest_pair] <= 0.1:
                if closest_pair == "12":
                    avg_score = (initial[0] + initial[1]) / 2
                    label_scores[labels[0]] = avg_score
                    label_scores[labels[1]] = avg_score
                    label_scores[labels[2]] = initial[2]  # Giữ nguyên nhãn 3
                elif closest_pair == "13":
                    avg_score = (initial[0] + initial[2]) / 2
                    label_scores[labels[0]] = avg_score
                    label_scores[labels[2]] = avg_score
                    label_scores[labels[1]] = initial[1]  # Giữ nguyên nhãn 2
                elif closest_pair == "23":
                    avg_score = (initial[1] + initial[2]) / 2
                    label_scores[labels[1]] = avg_score
                    label_scores[labels[2]] = avg_score
                    label_scores[labels[0]] = initial[0]  # Giữ nguyên nhãn 1

                print(f"Điểm sau khi điều chỉnh: {label_scores}")

            else:
                # Không có cặp nào thỏa mãn, giữ nguyên điểm gốc
                for i, label in enumerate(labels):
                    label_scores[label] = initial[i]

                print(f"Điểm giữ nguyên: {label_scores}")

        # Xử lý các nhãn còn lại nếu có
        # for i in range(3, len(top_labels)):
        #     label_scores[top_labels[i]] = initial[i]

        return label_scores

    # sắp xếp các nhãn theo xác suất giảm dần
    def process_model_prediction(prediction):
        # Sắp xếp chỉ số của các nhãn theo thứ tự giảm dần của xác suất
        sorted_indices = np.argsort(prediction)[::-1]
        top_labels = sorted_indices[:3]
        top_probs = prediction[top_labels]  # Lấy xác suất tương ứng

        print("\n==== Sắp xếp các nhãn theo xác suất giảm dần ====")
        print(f"Prediction vector: {prediction}")
        print(f"Sorted labels: {top_labels}")
        print(f"Sorted probabilities: {top_probs}")

        return calculate_label_scores(top_labels, top_probs)

    return process_model_prediction

# Tổng hợp dự đoán từ hai mô hình ảnh và văn bản bằng cách tính điểm số dựa trên trọng số cho từng mô hình


def weighted_voting_with_scoring(prob_images, prob_texts, top_k=3):
    """
    Args:
    prob_images: mảng xác suất dự đoán từ 2 model ảnh
    prob_texts:mảng xác suất dự đoán từ 2 model text
    top_k: Số lượng dự đoán hàng đầu để xem xét

    Returns:
    Final label, top 1 and 3 predictions with scores
    """
    print("\n--- Initial Inputs ---")
    print("Probability from Image Model:", prob_images)
    print("Probability from Text Model:", prob_texts)
    # Tính toán điểm số từ mảng xác suất
    image_scoring_func = calculate_model_scores(
        prob_images, model_type='image')
    text_scoring_func = calculate_model_scores(prob_texts, model_type='text')

    # Calculate scores for each model
    image_model_scores = []
    for prob in prob_images:
        model_label_scores = image_scoring_func(prob)
        image_model_scores.append(model_label_scores)

    text_model_scores = []
    for prob in prob_texts:
        model_label_scores = text_scoring_func(prob)
        text_model_scores.append(model_label_scores)

    # Combine scores with model type weights
    combined_scores = {}

    # Combine image model scores (40% weight)
    for model_scores in image_model_scores:
        for label, score in model_scores.items():
            if label not in combined_scores:
                combined_scores[label] = 0
            combined_scores[label] += score * 0.4

    # Combine text model scores (60% weight)
    for model_scores in text_model_scores:
        for label, score in model_scores.items():
            if label not in combined_scores:
                combined_scores[label] = 0
            combined_scores[label] += score * 0.6

    # Làm tròn tất cả các giá trị trong combined_scores
    rounded_scores = {key: round(value, 3)
                      for key, value in combined_scores.items()}

    # Sort labels by combined scores in descending order
    sorted_labels = sorted(rounded_scores.items(),
                           key=lambda x: x[1], reverse=True)

    # Return top predictions
    top_1_prediction = sorted_labels[:1]
    top_3_predictions = sorted_labels[:top_k]

    # Final label is the one with highest score
    final_label = top_1_prediction[0][0]

    print("\n--- Scoring Details ---")
    print("Image Model Scores:", image_model_scores)
    print("Text Model Scores:", text_model_scores)
    print("Combined Scores:", rounded_scores)
    print(
        f"Final Label: {final_label}, Top 1 Predictions: {top_1_prediction}, Top 3 Predictions: {top_3_predictions}")

    return (
        final_label,
        top_1_prediction,
        top_3_predictions,
        image_model_scores,
        text_model_scores,
        combined_scores
    )
