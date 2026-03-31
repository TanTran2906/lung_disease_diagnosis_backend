# gnn_service.py
import torch
import numpy as np
import cv2
from PIL import Image
import os
from models.model_loader import ModelLoader
import torch.nn.functional as F
import pandas as pd
import pickle
import os


class ReferenceDataset:
    def __init__(self, samples_per_class=5):
        self.samples_per_class = samples_per_class
        self.data = None
        self.reference_features = None
        self.reference_path = "D:/Data_Store/LV_KHMT/Data/backend/models/gnn/reference_data.pkl"

    def load_or_create(self, force_create=False):
        """Load existing reference data or create new with better sampling"""
        if os.path.exists(self.reference_path) and not force_create:
            print("Loading existing reference data...")
            with open(self.reference_path, 'rb') as f:
                data = pickle.load(f)
                self.data = data
                return data

        print("Creating new reference dataset...")
        # Load training data CSV
        csv_path = "D:/Data_Store/LV_KHMT/Data/clinical_train.csv"
        df = pd.read_csv(csv_path)

        # Get unique labels
        labels = df['Nhãn bệnh'].unique()

        # For each label, select diverse samples
        reference_samples = []

        for label in labels:
            class_df = df[df['Nhãn bệnh'] == label]

            # Chọn mẫu đa dạng bằng cách sử dụng các chiến lược khác nhau
            if len(class_df) <= self.samples_per_class:
                # Nếu số mẫu ít hơn yêu cầu, lấy tất cả
                selected_samples = class_df
            else:
                # Chọn 1/3 từ đầu tập dữ liệu
                head_count = max(1, self.samples_per_class // 3)
                head_samples = class_df.head(head_count)

                # Chọn 1/3 từ cuối tập dữ liệu
                tail_count = max(1, self.samples_per_class // 3)
                tail_samples = class_df.tail(tail_count)

                # Chọn phần còn lại ngẫu nhiên
                middle_count = self.samples_per_class - head_count - tail_count
                if middle_count > 0:
                    middle_samples = class_df.iloc[head_count:-tail_count].sample(
                        min(middle_count, len(class_df) - head_count - tail_count)
                    )
                else:
                    middle_samples = pd.DataFrame()

                # Kết hợp các mẫu đã chọn
                selected_samples = pd.concat(
                    [head_samples, middle_samples, tail_samples])

            reference_samples.append(selected_samples)

        # Combine into single dataframe
        reference_df = pd.concat(reference_samples)

        # Save to disk
        with open(self.reference_path, 'wb') as f:
            pickle.dump(reference_df, f)

        self.data = reference_df
        return reference_df

    def precompute_features(self):
        """Precompute image and text features for reference samples with improved processing"""
        if self.data is None:
            self.load_or_create()

        # Get GNN components
        gnn_components = ModelLoader.load_gnn_model()
        extractor = gnn_components["extractor"]
        text_model = gnn_components["text_model"]
        transform = gnn_components["transform"]
        device = gnn_components["device"]

        # Where to store precomputed features
        features_path = "D:/Data_Store/LV_KHMT/Data/backend/models/gnn/reference_features.pkl"

        # Check if precomputed features exist
        if os.path.exists(features_path):
            with open(features_path, 'rb') as f:
                self.reference_features = pickle.load(f)
                return self.reference_features

        # Process each sample
        image_dir = "D:/Data_Store/LV_KHMT/Data/img_xray/train"
        reference_features = []

        for _, row in self.data.iterrows():
            try:
                # Get image path
                label_str = row['Nhãn bệnh']
                img_name = row['Tên tệp']
                image_path = os.path.join(
                    image_dir, label_str, f"{img_name}.jpg")

                # Process image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Không thể đọc hình ảnh: {image_path}")
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = transform(image).unsqueeze(0).to(device)

                # Extract image features
                with torch.no_grad():
                    img_features = extractor(image)

                # Get text embedding
                text = row['Nội dung']
                text_embedding = text_model.encode(
                    text, convert_to_tensor=True).to(device)

                # Normalize features for consistency
                img_features = F.normalize(img_features, p=2, dim=1)
                text_embedding = F.normalize(text_embedding, p=2, dim=0)

                # Store features
                features = {
                    'img_features': img_features.cpu(),
                    'text_embedding': text_embedding.cpu(),
                    'label': row['Nhãn bệnh'],
                    'label_idx': ModelLoader.get_reverse_label_mapping("GNN").get(row['Nhãn bệnh'], -1),
                    'text': text  # Lưu thêm nội dung văn bản để tiện tham khảo
                }

                reference_features.append(features)

            except Exception as e:
                print(f"Error processing reference sample: {str(e)}")
                continue

        # Đảm bảo cân bằng giữa các lớp
        label_counts = {}
        balanced_features = []

        for feature in reference_features:
            label = feature['label']
            if label not in label_counts:
                label_counts[label] = 0

            if label_counts[label] < self.samples_per_class:
                balanced_features.append(feature)
                label_counts[label] += 1

        # Save to disk
        with open(features_path, 'wb') as f:
            pickle.dump(balanced_features, f)

        self.reference_features = balanced_features
        return balanced_features


def construct_inference_graph(new_img_features, new_text_embedding, device, similarity_threshold=0.6):
    """
    Construct a graph for inference with improved connections for better class separation
    """
    # Get reference data
    reference_dataset = ReferenceDataset(
        samples_per_class=5)  # Tăng số lượng mẫu tham chiếu
    reference_features = reference_dataset.precompute_features()

    # Get GNN components
    gnn_components = ModelLoader._loaded_models["GNN"]
    model = gnn_components["model"]

    # Extract number of reference samples
    ref_count = len(reference_features)

    # Prepare all features
    reshaped_img_features = []
    reshaped_text_embeddings = []
    all_labels = []

    # Add reference samples
    for ref in reference_features:
        img_feat = ref['img_features'].to(device)
        text_emb = ref['text_embedding'].to(device)

        if len(img_feat.shape) == 1:
            img_feat = img_feat.unsqueeze(0)
        if len(text_emb.shape) == 1:
            text_emb = text_emb.unsqueeze(0)

        reshaped_img_features.append(img_feat)
        reshaped_text_embeddings.append(text_emb)
        all_labels.append(ref['label_idx'])

    # Process new sample
    if len(new_img_features.shape) == 1:
        new_img_features = new_img_features.unsqueeze(0)
    if len(new_text_embedding.shape) == 1:
        new_text_embedding = new_text_embedding.unsqueeze(0)

    # Add new sample
    reshaped_img_features.append(new_img_features)
    reshaped_text_embeddings.append(new_text_embedding)

    try:
        # Convert to tensors
        all_img_features = torch.cat(reshaped_img_features, dim=0)
        all_text_embeddings = torch.cat(reshaped_text_embeddings, dim=0)
        all_labels = torch.tensor(all_labels, device=device)

        # Normalize features
        img_features = torch.nn.functional.normalize(
            all_img_features, p=2, dim=1)
        text_features = torch.nn.functional.normalize(
            all_text_embeddings, p=2, dim=1)

        # Create node features
        img_features = model.image_fc(img_features)
        text_features = model.text_fc(text_features)

        # Total number of samples (reference + new)
        total_samples = ref_count + 1

        # Interleave image and text features for the graph
        x = torch.zeros(
            (total_samples * 2, img_features.size(1)), device=device)
        x[0::2] = img_features  # Even indices for images
        x[1::2] = text_features  # Odd indices for texts

        # Initialize edge list
        edge_list = []

        # 1. Connect each image with its corresponding text (bidirectional)
        for i in range(total_samples):
            img_idx = i * 2
            text_idx = i * 2 + 1
            edge_list.append([img_idx, text_idx])  # Image -> Text
            edge_list.append([text_idx, img_idx])  # Text -> Image

        # 2. Connect samples with the same label from reference dataset
        if len(all_labels) > 0:  # Chỉ áp dụng cho dữ liệu tham chiếu
            label_dict = {}
            for i in range(ref_count):
                label_item = all_labels[i].item()
                if label_item not in label_dict:
                    label_dict[label_item] = []
                label_dict[label_item].append(i)

            # Connect nodes with the same label
            for label, indices in label_dict.items():
                if len(indices) > 1:
                    for i in range(len(indices)):
                        for j in range(i+1, len(indices)):
                            # Kết nối có trọng số cao hơn cho các nút cùng nhãn
                            src_text_idx = indices[i] * 2 + 1
                            dst_text_idx = indices[j] * 2 + 1
                            edge_list.append([src_text_idx, dst_text_idx])
                            edge_list.append([dst_text_idx, src_text_idx])

                            src_img_idx = indices[i] * 2
                            dst_img_idx = indices[j] * 2
                            edge_list.append([src_img_idx, dst_img_idx])
                            edge_list.append([dst_img_idx, src_img_idx])

        # 3. Connect new sample to reference samples based on similarity
        # Tính toán tương đồng giữa mẫu mới và các mẫu tham chiếu
        new_img_idx = ref_count * 2
        new_text_idx = ref_count * 2 + 1

        # Lấy đặc trưng mẫu mới
        new_img_feature = img_features[ref_count].unsqueeze(0)
        new_text_feature = text_features[ref_count].unsqueeze(0)

        # Tính toán tương đồng với các mẫu tham chiếu
        ref_img_features = img_features[:ref_count]
        ref_text_features = text_features[:ref_count]
        # Tính toán ma trận tương đồng
        img_similarities = torch.mm(
            new_img_feature, ref_img_features.t()).squeeze()
        text_similarities = torch.mm(
            new_text_feature, ref_text_features.t()).squeeze()

        # Kết nối với các mẫu tham chiếu có độ tương đồng cao
        img_connections = torch.where(
            img_similarities > similarity_threshold)[0]
        text_connections = torch.where(
            text_similarities > similarity_threshold)[0]

        if ref_count == 0:
            # Xử lý trường hợp không có mẫu tham chiếu
            x = torch.cat([new_img_features, new_text_embedding],
                          dim=0).unsqueeze(0)
            edge_index = torch.tensor([], device=device)
            return x, edge_index, 1  # Chỉ có 2 nút (ảnh + văn bản mới)

        # Trong phần kết nối hình ảnh
        for ref_idx in img_connections:
            if ref_idx < ref_count:  # Thêm điều kiện kiểm tra
                edge_list.append([new_img_idx, ref_idx * 2])
                edge_list.append([ref_idx * 2, new_img_idx])

        # Trong phần kết nối văn bản
        for ref_idx in text_connections:
            if ref_idx < ref_count:  # Thêm điều kiện kiểm tra
                edge_list.append([new_text_idx, ref_idx * 2 + 1])
                edge_list.append([ref_idx * 2 + 1, new_text_idx])

        # 4. Sử dụng thông tin từ cả hình ảnh và văn bản để cải thiện kết nối
        combined_similarities = (img_similarities + text_similarities) / 2
        combined_connections = torch.where(
            combined_similarities > (similarity_threshold - 0.1))[0]

        for ref_idx in combined_connections:
            if ref_idx < ref_count:  # Thêm điều kiện
                edge_list.append([new_text_idx, ref_idx * 2])
                edge_list.append([new_img_idx, ref_idx * 2 + 1])

        # 5. Chỉ kết nối với top-k mẫu tương tự nhất nếu có quá nhiều kết nối
        if len(edge_list) > 300:  # Giới hạn tổng số kết nối
            # Giữ các kết nối cơ bản (mẫu với đặc trưng tương ứng)
            basic_connections = []
            for i in range(total_samples):
                img_idx = i * 2
                text_idx = i * 2 + 1
                basic_connections.append([img_idx, text_idx])
                basic_connections.append([text_idx, img_idx])

            # Thêm top-k kết nối dựa trên tương đồng
            k = 100  # Số lượng kết nối bổ sung
            sorted_indices = torch.argsort(
                combined_similarities, descending=True)
            top_connections = []
            for idx in sorted_indices[:min(k, len(sorted_indices))]:
                ref_idx = idx.item()
                top_connections.append([new_text_idx, ref_idx * 2 + 1])
                top_connections.append([ref_idx * 2 + 1, new_text_idx])
                top_connections.append([new_img_idx, ref_idx * 2])
                top_connections.append([ref_idx * 2, new_img_idx])

            # Kết hợp kết nối cơ bản và top-k
            edge_list = basic_connections + top_connections

        # Convert edge list to tensor
        if edge_list:
            edge_index = torch.tensor(
                edge_list, device=device).t().contiguous()
        else:
            # Fallback to basic connections if no edges were created
            edge_index = torch.tensor(
                [[i*2, i*2+1] for i in range(total_samples)], device=device
            ).t().contiguous()
            edge_index = torch.cat(
                [edge_index, torch.flip(edge_index, [0])], dim=1)

        print(
            f"Total nodes: {x.shape[0]}, Max edge index: {edge_index.max() if edge_index.numel() > 0 else 0}")

        # Return the graph structure and the index of the new sample's text node
        new_sample_text_idx = (total_samples - 1) * 2 + 1
        return x, edge_index, new_sample_text_idx

    except Exception as e:
        print(f"Error during graph construction: {str(e)}")
        print(
            f"Shapes of image features: {[f.shape for f in reshaped_img_features]}")
        print(
            f"Shapes of text embeddings: {[f.shape for f in reshaped_text_embeddings]}")
        raise e


def predict_gnn(image_path, text):
    """
    Make a prediction using the GNN model with improved inference methodology
    """
    # Load GNN components
    gnn_components = ModelLoader.load_gnn_model()
    if not gnn_components:
        return {"error": "GNN model could not be loaded"}

    model = gnn_components["model"]
    extractor = gnn_components["extractor"]
    text_model = gnn_components["text_model"]
    transform = gnn_components["transform"]
    device = gnn_components["device"]

    try:
        # Process the image
        image = cv2.imread(image_path)
        if image is None:
            return {"error": "Image could not be read"}

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = transform(image).unsqueeze(0).to(device)

        # Extract image features
        with torch.no_grad():
            img_features = extractor(image)

        # Get text embedding
        text_embedding = text_model.encode(
            text, convert_to_tensor=True).to(device)

        # Apply ensemble approach with multiple thresholds
        thresholds = [0.5, 0.6, 0.7]
        predictions = []

        for threshold in thresholds:
            # Construct graph with different similarity thresholds
            x, edge_index, new_sample_idx = construct_inference_graph(
                img_features, text_embedding, device, similarity_threshold=threshold
            )

            # Make prediction
            with torch.no_grad():
                output = model(x, edge_index)
                # Extract only the prediction for our new sample
                print(
                    f"Output shape: {output.shape}, new_sample_idx: {new_sample_idx}")
                new_sample_batch_idx = output.shape[0] - 1
                sample_output = output[new_sample_batch_idx].unsqueeze(0)
                prediction = torch.softmax(sample_output, dim=1)
                predictions.append(prediction)

        # Combine predictions from different thresholds
        final_prediction = torch.mean(torch.stack(predictions), dim=0)

        # Convert to numpy for processing
        probs = final_prediction.cpu().numpy()[0]

        # Apply temperature scaling to sharpen or soften the distribution
        temperature = 0.8  # <1 sharpens predictions, >1 softens them
        probs = np.exp(np.log(probs) / temperature)
        probs = probs / np.sum(probs)  # Re-normalize

        # Get the top prediction and probability
        label_map = ModelLoader.get_label_mapping("GNN")
        predicted_class = int(np.argmax(probs))
        predicted_label = label_map[predicted_class]
        confidence = float(probs[predicted_class])

        # Get all class probabilities
        class_probabilities = {
            label_map[i]: float(probs[i]) for i in range(len(probs))
        }

        # Apply calibration to confidence scores to avoid overconfidence
        # Simple scaling factor for demonstration
        if confidence > 0.9:
            calibrated_confidence = 0.9 + (confidence - 0.9) * 0.5
        else:
            calibrated_confidence = confidence

        return {
            "predicted_class": predicted_class,
            "predicted_label": predicted_label,
            "confidence": calibrated_confidence,
            "probabilities": class_probabilities
        }

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
