from models.model_loader import ModelLoader
import re
from underthesea import word_tokenize
import unidecode
import torch

# Tập hợp stopwords để loại bỏ trong quá trình tiền xử lý
STOPWORDS = {"là", "và", "có", "của", "cho",
             "với", "đã", "này", "một", "các", "về", "ở"}


def preprocess_text(text: str) -> str:
    """Tiền xử lý văn bản"""
    text = text.lower().strip()  # Chuyển thành chữ thường & loại bỏ khoảng trắng đầu/cuối
    text = unidecode.unidecode(text)  # Loại bỏ dấu tiếng Việt
    text = re.sub(r'\d+', '', text)  # Loại bỏ số
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ ký tự đặc biệt
    text = word_tokenize(text, format="text")  # Tách từ dùng Underthesea
    # Loại bỏ stopwords
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text


async def read_file_content(file):
    """Đọc nội dung từ file nếu có"""
    if file:
        try:
            content = await file.read()
            # Thử nhiều encoding khác nhau
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    return content.decode(encoding).strip()
                except UnicodeDecodeError:
                    continue
            # Nếu tất cả encoding đều thất bại, trả về lỗi
            return None
        except Exception as e:
            print(f"Lỗi đọc file: {str(e)}")
            return None
    return None


async def predict_text(model_name: str, text: str, file=None):
    """Dự đoán văn bản với mô hình được chọn"""
    try:
        file_text = await read_file_content(file)
        text = file_text or text  # Dùng nội dung file nếu có, nếu không giữ nguyên text

        if not text:
            return {"error": "Không có nội dung văn bản để dự đoán!"}

        # print(f"Input text before preprocessing: {text}")
        text = preprocess_text(text)
        # print(f"Input text after preprocessing: {text}")

        model_info = ModelLoader.load_text_model(model_name)
        if not model_info:
            return {"error": f"Model '{model_name}' không tồn tại!"}

        print(f"Model loaded: {model_name}")

        # Lấy ánh xạ nhãn
        label_map = ModelLoader.get_label_mapping('text')

        if model_name == "FastText":
            # prediction, confidence = model_info.predict(text)
            # confidence = max(confidence)
            # Dự đoán và xử lý kết quả
            predictions, confidences = model_info.predict(text)
            raw_label = predictions[0]
            confidence = confidences[0]

            # Trích xuất tên nhãn (bỏ prefix __label__)
            label_name = raw_label.replace("__label__", "").strip()

            # Debug: In thông tin nhãn
            print(f"FastText raw label: {raw_label}")
            print(f"Cleaned label: {label_name}")

            # Lấy reverse mapping
            reverse_label_map = ModelLoader.get_reverse_label_mapping('text')

            # Debug: In ánh xạ
            print(f"Reverse label map: {reverse_label_map}")

            # Chuyển đổi sang chỉ số
            prediction_idx = reverse_label_map.get(label_name)

            if prediction_idx is None:
                return {
                    "error": f"Không tìm thấy nhãn '{label_name}' trong ánh xạ. Danh sách nhãn hợp lệ: {list(reverse_label_map.keys())}"
                }

        else:
            tokenizer = model_info["tokenizer"]
            model = model_info["model"]
            inputs = tokenizer(text, return_tensors="pt" if isinstance(
                model, torch.nn.Module) else "tf", padding=True, truncation=True)

            with torch.no_grad():
                outputs = model(**inputs)

            if isinstance(model, torch.nn.Module):
                prediction_idx = int(outputs.logits.argmax().item())
                confidence = float(outputs.logits.softmax(dim=1).max().item())
            else:
                import tensorflow as tf
                prediction_idx = int(
                    tf.argmax(outputs.logits, axis=1).numpy()[0])
                confidence = float(tf.nn.softmax(
                    outputs.logits, axis=1).numpy().max())

        return {
            "model": model_name,
            "prediction": prediction_idx,
            "label": label_map[prediction_idx],
            "confidence": round(confidence, 4),
        }
    except Exception as e:
        import traceback
        print(f"Error in predict_text: {str(e)}")
        print(traceback.format_exc())
        return {"error": f"Lỗi khi dự đoán: {str(e)}"}
