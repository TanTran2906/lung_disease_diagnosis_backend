import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from models.model_loader import ModelLoader
from sentence_transformers import SentenceTransformer


async def process_multimodal_input(image_file, text_input, model_name):
    # Validate text input
    if not text_input or len(text_input.strip()) == 0:
        text_input = "Không có mô tả triệu chứng"
    # Xử lý ảnh
    if hasattr(image_file, "read"):  # là UploadFile
        image_content = await image_file.read()
    else:  # là bytes
        image_content = image_file
    image = np.frombuffer(image_content, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Tiền xử lý ảnh
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Xử lý text
    model_data = ModelLoader.load_multimodal_model(model_name)
    text_model = model_data["text_model"]

    # Chuẩn hóa text input
    text = text_input.strip()
    if len(text) == 0:
        text = "Không có mô tả triệu chứng"

    text_embedding = text_model.encode(
        text, convert_to_tensor=True).unsqueeze(0)

    return image_tensor, text_embedding


async def predict_multimodal(model_name, image_file, text_input):
    try:
        # Load model
        model_data = ModelLoader.load_multimodal_model(model_name)
        if not model_data:
            return {"error": f"Model {model_name} không tồn tại"}

        # Lấy ánh xạ nhãn
        label_map = ModelLoader.get_label_mapping('multimodal')

        model = model_data["model"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Chuẩn bị dữ liệu
        image_tensor, text_embedding = await process_multimodal_input(image_file, text_input, model_name)

        # Dự đoán
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            text_embedding = text_embedding.to(device)

            outputs = model(image_tensor, text_embedding)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)

        prediction_idx = prediction.item()
        # Kiểm tra chỉ số hợp lệ
        if prediction_idx not in label_map:
            return {
                "error": f"Chỉ số dự đoán {prediction_idx} không hợp lệ"
            }

        return {
            "model": model_name,
            "prediction": prediction_idx,
            "label": label_map[prediction_idx],
            "confidence": round(confidence.item(), 4)
        }

    except Exception as e:
        return {"error": f"Lỗi dự đoán: {str(e)}"}
