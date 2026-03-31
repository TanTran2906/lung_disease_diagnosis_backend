import numpy as np
import cv2
import torch
from torchvision import transforms
from models.model_loader import ModelLoader


async def read_image(file):
    """Đọc và giải mã ảnh từ file binary"""
    try:
        if hasattr(file, "read"):  # là UploadFile
            content = await file.read()
        else:  # là bytes
            content = file
        image = np.frombuffer(content, np.uint8)
        # Giải mã thành ảnh màu BGR
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if image is None:
            return None, {"error": "Không thể đọc ảnh, có thể file bị hỏng!"}
        return image, None
    except Exception as e:
        return None, {"error": f"Lỗi đọc ảnh: {str(e)}"}


def preprocess_image(image, model_name):
    """Tiền xử lý ảnh theo từng model"""
    try:
        if model_name == "Lenet":
            # Resize về 128x128 và giữ nguyên 3 kênh màu
            image = cv2.resize(image, (128, 128))
            # Chuẩn hóa giá trị pixel
            image = image.astype(np.float32) / 255.0
            # Thêm batch dimension và đảm bảo 3 kênh màu
            image = np.expand_dims(image, axis=0)

        elif model_name == "ViT":
            # Convert BGR to RGB (OpenCV loads as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Resize to 224x224 which is standard for ViT
            image = cv2.resize(image, (224, 224))
            # Use the same transforms as during training
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                     0.229, 0.224, 0.225])
            ])
            # Add batch dimension: (1, 3, 224, 224)
            image = transform(image).unsqueeze(0)

        else:
            image = cv2.resize(image, (224, 224)) / 255.0
            image = np.expand_dims(image, axis=0)  # Shape: (1, 224, 224, 3)

        return image, None
    except Exception as e:
        return None, {"error": f"Lỗi tiền xử lý ảnh: {str(e)}"}


async def predict_image(model_name, file):
    """Dự đoán ảnh với model"""
    image, error = await read_image(file)
    if error:
        return error

    image, error = preprocess_image(image, model_name)
    if error:
        return error

    model = ModelLoader.load_image_model(model_name)
    if not model:
        return {"error": f"Model '{model_name}' không tồn tại!"}

    # Lấy ánh xạ nhãn
    label_map = ModelLoader.get_label_mapping('image')

    try:
        if model_name == "ViT":
            with torch.no_grad():
                # For timm ViT models
                output = model(image)  # Should return logits directly

                # Handle different output types
                if isinstance(output, tuple):
                    # Some models return multiple outputs (logits, features, etc.)
                    logits = output[0]
                else:
                    # Most models return logits directly
                    logits = output

                # Get class prediction and confidence
                probabilities = torch.softmax(logits, dim=1)
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
                    "confidence": round(confidence.item(), 4),
                }
        else:
            prediction = model.predict(image)
            confidence = float(np.max(prediction))
            prediction = int(np.argmax(prediction))

            prediction_idx = prediction
            # Kiểm tra chỉ số hợp lệ
            if prediction_idx not in label_map:
                return {
                    "error": f"Chỉ số dự đoán {prediction_idx} không hợp lệ"
                }

            return {
                "model": model_name,
                "prediction": prediction_idx,
                "label": label_map[prediction_idx],
                "confidence": round(confidence, 4),
            }
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        print(f"Error in prediction: {str(e)}\n{traceback_str}")
        return {"error": f"Lỗi khi dự đoán: {str(e)}"}
