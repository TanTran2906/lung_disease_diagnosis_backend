# services/prediction_handler.py
from typing import Dict, Any, List
from services import text_service, image_service, multimodal_service
from models.model_loader import ModelLoader
import numpy as np
import json
from io import BytesIO


async def handle_prediction(
    selected_models: List[str],
    image_content: bytes,
    text_content: str
) -> Dict[str, Any]:
    """Xử lý dự đoán tổng hợp cho nhiều loại model"""
    results = []
    label_mapping = None
    # Xử lý ['["DenseNet121", "FastText"]']
    if len(selected_models) == 1 and isinstance(selected_models[0], str) and selected_models[0].startswith('['):
        try:
            selected_models = json.loads(selected_models[0])
        except json.JSONDecodeError:
            return {"error": "Invalid model selection format"}

    # Validate input
    if not image_content or not text_content:
        return {"error": "Missing image/text content"}

    # Phân loại model
    model_types = {
        'text': ["FastText", "Electra", "DistillBERT"],
        'image': ["ViT", "Lenet", "MobileNet", "DenseNet121", "DenseNet169"],
        'multimodal': ["resnet_sbert", "mobilenet_sbert",
                       "densenet121_sbert", "densenet169_sbert"]
    }

    try:
        # Xử lý từng model được chọn
        for model_name in selected_models:
            model_type = next(
                (k for k, v in model_types.items() if model_name in v),
                None
            )

            if not model_type:
                continue

            # Gọi service tương ứng
            if model_type == 'text':
                result = await text_service.predict_text(
                    model_name=model_name,
                    text=text_content
                )
            elif model_type == 'image':
                # image_file = BytesIO(image_content)
                result = await image_service.predict_image(
                    model_name=model_name,
                    file=image_content
                )
            elif model_type == 'multimodal':
                result = await multimodal_service.predict_multimodal(
                    model_name=model_name,
                    image_file=image_content,
                    text_input=text_content
                )

            # Lưu label mapping từ model đầu tiên
            if not label_mapping:
                label_mapping = ModelLoader.get_label_mapping(model_type)

            results.append(result)

        # Xử lý voting nếu có nhiều model
        final_prediction = _calculate_final_prediction(results)

        return {
            "predictions": results,
            "final_prediction": final_prediction,
            "label_mapping": label_mapping
        }

    except Exception as e:
        return {"error": f"Prediction error: {str(e)}"}


def _calculate_final_prediction(predictions: List[Dict]) -> Dict:
    """Tính toán kết quả cuối cùng từ nhiều model"""
    # Logic voting/weighted average
    confidences = {}
    for pred in predictions:
        label = pred.get('label')
        confidence = pred.get('confidence', 0)
        if label in confidences:
            confidences[label] += confidence
        else:
            confidences[label] = confidence

    if not confidences:
        return {}

    max_label = max(confidences, key=confidences.get)
    return {
        "label": max_label,
        "confidence": round(confidences[max_label]/len(predictions), 4),
        "total_models": len(predictions)
    }
