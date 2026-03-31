from fastapi import APIRouter, UploadFile, File, Form
from services.image_service import predict_image

router = APIRouter()


@router.post("/predict-image/")
async def predict_image_api(model_name: str = Form(...), file: UploadFile = File(...)):
    """
    API dự đoán ảnh bằng mô hình đã chọn.
    - model_name: Tên mô hình (ViT, Lenet, MobileNet, DenseNet121, DenseNet169).
    - file: Ảnh upload.
    """
    print(f"Nhận request dự đoán với model: {model_name}")
    print(f"File input: {file.filename if file else None}")
    # Gọi hàm async đúng cách
    prediction = await predict_image(model_name, file)

    if prediction is None:
        return {"error": "Không thể dự đoán ảnh !"}

    return prediction  # Trả về luôn kết quả JSON từ `predict_image()`
