from fastapi import APIRouter, UploadFile, File, Form
from services.text_service import predict_text

router = APIRouter()


@router.post("/predict-text/")
async def predict_text_api(model_name: str = Form(...), text: str = Form(None), file: UploadFile = File(None)):
    """
    API dự đoán văn bản với mô hình đã chọn.
    - model_name: Tên mô hình (FastText, Electra, DistillBERT) => Nhận từ params
    - text: Văn bản cần dự đoán. => Nhận từ form-data
    - file: Nếu có file, đọc nội dung file để dự đoán.
    """
    # print(f"Nội dung text: {text}")
    result = await predict_text(model_name, text, file)
    return result
