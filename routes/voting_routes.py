from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.voting_service import predict_voting

router = APIRouter()


@router.post("/predict-voting/")
async def predict_voting_api(
    image: UploadFile = File(...),
    text: str = Form(None),
    text_file: UploadFile = File(None)
):
    """
    API kết hợp dự đoán từ model ảnh và text
    - image: File ảnh X-quang (bắt buộc)
    - text: Mô tả triệu chứng (text trực tiếp)
    - text_file: File text mô tả triệu chứng (upload file)
    """
    try:
        result = await predict_voting(image, text, text_file)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")
