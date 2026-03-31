from fastapi import APIRouter, UploadFile, File, Form
from services.maxsat_service import MaxSATService

router = APIRouter()
maxsat_service = MaxSATService()


@router.post("/diagnose")
async def maxsat_diagnose(
    text: str = Form(None),
    file: UploadFile = File(None)
):
    """
    Chẩn đoán bệnh kết hợp MaxSAT và FastText
    - text: Văn bản mô tả triệu chứng
    - file: File text chứa triệu chứng (nếu có)
    """
    return await maxsat_service.get_diagnosis(text, file)
