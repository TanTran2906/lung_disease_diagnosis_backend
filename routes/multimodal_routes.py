from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from services.multimodal_service import predict_multimodal

router = APIRouter()


@router.post("/predict-multimodal/")
async def predict_multimodal_api(
    model_name: str = Form(...),
    image: UploadFile = File(...),
    text: str = Form(None),
    text_file: UploadFile = File(None)
):
    """
    API dự đoán đa phương thức
    - model_name: Tên mô hình (resnet_sbert, mobilenet_sbert,...)
    - image: File ảnh X-quang
    - text: Mô tả triệu chứng bằng văn bản (nhập trực tiếp)
    - text_file: File text mô tả triệu chứng (upload file)
    """
    try:
        # Xử lý logic đọc text
        final_text = await process_text_input(text, text_file)
        result = await predict_multimodal(model_name, image, final_text)

        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống: {str(e)}")


async def process_text_input(text: str, text_file: UploadFile):
    """Xử lý đầu vào text từ cả hai nguồn: text trực tiếp và file upload"""
    # Kiểm tra cả hai đều null
    if not text and not text_file:
        raise ValueError("Cần cung cấp text hoặc file text mô tả triệu chứng")

    # Ưu tiên text trực tiếp nếu có cả hai
    if text:
        return text.strip()

    # Xử lý file upload
    if text_file.content_type not in ["text/plain", "application/octet-stream"]:
        raise ValueError("File text phải có định dạng .txt")

    # Đọc nội dung file
    content = await text_file.read()
    try:
        return content.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise ValueError("File text phải được mã hóa UTF-8")
