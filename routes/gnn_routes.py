
from fastapi import APIRouter, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
import tempfile
from typing import Optional
from services.gnn_service import predict_gnn
router = APIRouter()


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    text: str = Form(...)
):
    try:
        # Lưu file upload vào thư mục tạm
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

        # Gọi hàm predict với đường dẫn file
        result = predict_gnn(temp_path, text)

        # Xóa file tạm
        os.unlink(temp_path)

        return result

    except Exception as e:
        # Xóa file tạm nếu có lỗi
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
