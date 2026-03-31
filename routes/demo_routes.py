from fastapi import APIRouter, HTTPException, Form
from services.demo_service import DemoService
from services.prediction_handler import handle_prediction
from typing import Dict, List
router = APIRouter()
demo_service = DemoService()


@router.get("/diseases")
async def get_available_diseases():
    return {
        "diseases": demo_service.get_available_diseases(),
        "count_per_disease": 3  # Mỗi bệnh có 3 mẫu
    }


@router.get("/samples/{disease}")
async def get_disease_samples(disease: str):
    samples = demo_service.get_samples_by_disease(disease)
    if not samples["images"] and not samples["texts"]:
        raise HTTPException(status_code=404, detail="Disease not found")

    return {
        "disease": disease,
        "images": [{"id": img["id"], "filename": img["filename"]} for img in samples["images"]],
        "texts": [{"id": txt["id"], "filename": txt["filename"]} for txt in samples["texts"]]
    }


@router.post("/predict")
async def demo_predict(
    image_sample_id: str = Form(...),
    text_sample_id: str = Form(...),
    selected_models: List[str] = Form(...)
):
    # Validate sample IDs
    if not all(["_img_" in image_sample_id, "_txt_" in text_sample_id]):
        raise HTTPException(
            status_code=400,
            detail="Invalid sample ID format. Expected: <disease>_img_<num>"
        )

    # Lấy nội dung
    image_content = demo_service.get_sample_content(image_sample_id)
    text_content = demo_service.get_sample_content(text_sample_id)

    if not image_content or not text_content:
        raise HTTPException(
            status_code=404,
            detail="Sample content not found"
        )

    print(selected_models)

    # Xử lý dự đoán
    return await handle_prediction(selected_models, image_content, text_content)
