from fastapi import APIRouter, UploadFile, File, Form, Request, HTTPException
from fastapi.responses import JSONResponse
import logging
from services.rag_service import MedicalRAG, LLMProcessor

router = APIRouter()


@router.post("/diagnose")
async def rag_diagnose(
    request: Request,
    text: str = Form(None),
    file: UploadFile = File(None),
    top_k: int = Form(5, gt=0, le=10),
    llm_choice: str = Form("gemini")
):
    # Validate input
    if not text and not file:
        raise HTTPException(
            status_code=400,
            detail="Vui lòng cung cấp text hoặc file txt"
        )

    # Read input
    input_text = text or (await file.read()).decode("utf-8")

    # Get RAG system
    rag_system = request.app.state.rag_system
    if not rag_system:
        logging.error("Hệ thống RAG chưa được khởi tạo")
        raise HTTPException(
            status_code=500,
            detail="Hệ thống đang khởi tạo..."
        )

    try:
        # Retrieve similar cases
        similar_cases = rag_system.retrieve_similar_cases(input_text, top_k)

        # Generate diagnosis
        llm_processor = LLMProcessor(llm_choice=llm_choice)
        prompt = llm_processor.create_prompt(similar_cases, input_text)
        diagnosis = llm_processor.generate_diagnosis(prompt)

        # Format response
        return {
            "diagnosis": diagnosis,
            "similar_cases": [
                {
                    "label": case["label"],
                    "similarity": round(case["similarity_score"], 3),
                    # Truncate long content
                    "content": case["content"][:200] + "..."
                }
                for case in similar_cases
            ]
        }

    except Exception as e:
        logging.error(f"Lỗi xử lý: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi xử lý yêu cầu: {str(e)}"
        )
