from routes.rag_routes import router as rag_router
from routes.maxsat_routes import router as maxsat_router
from routes.voting_routes import router as voting_router
from fastapi import FastAPI, APIRouter, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from routes.text_routes import router as text_router
from routes.image_routes import router as image_router
from routes.multimodal_routes import router as multimodal_router
from routes.demo_routes import router as demo_router

from models.model_loader import ModelLoader
from services.image_service import predict_image
import uvicorn
import asyncio
from services.rag_service import MedicalRAG
from concurrent.futures import ThreadPoolExecutor
from routes.gnn_routes import router as gnn_router
app = FastAPI()


@app.on_event("startup")
async def initialize_models():
    """Khởi tạo tất cả model khi server start"""

    # Sử dụng ThreadPool để tránh block event loop
    with ThreadPoolExecutor(max_workers=4) as executor:
        await asyncio.get_event_loop().run_in_executor(
            executor,
            ModelLoader.load_all_models
        )

    # Khởi tạo RAG
    rag_system = MedicalRAG()
    if rag_system.initialize():
        app.state.rag_system = rag_system
        print("Đã khởi tạo RAG thành công")
    else:
        print("Khởi tạo RAG thất bại")

# Các phần khởi tạo model khác...
# ModelLoader.load_all_models()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)


# Thêm router để quản lý API
# http://127.0.0.1:8000/image/predict-image/?model_name=ViT

# tags=["Text"] → Dùng để nhóm API trong tài liệu tự động của FastAPI (http://127.0.0.1:8000/docs)
app.include_router(text_router, prefix="/text", tags=["Text"])
app.include_router(image_router, prefix="/image", tags=["Image"])
app.include_router(multimodal_router, prefix="/multimodal",
                   tags=["Multimodal"])
app.include_router(voting_router, prefix="/voting", tags=["Voting"])
app.include_router(demo_router, prefix="/demo", tags=["Demo"])

app.include_router(maxsat_router, prefix="/maxsat", tags=["MaxSAT"])

# Thêm vào phần include_router
app.include_router(rag_router, prefix="/rag", tags=["RAG"])
app.include_router(gnn_router, prefix="/gnn", tags=["GNN"])


@app.get("/")
def home():
    return {"message": "Chào mừng đến với hệ thống chẩn đoán bệnh phổi"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
