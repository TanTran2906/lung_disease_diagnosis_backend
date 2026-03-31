from models.model_loader import ModelLoader
from typing import Optional
import asyncio
from .medical_diagnosis_system import MedicalDiagnosisSystem
from fastapi import APIRouter, UploadFile, File, Form


class MaxSATService:
    def __init__(self):
        self.system = None
        self.initialized = False

    async def initialize(self):
        """Khởi tạo hệ thống MaxSAT"""
        if not self.initialized:
            # Load FastText model từ ModelLoader
            ft_model = ModelLoader.load_text_model("FastText")

            # Khởi tạo hệ thống chẩn đoán
            self.system = MedicalDiagnosisSystem()
            self.system.fasttext_model = ft_model  # Sử dụng model đã load

            # Khởi tạo dữ liệu
            await asyncio.to_thread(self.system.extract_symptoms_from_files)
            await asyncio.to_thread(self.system.define_disease_conditions)

            self.initialized = True

    async def read_file_content(self, file):
        """Đọc nội dung file nếu có"""
        if file:
            content = await file.read()
            return content.decode("utf-8").strip()
        return None

    async def get_diagnosis(self,
                            text,
                            file):
        """Xử lý chẩn đoán tổng hợp"""
        try:
            if not self.initialized:
                await self.initialize()

            # Đọc nội dung đầu vào
            file_text = await self.read_file_content(file)
            input_text = file_text or text

            if not input_text:
                return {"error": "Không có thông tin triệu chứng đầu vào"}

            # Thực hiện chẩn đoán
            diagnosis = await asyncio.to_thread(
                self.system.classify_disease,
                input_text
            )

            # Chuẩn hóa định dạng đầu ra
            return {
                "maxsat_predictions": diagnosis["maxsat_predictions"],
                "maxsat_confidence": diagnosis["confidence"],
                "fasttext_prediction": diagnosis["fasttext_prediction"],
                "confidence": diagnosis["fasttext_confidence"],
                "final_diagnosis": diagnosis["final_diagnosis"],
                "detected_symptoms": diagnosis["detected_symptoms"]
            }

        except Exception as e:
            return {"error": f"Lỗi chẩn đoán: {str(e)}"}
