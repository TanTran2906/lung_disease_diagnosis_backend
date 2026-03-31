import os
import re
from typing import Dict, List
from config import SAMPLE_IMAGES_DIR, SAMPLE_TEXTS_DIR


class DemoService:
    def __init__(self):
        self.samples: Dict[str, Dict[str, List]] = self._load_samples()

    def _parse_file_info(self, filename: str) -> tuple:
        """Trích xuất thông tin từ tên file"""
        match = re.match(
            r'^(img|text)_([A-Za-z]+)_(\d+)\.dcm\.(jpg|txt)$', filename)
        if match:
            return match.group(1), match.group(2).lower(), int(match.group(3))
        return None, None, None

    def _load_samples(self) -> Dict[str, Dict[str, List]]:
        """Tải và tổ chức mẫu theo cấu trúc bệnh"""
        samples = {}

        # Xử lý ảnh
        for img_file in os.listdir(SAMPLE_IMAGES_DIR):
            file_type, disease, num = self._parse_file_info(img_file)
            if file_type == "img":
                if disease not in samples:
                    samples[disease] = {"images": [], "texts": []}

                samples[disease]["images"].append({
                    "id": f"{disease}_img_{num}",
                    "filename": img_file,
                    "path": os.path.join(SAMPLE_IMAGES_DIR, img_file).replace("\\", "/"),
                    "disease": disease,
                    "sample_number": num
                })

        # Xử lý text
        for txt_file in os.listdir(SAMPLE_TEXTS_DIR):
            file_type, disease, num = self._parse_file_info(txt_file)
            if file_type == "text":
                if disease not in samples:
                    samples[disease] = {"images": [], "texts": []}

                samples[disease]["texts"].append({
                    "id": f"{disease}_txt_{num}",
                    "filename": txt_file,
                    "path": os.path.join(SAMPLE_TEXTS_DIR, txt_file).replace("\\", "/"),
                    "disease": disease,
                    "sample_number": num
                })

        print(samples)

        return samples

    def get_available_diseases(self) -> List[str]:
        """Lấy danh sách các bệnh có sẵn"""
        return list(self.samples.keys())

    def get_samples_by_disease(self, disease: str) -> Dict[str, List]:
        """Lấy mẫu theo loại bệnh"""
        return self.samples.get(disease.lower(), {"images": [], "texts": []})

    def get_sample_content(self, sample_id: str):
        """Lấy nội dung mẫu theo ID"""
        parts = sample_id.split('_')
        if len(parts) != 3:
            return None

        disease = parts[0]
        sample_type = parts[1]
        sample_num = int(parts[2])

        target_samples = self.samples.get(disease, {}).get(
            "images" if sample_type == "img" else "texts", []
        )

        for sample in target_samples:
            if sample["sample_number"] == sample_num:
                try:
                    if sample_type == "img":
                        with open(sample["path"], 'rb') as f:
                            return f.read()
                    else:  # text
                        with open(sample["path"], 'r', encoding='utf-8') as f:
                            return f.read()
                except Exception as e:
                    print(f"Error reading sample {sample_id}: {str(e)}")
                    return None
        return None
