import os
import numpy as np
import pandas as pd
import logging
import faiss
import pickle
import requests
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import google.generativeai as genai  # Thư viện Gemini API


# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Paths
DATA_PATH = "D:/Data_Store/LV_KHMT/Data/text_clinical/train"
TRAIN_CSV = "D:/Data_Store/LV_KHMT/Data/clinical_train.csv"
VAL_CSV = "D:/Data_Store/LV_KHMT/Data/clinical_val.csv"
CACHE_DIR = 'D:/Data_Store/LV_KHMT/Data/RAG_FAISS/cache'

# Gemini API Key - https://aistudio.google.com/prompts/new_chat
GEMINI_API_KEY = "AIzaSyD55ElQ5lUwXPsRvoq5FHCPAaQ6MV_x8nE"


class MedicalRAG:
    def __init__(self):
        self.text_encoder = SentenceTransformer("keepitreal/vietnamese-sbert")
        logging.info("Text encoder model loaded")
        self.text_index = None
        self.train_data = None
        self.train_labels = None
        self.train_contents = None
        self.train_text_embeddings = None

    def ensure_cache_dir(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    def load_data(self):
        logging.info(f"Loading training data from {TRAIN_CSV}")
        try:
            self.train_data = pd.read_csv(TRAIN_CSV)
            logging.info(f"Loaded {len(self.train_data)} training samples")

            self.train_labels = self.train_data["Nhãn bệnh"].tolist()
            self.train_contents = self.train_data["Nội dung"].tolist()

            self.train_text_embeddings = []
            for emb_str in self.train_data["Đặc trưng văn bản"]:
                if isinstance(emb_str, str):
                    emb_str = emb_str.strip('[]')
                    emb = np.fromstring(emb_str, sep=' ')
                    self.train_text_embeddings.append(emb)
                else:
                    self.train_text_embeddings.append(np.array(emb_str))

            self.train_text_embeddings = np.array(
                self.train_text_embeddings, dtype=np.float32)
            logging.info(
                f"Text embeddings shape: {self.train_text_embeddings.shape}")

            return True
        except Exception as e:
            logging.error(f"Error loading training data: {e}")
            return False

    def build_faiss_index(self):
        logging.info("Building FAISS index...")
        text_dim = self.train_text_embeddings.shape[1]
        self.text_index = faiss.IndexFlatIP(text_dim)
        faiss.normalize_L2(self.train_text_embeddings)
        self.text_index.add(self.train_text_embeddings)
        logging.info(
            f"Built text index with {self.text_index.ntotal} vectors of dimension {text_dim}")
        return True

    def retrieve_similar_cases(self, query_text, top_k=5):
        logging.info(
            f"Retrieving similar cases for query: {query_text[:50]}...")

        query_text_embedding = self.text_encoder.encode(query_text)
        query_text_embedding = np.array(
            [query_text_embedding], dtype=np.float32)
        faiss.normalize_L2(query_text_embedding)
        logging.info(
            f"Query text encoded. Embedding shape: {query_text_embedding.shape}")

        text_distances, text_indices = self.text_index.search(
            query_text_embedding, top_k)
        logging.info(
            f"Text search completed. Found {len(text_indices[0])} matches.")
        logging.info(f"Top text distance scores: {text_distances[0][:5]}")

        similar_cases = []
        for i, idx in enumerate(text_indices[0]):
            if 0 <= idx < len(self.train_data):
                similar_cases.append({
                    'label': self.train_labels[idx],
                    'content': self.train_contents[idx],
                    'file_name': self.train_data.iloc[idx]['Tên tệp'],
                    'similarity_score': float(text_distances[0][i])
                })

        logging.info(f"Retrieved {len(similar_cases)} similar cases")
        return similar_cases

    def analyze_similar_cases(self, similar_cases):
        label_weights = {}
        for case in similar_cases:
            label = case['label']
            similarity = case['similarity_score']
            if label in label_weights:
                label_weights[label] += similarity
            else:
                label_weights[label] = similarity

        most_common_labels = sorted(
            label_weights.items(), key=lambda x: x[1], reverse=True)
        logging.info(f"Most common labels sorted: {most_common_labels}")
        return most_common_labels

    def initialize(self):
        success = True
        self.ensure_cache_dir()

        if not self.load_data():
            logging.error("Failed to load training data")
            success = False

        if not self.build_faiss_index():
            logging.error("Failed to build FAISS index")
            success = False

        return success


class LLMProcessor:
    def __init__(self, llm_choice="gemini"):
        self.llm_choice = llm_choice

        if llm_choice == "gemini":
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('models/gemini-2.0-flash')
            logging.info("Initialized Google Gemini API")

    def create_prompt(self, similar_cases, patient_text):
        prompt = f"Thông tin bệnh nhân: {patient_text}\n\n"
        prompt += "Dưới đây là thông tin từ 5 ca bệnh tương tự:\n\n"

        for i, case in enumerate(similar_cases, 1):
            prompt += f"Ca {i} - {case['label']}:\n"
            prompt += f"Nội dung: {case['content']}\n"
            prompt += f"Điểm tương đồng: {case['similarity_score']:.3f}\n\n"

        prompt += "Dựa trên thông tin trên, hãy phân tích và đưa ra:\n"
        prompt += "1. Chẩn đoán chính xác nhất\n"
        prompt += "2. Mức độ tin cậy của chẩn đoán (Cao/Trung bình/Thấp)\n"
        prompt += "3. Các triệu chứng chính được phát hiện\n"
        prompt += "4. Giải thích chi tiết lý do chọn chẩn đoán này\n"
        prompt += "5. Khuyến nghị điều trị ban đầu\n\n"
        prompt += "Hãy trả lời ngắn gọn, súc tích và có cấu trúc rõ ràng."

        return prompt

    def generate_diagnosis(self, prompt):
        try:
            # Thêm retry logic
            for _ in range(3):
                try:
                    response = self.model.generate_content(prompt)
                    return response.text
                except Exception as e:
                    logging.warning(f"Lỗi API, thử lại... {str(e)}")
                    continue
            return "Không thể tạo chẩn đoán"
        except Exception as e:
            logging.error(f"Lỗi LLM: {str(e)}")
            return "Lỗi trong quá trình tạo chẩn đoán"


def save_results(input_text, similar_cases, llm_diagnosis, output_file="D:/Data_Store/LV_KHMT/Data/RAG_FAISS/diagnosis_results.txt"):
    """Save diagnosis results to file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"PATIENT INPUT: {input_text}\n\n")
        f.write("SIMILAR CASES:\n")
        for i, case in enumerate(similar_cases, 1):
            f.write(
                f"Case {i} - {case['label']} (score: {case['similarity_score']:.3f}):\n")
            f.write(f"{case['content'][:500]}...\n\n")

        f.write("\nLLM DIAGNOSIS:\n")
        f.write(llm_diagnosis)

    logging.info(f"Results saved to {output_file}")


# Main function
def main():
    # Choose LLM
    llm_choice = "gemini"  # Change to "huggingface" if preferred

    # Create and initialize RAG system
    rag_system = MedicalRAG()
    if not rag_system.initialize():
        logging.error("Failed to initialize RAG system")
        return

    # Initialize LLM processor
    llm_processor = LLMProcessor(llm_choice=llm_choice)

    # Get patient input
    input_text = "ho khan, khó thở, phổi ran, sốt, mệt mỏi, chán ăn"
    logging.info(f"Processing patient input: {input_text}")

    # Retrieve similar cases
    similar_cases = rag_system.retrieve_similar_cases(input_text, top_k=5)

    # Create prompt for LLM
    prompt = llm_processor.create_prompt(similar_cases, input_text)
    logging.info(f"Created prompt for LLM: {len(prompt)} characters")

    # Generate diagnosis using LLM
    llm_diagnosis = llm_processor.generate_diagnosis(prompt)
    logging.info("LLM diagnosis generated")

    # Print results
    print("\n=== PATIENT INPUT ===")
    print(input_text)

    print("\n=== SIMILAR CASES ===")
    for i, case in enumerate(similar_cases, 1):
        print(
            f"{i}. {case['label']} - {case['file_name']} (similarity: {case['similarity_score']:.3f})")
        print(f"   {case['content'][:100]}...")

    print("\n=== LLM DIAGNOSIS ===")
    print(llm_diagnosis)

    # Save results
    save_results(input_text, similar_cases, llm_diagnosis)


if __name__ == "__main__":
    main()
