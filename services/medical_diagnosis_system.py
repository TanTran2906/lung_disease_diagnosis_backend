import os
import re
# import nltk
import pandas as pd
import fasttext
import logging
from collections import Counter
from pysat.formula import WCNF
from pysat.examples.rc2 import RC2
from underthesea import word_tokenize
import unidecode
# from underthesea import pos_tag
# from underthesea import chunk
# import underthesea
from tqdm import tqdm
import pickle
from vncorenlp import VnCoreNLP  # thư viện xử lý ngôn ngữ tự nhiên cho tiếng Việt
import json
from sklearn.feature_extraction.text import TfidfVectorizer
# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger')  # cho POS tagging
# nltk.download('averaged_perceptron_tagger_eng')
# nltk.download('maxent_ne_chunker')           # cho named entity recognition
# nltk.download('words')                       # hỗ trợ cho ne_chunk


# Define constants
DATA_PATH = "D:/Data_Store/LV_KHMT/Data/text_clinical/train"
FASTTEXT_MODEL_PATH = "D:/Data_Store/LV_KHMT/Data/FastText_v2/fasttext_model.bin"
CACHE_DIR = 'D:/Data_Store/LV_KHMT/Data/MaxSAT/cache'


class MedicalDiagnosisSystem:
    def __init__(self):
        """Initialize the medical diagnosis system"""
        self.symptoms_list = [
        ]  # Danh sách các triệu chứng  (sau khi lọc từ file txt)
        self.conditions = {}  # ánh xạ triệu chứng với bệnh
        self.fasttext_model = None
        # Ngưỡng: triệu chứng phải xuất hiện ít nhất 5 lần mới được tính
        self.symptom_threshold = 5

    def ensure_cache_dir(self):
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)

    def extract_symptoms_from_files(self):
        logging.info("Extracting symptom phrases using TF-IDF...")
        # lưu lại danh sách triệu chứng
        cache_path = os.path.join(CACHE_DIR, "tfidf_symptoms_cache.pkl")

        # Nếu cache tồn tại, load từ cache
        if os.path.exists(cache_path):
            logging.info("Loading TF-IDF symptoms from cache...")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.symptoms_list = cache_data['symptoms_list']
                # điểm TF-IDF tương ứng với từng cụm từ
                feature_scores = cache_data['feature_scores']
        else:
            documents = []
            file_count = 0

            # Thu thập tất cả văn bản có trong các folder bệnh
            disease_folders = [f for f in os.listdir(
                DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, f))]
            for disease_folder in tqdm(disease_folders, desc="📁 Reading disease folders"):
                disease_path = os.path.join(DATA_PATH, disease_folder)
                txt_files = [f for f in os.listdir(
                    disease_path) if f.endswith('.txt')]
                for filename in tqdm(txt_files, desc=f"📄 {disease_folder}", leave=False):
                    file_path = os.path.join(disease_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        documents.append(f.read().lower())
                        file_count += 1

            logging.info(f"Collected {file_count} documents.")

            # TF-IDF vectorizer: Trích xuất các n-gram từ đơn (1 từ) đến 5 từ
            # min_df (minimum document frequency): tần suất xuất hiện tối thiểu để giữ lại 1 cụm từ
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 5),
                min_df=self.symptom_threshold / len(documents)
            )
            # Tính toán ma trận TF-IDF: mỗi hàng là 1 văn bản, mỗi cột là 1 cụm từ n-gram
            X = vectorizer.fit_transform(documents)

            # danh sách tất cả các cụm từ (n-gram) được giữ lại sau khi lọc
            feature_names = vectorizer.get_feature_names_out()
            # X.sum(axis=0): cộng theo từng cột, tức là tổng TF-IDF score của mỗi cụm từ trên toàn bộ văn bản. Sau đó chuyển kết quả từ ma trận thành mảng 1 chiều
            feature_scores = X.sum(axis=0).A1

            # Lấy 5000 cụm từ có điểm TF-IDF cao nhất
            top_indices = feature_scores.argsort()[-5000:][::-1]
            self.symptoms_list = [feature_names[i] for i in top_indices]

            # Cache lại kết quả
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'symptoms_list': self.symptoms_list,
                    'feature_scores': feature_scores
                }, f)

        # Ghi ra file văn bản
        # output_path = os.path.join(CACHE_DIR, "extracted_symptoms_tfidf.txt")
        # with open(output_path, 'w', encoding='utf-8') as f:
        #     for i, symptom in enumerate(self.symptoms_list):
        #         score = feature_scores[top_indices[i]]
        #         f.write(f"{symptom} ({score:.4f})\n")

        logging.info(
            f"✅ Extracted {len(self.symptoms_list)} symptom phrases using TF-IDF.")

    def define_disease_conditions(self):
        """Điều kiện về triệu chứng tương ứng với từng bệnh"""
        logging.info("Defining disease conditions...")

        # Tạo một mapping giữa mỗi triệu chứng đã trích xuất (self.symptoms_list) với một số nguyên index bắt đầu từ 1 => mã hóa triệu chứng
        self.symptom_to_index = {symptom: i+1 for i,
                                 symptom in enumerate(self.symptoms_list)}

        # Define condition clauses for diseases
        """ 
            WHO (Tổ chức Y tế Thế giới) - Tiêu chuẩn chẩn đoán bệnh hô hấp
            CDC (Trung tâm Kiểm soát Dịch bệnh Hoa Kỳ) - Hướng dẫn triệu chứng COVID-19
            Mayo Clinic - Triệu chứng các bệnh phổi
            Hiệp hội Phổi Hoa Kỳ - Tiêu chuẩn chẩn đoán COPD
            Tài liệu của Bộ Y tế Việt Nam về triệu chứng bệnh Lao
        """
        # Format: {disease_name: [[Các triệu chứng BẮT BUỘC phải có], [Có ít nhất 1 trong số này là được], [KHÔNG được có triệu chứng này]]}
        self.conditions = {
            "Binhthuong": {
                "required": [],
                "at_least_one": ["khỏe mạnh", "không có triệu chứng", "sức khỏe bình thường"],
                "excluded": ["ho", "sốt", "khó thở", "đau ngực"]
            },
            "COPD": {
                "required": ["ho kéo dài", "khó thở", "ho đàm"],
                "at_least_one": ["khò khè", "đau ngực", "viêm phế quản mãn tính", "ho đàm", "thở rít", "giảm dung tích phổi"],
                "excluded": ["sốt cao đột ngột", "ho ra máu nhiều", "khởi phát cấp tính"]
            },
            "Covid": {
                "required": ["sốt", "ho khan"],
                "at_least_one": ["mất vị giác", "mất khứu giác", "mệt mỏi", "đau họng", "khó thở"],
                "excluded": ["dị ứng theo mùa", "ho ra máu", "đàm vàng", "viêm xoang"]
            },
            "Hen": {
                "required": ["khò khè", "khó thở"],
                "at_least_one": ["ho về đêm", "tức ngực", "thở rít"],
                "excluded": ["đau ngực dữ dội kéo dài", "sốt", "đàm vàng"]
            },
            "Lao": {
                "required": ["ho kéo dài", "sụt cân", "sốt nhẹ"],
                "at_least_one": ["đổ mồ hôi đêm", "ho ra máu", "mệt mỏi", "chán ăn"],
                "excluded": ["khó thở cấp tính"]
            },
            "Phuphoi": {
                "required": ["đau ngực dữ dội", "thở nhanh", "khó thở đột ngột"],
                "at_least_one": ["suy hô hấp", "tím tái", "huyết áp thấp"],
                "excluded": ["sốt cao"]
            },
            "Suyhohap": {
                "required": ["khó thở nặng", "nhịp thở nhanh"],
                "at_least_one": ["thở gấp", "tím môi", "hôn mê", "co kéo cơ hô hấp"],
                "excluded": []
            },
            "Trandich": {
                "required": ["đau ngực", "khó thở khi nằm"],
                "at_least_one": ["tiếng cọ màng phổi", "đau tăng khi hít sâu", "mờ đáy phổi trên X-quang", "dịch trong màng phổi"],
                "excluded": ["ho có đàm"]
            },
            "Trankhi": {
                "required": ["đau ngực đột ngột", "khó thở"],
                "at_least_one": ["thở nhanh", "tràn khí màng phổi"],
                "excluded": ["ho kéo dài"]
            },
            "Uphoi": {
                "required": ["ho kéo dài", "đau ngực", "khó thở"],
                "at_least_one": ["sụt cân", "ho ra máu", "mệt mỏi"],
                "excluded": ["nhiễm khuẩn cấp tính"]
            },
            "Viemphoi": {
                "required": ["sốt cao", "ho có đàm", "đau ngực khi thở"],
                "at_least_one": ["khó thở", "ớn lạnh", "mệt mỏi", "ran nổ", "nhịp thở nhanh", "SpO2 giảm"],
                "excluded": ["suy hô hấp mạn"]
            },
            "Xepphoi": {
                "required": ["khó thở tiến triển", "mệt mỏi"],
                "at_least_one": ["ho khan", "thở nhanh", "tăng CO2 máu", "da xanh tái"],
                "excluded": ["sốt", "viêm"]
            }
        }

        logging.info(f"Defined conditions for {len(self.conditions)} diseases")
        return self.conditions

    def load_fasttext_model(self):
        """Load the pre-trained FastText model"""
        logging.info(f"Loading FastText model from {FASTTEXT_MODEL_PATH}")
        try:
            self.fasttext_model = fasttext.load_model(FASTTEXT_MODEL_PATH)
            logging.info("FastText model loaded successfully")
            return True
        except Exception as e:
            logging.error(f"Error loading FastText model: {e}")
            return False

    def preprocess_text(self, text):
        """Preprocess input text for models"""
        text = text.lower()
        text = unidecode.unidecode(text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = word_tokenize(text, format="text")
        return text

    # Xác định triệu chứng nào trong self.symptoms_list xuất hiện trong input_text
    def extract_symptoms(self, text):
        """Extract symptoms from patient text input"""
        logging.info(f"Extracting symptoms from: {text}")

        # Chuyển input về chữ thường
        text_lower = text.lower()

        # Lọc danh sách triệu chứng bằng cách:
        # 1. Ưu tiên các cụm từ dài hơn
        # 2. Loại bỏ các từ đơn trừ khi chúng là triệu chứng y tế được định nghĩa rõ
        valid_symptoms = {}  # lưu trữ các triệu chứng hợp lệ được phát hiện

        # Sắp xếp triệu chứng theo độ dài (từ dài đến ngắn) để ưu tiên các cụm từ dài hơn
        sorted_symptoms = sorted(self.symptoms_list, key=len, reverse=True)

        # Một số từ đơn quan trọng cần giữ lại (có thể mở rộng danh sách này)
        important_single_words = {
            # Triệu chứng hô hấp
            "ho", "sốt", "khạc", "khò", "thở", "khàn", "hắt",

            # Bổ sung thêm triệu chứng hô hấp
            "ran", "rale", "đàm", "phổi", "ngực", "giảm", "thô", "co", "kéo", "lõm",

            # Triệu chứng tiêu hóa
            "nôn", "tiêu", "ói", "buồn", "đau", "sôi", "chán", "táo",

            # Bổ sung thêm triệu chứng tiêu hóa
            "bụng", "gan", "lách", "vị", "ấn", "mềm", "to", "lỏng",

            # Triệu chứng thần kinh
            "chóng", "mệt", "mờ", "nhức", "tê", "ngất", "co", "run", "liệt", "tê",

            # Bổ sung thêm triệu chứng thần kinh
            "tỉnh", "lơ", "mơ", "mê", "yếu", "suy", "nhược", "trú",

            # Triệu chứng tim mạch
            "tím", "sưng", "phù", "hồi", "thắt", "nhói", "tăng", "hạ",

            # Bổ sung thêm triệu chứng tim mạch
            "tim", "mạch", "quay", "rõ", "đều", "huyết", "áp", "ấm",

            # Các triệu chứng khác
            "sụt", "gầy", "sút", "rét", "lạnh", "suy", "yếu", "nhợt", "ngứa", "xanh",
            "đổ", "chảy", "tiểu", "vã", "ù", "viêm", "loét", "đỏ", "phát", "ra",
            "mất", "rối", "lo", "sợ", "hoảng", "bứt", "nổi", "giảm", "tăng",

            # Bổ sung thêm triệu chứng khác
            "ớn", "kiệt", "ăn", "uống", "kém", "trạng", "nhiệt", "độ", "môi", "hồng", "nhạt",
            "niêm", "tiếp", "xúc", "tốt", "chi", "lưng", "cơ", "mỏi", "lói", "cứng", "giật",
            "buốt", "nhiều"
        }
        # Duyệt qua danh sách triệu chứng đã sắp xếp
        for symptom in sorted_symptoms:
            # Tách triệu chứng thành các từ riêng lẻ,  kiểm tra xem triệu chứng có phải là từ đơn không
            # Nếu là từ đơn VÀ không nằm trong danh sách từ đơn quan trọng → bỏ qua (continue)
            if len(symptom.split()) == 1 and symptom not in important_single_words:
                continue

            # Kiểm tra xem triệu chứng có trong văn bản không
            if symptom in text_lower:
                # Kiểm tra xem triệu chứng này có phải là một phần của triệu chứng dài hơn đã tìm thấy
                # Ví dụ: nếu "ho có đàm" đã được phát hiện, thì "ho" sẽ được coi là phần con và bị loại bỏ
                is_substring = False
                for detected in valid_symptoms:
                    # symptom != detected: đảm bảo không phải là cùng một triệu chứng
                    if symptom in detected and symptom != detected:
                        is_substring = True
                        break
                # Nếu triệu chứng KHÔNG phải là chuỗi con của triệu chứng dài hơn → thêm vào danh sách các triệu chứng hợp lệ
                if not is_substring:
                    valid_symptoms[symptom] = True

        # Duyệt qua toàn bộ danh sách triệu chứng ban đầu (self.symptoms_list)
        # Với những triệu chứng không có trong text, đặt giá trị là False
        result = {
            symptom: symptom in valid_symptoms for symptom in self.symptoms_list}

        # Log ra danh sách các triệu chứng được phát hiện
        detected_symptoms = [s for s, present in result.items() if present]
        logging.info(f"Extracted symptoms: {detected_symptoms}")

        return result

    # Dự đoán bệnh từ mô tả triệu chứng bằng mô hình FastText đã huấn luyện
    def classify_disease_fasttext(self, text):
        """Classify disease using FastText model"""
        if not self.fasttext_model:
            logging.error("FastText model not loaded")
            return None

        # Preprocess the text
        preprocessed_text = self.preprocess_text(text)

        # Get prediction from FastText model
        prediction = self.fasttext_model.predict(preprocessed_text)
        predicted_label = prediction[0][0].replace("__label__", "")
        confidence = prediction[1][0]

        logging.info(
            f"FastText prediction: {predicted_label} with confidence {confidence:.4f}")
        return predicted_label, confidence

    def solve_maxsat(self, symptoms):
        """Solve MaxSAT to find the most likely disease based on symptoms and conditions"""
        # wcnf là một đối tượng lưu các mệnh đề:
        # Hard clause: Bắt buộc thỏa mãn
        # Soft clause: Có trọng số, solver sẽ cố gắng thỏa mãn càng nhiều càng tốt
        wcnf = WCNF()

        # Thêm hard clause dựa trên triệu chứng hiện có (Hard clause = thông tin đầu vào không được thay đổi)
        """ 
        symptoms = {"sốt": True, "buồn nôn": False}
        symptom_to_index = {"sốt": 1, "buồn nôn": 2}
        → Hard clause: [1], [-2]
        """
        # Với mỗi triệu chứng trong symptoms (là input người dùng, ví dụ "ho": True, "đau họng": False)
        for symptom, is_present in symptoms.items():
            # Mỗi triệu chứng được ánh xạ sang một số nguyên (idx)
            if symptom in self.symptom_to_index:
                idx = self.symptom_to_index[symptom]
                # Nếu triệu chứng có xuất hiện => thêm +idx vào hard clause ngược lại thêm -idx (weight=None)
                wcnf.append([idx if is_present else -idx])
                logging.info(
                    f"Added hard clause for symptom '{symptom}': [{idx if is_present else -idx}]")

        # Thêm soft clause cho từng bệnh trong self.conditions
        for disease, condition_sets in self.conditions.items():
            # "required": các triệu chứng phải có nếu mắc bệnh, trọng số 3 (ưu tiên cao)
            for symptom in condition_sets["required"]:
                if symptom in self.symptom_to_index:
                    idx = self.symptom_to_index[symptom]
                    # Thêm +idx vào soft clause với trọng số cao
                    wcnf.append([idx], weight=3)
                    logging.info(
                        f"Added soft clause for required symptom '{symptom}' in {disease}: [{idx}] (weight 3)")

            # Với những triệu chứng nên có ít nhất một → gộp lại trong một mệnh đề OR
            if condition_sets["at_least_one"]:
                at_least_one_clause = []
                for symptom in condition_sets["at_least_one"]:
                    if symptom in self.symptom_to_index:
                        idx = self.symptom_to_index[symptom]
                        at_least_one_clause.append(idx)

                if at_least_one_clause:
                    # Add as soft clause with weight 2 (medium priority)
                    wcnf.append(at_least_one_clause, weight=2)
                    logging.info(
                        f"Added soft clause for at least one symptom in {disease}: {at_least_one_clause} (weight 2)")

            # Nếu có các triệu chứng này thì không phù hợp với bệnh
            for symptom in condition_sets["excluded"]:
                if symptom in self.symptom_to_index:
                    idx = self.symptom_to_index[symptom]
                    # -idx nghĩa là triệu chứng đó nên không có => trọng số thấp hơn
                    wcnf.append([-idx], weight=1)
                    logging.info(
                        f"Added soft clause for excluded symptom '{symptom}' in {disease}: [{-idx}] (weight 1)")

        # Solve the MaxSAT problem
        solver = RC2(wcnf)
        solution = solver.compute()

        # solution = [1, -2, 5, -6], triệu chứng có chỉ số 1 và 5, 2 và 6 không xảy ra
        if solution:
            logging.info(f"MaxSAT solution found: {solution}")

            # Duyệt từng bệnh để tính điểm
            disease_scores = {}
            for disease, condition_sets in self.conditions.items():
                # Mỗi bệnh được tính tổng điểm từ các phần sau:
                # required: +3 điểm nếu tất cả triệu chứng bắt buộc đều có
                # at_least_one: +2 điểm nếu có ít nhất 1 triệu chứng phù hợp
                # excluded: +1 điểm nếu tất cả triệu chứng cần loại trừ đều không xuất hiện
                score = 0

                # Check required symptoms
                required_satisfied = True
                for symptom in condition_sets["required"]:
                    if symptom in self.symptom_to_index:
                        idx = self.symptom_to_index[symptom]
                        # Nếu thiếu 1 cái => không cộng điểm
                        if idx not in solution:
                            required_satisfied = False
                            break

                if required_satisfied:
                    score += 3

                # Check "at least one" symptoms
                at_least_one_satisfied = False
                for symptom in condition_sets["at_least_one"]:
                    if symptom in self.symptom_to_index:
                        idx = self.symptom_to_index[symptom]
                        # Nếu có ít nhất một triệu chứng trong nhóm này xuất hiện → cộng 2 điểm
                        if idx in solution:
                            at_least_one_satisfied = True
                            break

                if at_least_one_satisfied:
                    score += 2

                # Check excluded symptoms
                excluded_satisfied = True
                for symptom in condition_sets["excluded"]:
                    if symptom in self.symptom_to_index:
                        idx = self.symptom_to_index[symptom]
                        # Nếu có bất kỳ triệu chứng nào xuất hiện → không cộng
                        if idx in solution:
                            excluded_satisfied = False
                            break

                if excluded_satisfied:
                    score += 1

                disease_scores[disease] = score

            # Find the disease with the highest score
            if disease_scores:
                # Lấy bệnh có điểm cao nhất
                max_score = max(disease_scores.values())
                # Nếu nhiều bệnh có cùng điểm cao nhất → trả về danh sách tất cả (đều có khả năng như nhau)
                best_matches = [
                    disease for disease, score in disease_scores.items() if score == max_score]

                logging.info(f"Disease scores: {disease_scores}")
                logging.info(f"Best matches from MaxSAT: {best_matches}")

                return best_matches

        logging.warning("No solution found for MaxSAT problem")
        return []

    def classify_disease(self, text):
        """Main method to classify disease from symptoms text"""
        # Trích xuất triệu chứng from text
        symptoms = self.extract_symptoms(text)

        # Phân loại bệnh bằng mô hình FastText
        ft_prediction, confidence = self.classify_disease_fasttext(text)

        # tìm ra các bệnh phù hợp nhất dựa trên logic triệu chứng => Trả về danh sách bệnh hợp lý nhất
        maxsat_predictions = self.solve_maxsat(symptoms)

        # Combine results
        final_result = {
            "fasttext_prediction": ft_prediction,
            "fasttext_confidence": confidence,
            "maxsat_predictions": maxsat_predictions,
            "detected_symptoms": [s for s, present in symptoms.items() if present]
        }

        # Quyết định chẩn đoán cuối cùng
        # Khi FastText và MaxSAT đều chọn chung 1 bệnh → rất chắc chắn
        if ft_prediction in maxsat_predictions:
            final_result["final_diagnosis"] = ft_prediction
            final_result["confidence"] = "Cao (được xác nhận bởi cả hai phương thức)"
        # FastText không chắc chắn (confidence < 0.7) → ưu tiên MaxSAT
        elif maxsat_predictions and confidence < 0.7:
            final_result["final_diagnosis"] = maxsat_predictions[0]
            final_result["confidence"] = "Trung bình (FastText không chắc chắn với độ tự tin < 70%)"
        # Nếu MaxSAT không đưa ra gì, nhưng FastText có confidence cao → chấp nhận kết quả FastText
        elif not maxsat_predictions and confidence >= 0.7:
            final_result["final_diagnosis"] = ft_prediction
            final_result["confidence"] = "Trung bình  (chỉ Fasttext, không xác nhận MaxSat)"
        # Chỉ MaxSAT đưa ra kết quả
        elif maxsat_predictions:
            final_result["final_diagnosis"] = maxsat_predictions[0]
            final_result["confidence"] = "Thấp (chỉ MaxSAT)"
        # Cả hai không đáng tin: Khi thông tin triệu chứng quá mơ hồ → vẫn trả về FastText nhưng báo độ tin cậy rất thấp
        else:
            final_result["final_diagnosis"] = ft_prediction
            final_result["confidence"] = "Rất thấp (thông tin triệu chứng không đủ"

        logging.info(
            f"Final diagnosis: {final_result['final_diagnosis']} with {final_result['confidence']}")
        return final_result


# Example usage
if __name__ == "__main__":
    # Create medical diagnosis system
    system = MedicalDiagnosisSystem()

    # Extract symptoms from files (this would be done once during initialization)
    system.extract_symptoms_from_files()

    # Define disease conditions
    system.define_disease_conditions()

    # Load FastText model
    if system.load_fasttext_model():
        # Test with a sample input
        input_text = "Đau ngực, khó thở, ho có đàm"
        # test_cases = [
        #     "Bệnh nhân sốt cao kèm ho khan và khó thở",
        #     "Triệu chứng ho mãn tính, thở khò khè về đêm",
        #     "Đau ngực, khó thở, ho có đờm"
        # ]
        diagnosis = system.classify_disease(input_text)

        print("\n=== DIAGNOSIS RESULT ===")
        print(f"Patient symptoms: {input_text}")
        print(f"Detected symptoms: {diagnosis['detected_symptoms']}")
        print(
            f"FastText prediction: {diagnosis['fasttext_prediction']} (confidence: {diagnosis['fasttext_confidence']:.2f})")
        print(f"MaxSAT predictions: {diagnosis['maxsat_predictions']}")
        print(f"Final diagnosis: {diagnosis['final_diagnosis']}")
        print(f"Confidence level: {diagnosis['confidence']}")
    else:
        print(
            "Error: Could not load FastText model. Please make sure the model file exists.")
