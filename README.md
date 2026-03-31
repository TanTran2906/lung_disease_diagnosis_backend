# Backend hệ thống chẩn đoán bệnh phổi đa mô hình

## 1. Tổng quan

Backend xây dựng bằng FastAPI cho bài toán hỗ trợ chẩn đoán bệnh phổi từ nhiều nguồn dữ liệu khác nhau, bao gồm:

- văn bản mô tả triệu chứng lâm sàng
- ảnh X-quang phổi
- dữ liệu đa phương thức kết hợp ảnh và văn bản
- cơ chế hợp nhất kết quả bằng voting
- suy luận dựa trên luật với MaxSAT
- truy hồi ca bệnh tương đồng bằng RAG
- suy luận dựa trên graph với GNN

Hệ thống được thiết kế theo hướng service-oriented:

- `routes/`: khai báo API endpoint
- `services/`: chứa business logic và pipeline suy luận
- `models/`: chứa `model_loader.py` và các model weights (dung lượng lớn nên không thể tải lên)
- `data/`: dữ liệu mẫu phục vụ demo

## 2. Mục tiêu chức năng

Backend cung cấp các nhóm chức năng chính sau:

- Text classification từ mô tả triệu chứng bằng `FastText`, `Electra`, `DistillBERT`
- Image classification từ ảnh X-quang bằng `ViT`, `Lenet`, `MobileNet`, `DenseNet121`, `DenseNet169`
- Multimodal classification bằng cách kết hợp đặc trưng ảnh và embedding văn bản
- Voting ensemble giữa nhóm model ảnh và model văn bản
- Rule-based diagnosis với `MaxSAT` kết hợp `FastText`
- Retrieval-Augmented Generation bằng `FAISS` và `Gemini`
- Demo API để chọn dữ liệu mẫu và chạy nhiều model trên cùng một ca bệnh

## 3. Miền nhãn dự đoán

Các module hiện đang dùng cùng một tập nhãn gồm 12 lớp:

| ID  | Nhãn hệ thống | Diễn giải                    |
| --- | ------------- | ---------------------------- |
| 0   | `Binhthuong`  | Bình thường                  |
| 1   | `COPD`        | Bệnh phổi tắc nghẽn mạn tính |
| 2   | `Covid`       | COVID-19                     |
| 3   | `Hen`         | Hen phế quản                 |
| 4   | `Lao`         | Lao phổi                     |
| 5   | `Phuphoi`     | Phù phổi                     |
| 6   | `Suyhohap`    | Suy hô hấp                   |
| 7   | `Trandich`    | Tràn dịch màng phổi          |
| 8   | `Trankhi`     | Tràn khí màng phổi           |
| 9   | `Uphoi`       | U phổi                       |
| 10  | `Viemphoi`    | Viêm phổi                    |
| 11  | `Xepphoi`     | Xẹp phổi                     |

## 4. Kiến trúc triển khai

### 4.1. Điểm vào ứng dụng

File khởi động chính là `main.py`.

Khi server start:

- nạp toàn bộ text model, image model và multimodal model qua `ModelLoader.load_all_models()`
- khởi tạo `MedicalRAG` và lưu vào `app.state.rag_system`
- bật CORS toàn cục với `allow_origins=["*"]`
- mount các router theo từng nhóm nghiệp vụ

### 4.2. Cấu trúc thư mục

```text
backend/
├── main.py
├── config.py
├── requirements.txt
├── data/
│   └── samples/
├── models/
│   ├── model_loader.py
│   ├── image/
│   ├── multimodal/
│   ├── text/
│   └── gnn/
├── routes/
│   ├── text_routes.py
│   ├── image_routes.py
│   ├── multimodal_routes.py
│   ├── voting_routes.py
│   ├── demo_routes.py
│   ├── maxsat_routes.py
│   ├── rag_routes.py
│   └── gnn_routes.py
└── services/
    ├── text_service.py
    ├── image_service.py
    ├── multimodal_service.py
    ├── voting_service.py
    ├── maxsat_service.py
    ├── medical_diagnosis_system.py
    ├── rag_service.py
    ├── gnn_service.py
    ├── demo_service.py
    └── prediction_handler.py
```

### 4.3. Luồng xử lý tổng quát

1. Client gửi request `multipart/form-data` hoặc `application/json` tùy endpoint.
2. Router nhận input và gọi service tương ứng.
3. Service chuẩn hóa dữ liệu đầu vào.
4. `ModelLoader` nạp model từ local storage nếu model chưa có trong cache.
5. Service thực hiện prediction hoặc reasoning.
6. API trả về JSON response cho frontend hoặc consumer khác.

## 5. Đặc tả các module

### 5.1. Text module

- File route: `routes/text_routes.py`
- File service: `services/text_service.py`
- Model hỗ trợ: `FastText`, `Electra`, `DistillBERT`
- Chức năng: phân loại mô tả triệu chứng sang 1 trong 12 nhãn bệnh

Pipeline chính:

- đọc `text` trực tiếp hoặc nội dung từ file `.txt`
- tiền xử lý bằng lowercase, bỏ dấu, loại số, loại ký tự đặc biệt, tokenize bằng `underthesea`
- suy luận bằng model đã chọn
- trả về `model`, `prediction`, `label`, `confidence`

### 5.2. Image module

- File route: `routes/image_routes.py`
- File service: `services/image_service.py`
- Model hỗ trợ: `ViT`, `Lenet`, `MobileNet`, `DenseNet121`, `DenseNet169`
- Chức năng: phân loại ảnh X-quang phổi

Pipeline chính:

- đọc binary của ảnh upload
- decode ảnh bằng `OpenCV`
- tiền xử lý theo từng backbone
- suy luận bằng TensorFlow hoặc PyTorch tùy model
- trả về `model`, `prediction`, `label`, `confidence`

### 5.3. Multimodal module

- File route: `routes/multimodal_routes.py`
- File service: `services/multimodal_service.py`
- Model hỗ trợ:
    - `resnet_sbert`
    - `mobilenet_sbert`
    - `densenet121_sbert`
    - `densenet169_sbert`

Chức năng:

- kết hợp ảnh X-quang và mô tả triệu chứng
- dùng `SentenceTransformer` để sinh text embedding
- ghép đặc trưng ảnh và đặc trưng văn bản trước lớp phân loại cuối

### 5.4. Voting module

- File route: `routes/voting_routes.py`
- File service: `services/voting_service.py`

Chiến lược ensemble hiện tại:

- model ảnh: `DenseNet121`, `DenseNet169`
- model text: `FastText`, `Electra`
- điểm của từng model được tính từ top-k prediction
- nhóm image được gán trọng số `0.4`
- nhóm text được gán trọng số `0.6`

Kết quả đầu ra gồm:

- `final_prediction`
- `top_predictions`
- `model_scores`
- `label_mapping`

### 5.5. MaxSAT module

- File route: `routes/maxsat_routes.py`
- File service: `services/maxsat_service.py`
- File logic lõi: `services/medical_diagnosis_system.py`

Chức năng:

- trích xuất triệu chứng từ dữ liệu huấn luyện bằng `TF-IDF`
- ánh xạ luật chẩn đoán bệnh theo tập điều kiện `required`, `at_least_one`, `excluded`
- giải bài toán tối ưu ràng buộc bằng `RC2` trong `python-sat`
- kết hợp kết quả luật với `FastText`

Kết quả trả về gồm:

- `maxsat_predictions`
- `maxsat_confidence`
- `fasttext_prediction`
- `confidence`
- `final_diagnosis`
- `detected_symptoms`

### 5.6. RAG module

- File route: `routes/rag_routes.py`
- File service: `services/rag_service.py`

Chức năng:

- đọc dữ liệu lâm sàng từ file CSV
- dựng FAISS index trên embedding văn bản
- truy hồi `top_k` ca bệnh tương tự
- tạo prompt và gọi LLM để sinh chẩn đoán tóm tắt

Thành phần chính:

- retriever: `FAISS`
- embedding model: `keepitreal/vietnamese-sbert`
- generator: `Gemini`

### 5.7. GNN module

- File route: `routes/gnn_routes.py`
- File service: `services/gnn_service.py`

Ý tưởng xử lý:

- trích xuất image embedding
- sinh text embedding
- tạo graph suy luận với dữ liệu tham chiếu
- phân loại bằng `GATv2Conv`

### 5.8. Demo module

- File route: `routes/demo_routes.py`
- File service: `services/demo_service.py`
- File xử lý hợp nhất: `services/prediction_handler.py`

Chức năng:

- liệt kê bệnh có dữ liệu mẫu
- liệt kê ảnh và text mẫu theo bệnh
- chạy dự đoán trên nhiều model được chọn cùng lúc

## 6. Danh sách API

### 6.1. Tổng quan endpoint

| Method | Endpoint                          | Mô tả                            |
| ------ | --------------------------------- | -------------------------------- |
| `GET`  | `/`                               | Health message đơn giản          |
| `POST` | `/text/predict-text/`             | Dự đoán từ văn bản               |
| `POST` | `/image/predict-image/`           | Dự đoán từ ảnh X-quang           |
| `POST` | `/multimodal/predict-multimodal/` | Dự đoán đa phương thức           |
| `POST` | `/voting/predict-voting/`         | Hợp nhất kết quả ảnh và text     |
| `GET`  | `/demo/diseases`                  | Lấy danh sách bệnh có sample     |
| `GET`  | `/demo/samples/{disease}`         | Lấy sample theo bệnh             |
| `POST` | `/demo/predict`                   | Dự đoán bằng sample có sẵn       |
| `POST` | `/maxsat/diagnose`                | Chẩn đoán bằng MaxSAT + FastText |
| `POST` | `/rag/diagnose`                   | Chẩn đoán bằng RAG + LLM         |
| `POST` | `/gnn/predict`                    | Dự đoán bằng GNN                 |

### 6.2. Input chi tiết

#### `POST /text/predict-text/`

`multipart/form-data`

| Trường       | Kiểu     | Bắt buộc | Ghi chú                              |
| ------------ | -------- | -------- | ------------------------------------ |
| `model_name` | `string` | Có       | `FastText`, `Electra`, `DistillBERT` |
| `text`       | `string` | Không    | Văn bản nhập trực tiếp               |
| `file`       | `file`   | Không    | File `.txt` chứa triệu chứng         |

#### `POST /image/predict-image/`

`multipart/form-data`

| Trường       | Kiểu     | Bắt buộc | Ghi chú                                                   |
| ------------ | -------- | -------- | --------------------------------------------------------- |
| `model_name` | `string` | Có       | `ViT`, `Lenet`, `MobileNet`, `DenseNet121`, `DenseNet169` |
| `file`       | `file`   | Có       | Ảnh X-quang                                               |

#### `POST /multimodal/predict-multimodal/`

`multipart/form-data`

| Trường       | Kiểu     | Bắt buộc | Ghi chú              |
| ------------ | -------- | -------- | -------------------- |
| `model_name` | `string` | Có       | Tên multimodal model |
| `image`      | `file`   | Có       | Ảnh X-quang          |
| `text`       | `string` | Không    | Mô tả triệu chứng    |
| `text_file`  | `file`   | Không    | File `.txt`, UTF-8   |

#### `POST /voting/predict-voting/`

`multipart/form-data`

| Trường      | Kiểu     | Bắt buộc | Ghi chú           |
| ----------- | -------- | -------- | ----------------- |
| `image`     | `file`   | Có       | Ảnh X-quang       |
| `text`      | `string` | Không    | Mô tả triệu chứng |
| `text_file` | `file`   | Không    | File `.txt`       |

#### `POST /demo/predict`

`multipart/form-data`

| Trường            | Kiểu            | Bắt buộc | Ghi chú                  |
| ----------------- | --------------- | -------- | ------------------------ |
| `image_sample_id` | `string`        | Có       | Ví dụ: `covid_img_1`     |
| `text_sample_id`  | `string`        | Có       | Ví dụ: `covid_txt_1`     |
| `selected_models` | `array[string]` | Có       | Danh sách model cần chạy |

#### `POST /maxsat/diagnose`

`multipart/form-data`

| Trường | Kiểu     | Bắt buộc | Ghi chú           |
| ------ | -------- | -------- | ----------------- |
| `text` | `string` | Không    | Mô tả triệu chứng |
| `file` | `file`   | Không    | File `.txt`       |

#### `POST /rag/diagnose`

`multipart/form-data`

| Trường       | Kiểu     | Bắt buộc | Ghi chú                        |
| ------------ | -------- | -------- | ------------------------------ |
| `text`       | `string` | Không    | Mô tả triệu chứng              |
| `file`       | `file`   | Không    | File `.txt`                    |
| `top_k`      | `int`    | Không    | Mặc định `5`, giới hạn `1..10` |
| `llm_choice` | `string` | Không    | Mặc định `gemini`              |

#### `POST /gnn/predict`

`multipart/form-data`

| Trường | Kiểu     | Bắt buộc | Ghi chú           |
| ------ | -------- | -------- | ----------------- |
| `file` | `file`   | Có       | Ảnh X-quang       |
| `text` | `string` | Có       | Mô tả triệu chứng |

### 6.3. Mẫu response chuẩn

Các endpoint classification thường trả về cấu trúc gần như sau:

```json
{
    "model": "DenseNet121",
    "prediction": 10,
    "label": "Viemphoi",
    "confidence": 0.9421
}
```

Riêng các endpoint tổ hợp như `voting`, `maxsat`, `rag`, `demo` sẽ trả về JSON có cấu trúc mở rộng tùy theo logic của từng service.

## 7. Hướng dẫn chạy dự án

### 7.1. Yêu cầu môi trường

- Python 3.10 trở lên
- Windows là môi trường phù hợp nhất với trạng thái code hiện tại do nhiều đường dẫn đang hard-code theo ổ `D:`
- GPU là tùy chọn, hệ thống có fallback CPU ở một số module

### 7.2. Tạo virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 7.3. Cài dependency

File `requirements.txt` hiện đang trống, vì vậy cần tự tạo dependency list hoặc cài thủ công các thư viện mà mã nguồn đang dùng, tối thiểu gồm:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `numpy`
- `pandas`
- `opencv-python`
- `pillow`
- `torch`
- `torchvision`
- `tensorflow`
- `transformers`
- `sentence-transformers`
- `timm`
- `fasttext`
- `underthesea`
- `unidecode`
- `python-sat`
- `scikit-learn`
- `faiss-cpu`
- `google-generativeai`
- `torch-geometric`
- `tqdm`
- `vncorenlp`

Ví dụ:

```powershell
pip install fastapi uvicorn python-multipart numpy pandas opencv-python pillow torch torchvision tensorflow transformers sentence-transformers timm fasttext underthesea unidecode python-sat scikit-learn faiss-cpu google-generativeai torch-geometric tqdm vncorenlp
```

### 7.4. Chạy server

```powershell
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Sau khi chạy:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## 8. Cấu hình dữ liệu và model

Hiện tại nhiều đường dẫn đang được hard-code theo máy phát triển ban đầu. Trước khi chạy trên máy khác, cần rà soát và cập nhật các file sau:

- `config.py`
- `models/model_loader.py`
- `services/rag_service.py`
- `services/medical_diagnosis_system.py`
- `services/gnn_service.py`

Các nhóm tài nguyên chính cần tồn tại:

- dữ liệu mẫu trong `data/samples/images` và `data/samples/texts`
- text model trong `models/text/`
- image model trong `models/image/`
- multimodal weights trong `models/multimodal/`
- GNN artifacts trong `models/gnn/`
- dữ liệu CSV và dữ liệu train ngoài repo nếu dùng `RAG`, `MaxSAT`, `GNN`

## 9. Ví dụ gọi API

### 9.1. Predict text

```bash
curl -X POST "http://127.0.0.1:8000/text/predict-text/" \
  -F "model_name=FastText" \
  -F "text=ho khan, sốt nhẹ, khó thở"
```

### 9.2. Predict image

```bash
curl -X POST "http://127.0.0.1:8000/image/predict-image/" \
  -F "model_name=DenseNet121" \
  -F "file=@sample.jpg"
```

### 9.3. Predict multimodal

```bash
curl -X POST "http://127.0.0.1:8000/multimodal/predict-multimodal/" \
  -F "model_name=resnet_sbert" \
  -F "image=@sample.jpg" \
  -F "text=ho kéo dài, tức ngực, khó thở"
```
