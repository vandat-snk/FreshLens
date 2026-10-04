# FreshLens 🍎🍌🍊🍅

FreshLens là đồ án môn **Xử lý ảnh** bằng Python, tập trung vào bài toán nhận diện **4 loại trái cây**:

- Táo (`apple`)
- Chuối (`banana`)
- Cam (`orange`)
- Cà chua (`tomato`)

Sau khi xác định loại quả, hệ thống tiếp tục dự đoán tình trạng bề ngoài:

- `fresh` — tươi
- `rotten` — có dấu hiệu hỏng

Phiên bản mô hình chính hiện tại sử dụng **EfficientNet-B0 + Transfer Learning** với PyTorch. Ứng dụng hỗ trợ **upload ảnh** và **chụp trực tiếp bằng camera** qua Streamlit.

> **Lưu ý:** `fresh/rotten` chỉ phản ánh dấu hiệu trực quan theo dữ liệu huấn luyện. Hệ thống không thay thế kiểm nghiệm an toàn thực phẩm và không kết luận trái cây có ăn được hay không.

---

## 1. Trạng thái hiện tại của dự án

FreshLens đã trải qua hai hướng tiếp cận:

### Baseline cũ — xử lý ảnh truyền thống + SVM

Pipeline ban đầu sử dụng:

- Resize / Letterbox
- CLAHE
- Gaussian Blur
- Canny / Sobel
- HSV / Lab histogram
- LBP
- Hu Moments
- HOG
- StandardScaler
- SVM Linear / RBF

Code baseline vẫn đang được giữ trong thư mục `src/` để phục vụ so sánh học thuật và truy vết quá trình phát triển.

### Mô hình chính hiện tại — EfficientNet-B0 CNN

Pipeline hiện dùng để train và demo thực tế nằm trong:

```text
FreshLens_Buoc3_CNN/
FreshLens_Buoc5_AppThucTe/
models/cnn_efficientnet_b0/
```

Mô hình chính:

```text
models/cnn_efficientnet_b0/best.pt
```

Kiến trúc hiện tại phân loại 8 tổ hợp:

```text
apple::fresh
apple::rotten
banana::fresh
banana::rotten
orange::fresh
orange::rotten
tomato::fresh
tomato::rotten
```

Sau đó hệ thống gộp xác suất để suy ra:

```text
Loại quả  -> apple / banana / orange / tomato
Tình trạng -> fresh / rotten
```

---

## 2. Kiến trúc tổng thể

```text
Ảnh upload / Camera
        │
        ▼
Đọc ảnh + sửa EXIF
        │
        ▼
RGB + Letterbox 224×224
        │
        ▼
ImageNet Normalization
        │
        ▼
EfficientNet-B0
        │
        ▼
Softmax 8 lớp
        │
        ├───────────────┐
        ▼               ▼
   Fruit prediction   Condition prediction
        │
        ▼
Open-set gate
        │
     ┌──┴───┐
     │      │
Supported Unsupported
     │      │
     ▼      ▼
Hiển thị   "Loại quả hiện
kết quả     chưa được hỗ trợ"
```

---

## 3. Cấu trúc repository hiện tại

```text
FreshLens/
│
├── FreshLens_Buoc1_DuLieu/
│   └── Chuẩn bị dữ liệu, MongoDB, manifest và kiểm tra metadata
│
├── FreshLens_Buoc2_ChiaTap/
│   └── Chia train / validation / test và kiểm soát rò rỉ dữ liệu
│
├── FreshLens_Buoc2B_GomNhom/
│   └── Gom ảnh gần trùng / cùng nhóm trước khi train
│
├── FreshLens_Buoc3_CNN/
│   ├── TRAIN_CNN.py
│   ├── EVALUATE_CNN.py
│   ├── PREDICT_CNN.py
│   ├── cnn_data.py
│   ├── cnn_model.py
│   ├── cnn_metrics.py
│   └── requirements-cnn.txt
│
├── FreshLens_Buoc4_ExternalTest/
│   └── External test V1
│
├── FreshLens_Buoc4B_MoRongExternalTest/
│   └── External test V2
│
├── FreshLens_Buoc5_AppThucTe/
│   ├── APP_CNN.py
│   ├── BUILD_OPENSET_GATE.py
│   ├── cnn_data.py
│   ├── cnn_model.py
│   ├── open_set.py
│   └── RUN_APP.cmd
│
├── FreshLens_Buoc5_FIX/
│   └── Bản sửa setup open-set gate trong quá trình phát triển
│
├── models/
│   └── cnn_efficientnet_b0/
│       ├── best.pt
│       ├── config.json
│       ├── training_summary.json
│       ├── history.csv
│       ├── history.png
│       ├── validation_confusion_matrix.png
│       ├── open_set_gate.npz
│       └── open_set_gate.json
│
├── src/
│   └── Pipeline baseline SVM / handcrafted features cũ
│
├── scripts/
├── tests/
├── docs/
├── requirements.txt
├── requirements-db.txt
├── run_app.bat
├── run_train.bat
└── README.md
```

> **Quan trọng:** `src/` hiện vẫn là pipeline baseline SVM cũ. Mô hình CNN cuối hiện chưa được refactor hoàn toàn vào `src/`. Đây là một hạng mục đang được nhóm nâng cấp.

---

## 4. Môi trường

### Yêu cầu tối thiểu

- Windows 10/11
- Python 3.10+
- RAM khuyến nghị: 8 GB trở lên
- GPU NVIDIA là tùy chọn nhưng rất hữu ích khi train
- CUDA PyTorch nếu muốn train bằng GPU

Project đã được chạy thực tế với GPU NVIDIA RTX 2050 4 GB.

---

## 5. Tạo virtual environment

Mở CMD hoặc PowerShell tại thư mục project:

```powershell
py -m venv .venv
```

Kích hoạt môi trường:

### PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### CMD

```cmd
.venv\Scripts\activate
```

Cài các thư viện chung:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu dùng MongoDB / Cloudinary:

```powershell
pip install -r requirements-db.txt
```

---

## 6. Cài PyTorch cho CNN

PyTorch CUDA được cài riêng để phù hợp với GPU và phiên bản CUDA trên máy.

Có thể dùng script:

```text
FreshLens_Buoc3_CNN\CAI_CNN_GPU.cmd
```

Hoặc kiểm tra GPU trước:

```powershell
python FreshLens_Buoc3_CNN\CHECK_GPU.py
```

Sau khi thành công, chương trình phải nhận được:

```text
device=cuda
```

Nếu không có GPU NVIDIA, có thể chạy bằng CPU nhưng thời gian train sẽ lâu hơn.

---

## 7. Dataset

Dữ liệu gốc không được commit trực tiếp lên GitHub do dung lượng lớn.

Cấu trúc ảnh hỗ trợ:

```text
dataset/
└── raw/
    ├── apple/
    │   ├── fresh/
    │   └── rotten/
    ├── banana/
    │   ├── fresh/
    │   └── rotten/
    ├── orange/
    │   ├── fresh/
    │   └── rotten/
    └── tomato/
        ├── fresh/
        └── rotten/
```

Pipeline hiện tại sử dụng metadata đã được khóa trong:

```text
data/cnn_dataset_v3/
```

Thư mục này chứa các file như:

```text
manifest.csv
final_split_report.json
dataset_lock.json
```

Các file trên giúp đảm bảo:

- không thay đổi split ngoài ý muốn;
- không để cùng `group_id` xuất hiện ở nhiều split;
- không để file trùng SHA-256 xuất hiện ở nhiều split;
- train / validation / test được cố định trước khi train.

---

## 8. Tiền xử lý ảnh của CNN

CNN hiện tại không sử dụng trực tiếp CLAHE, Canny, HOG hay LBP cho prediction.

Pipeline CNN thực tế là:

```text
Image
  ↓
Pillow decode
  ↓
EXIF transpose
  ↓
RGBA -> nền trắng nếu cần
  ↓
RGB
  ↓
Letterbox 224×224
  ↓
ToTensor
  ↓
ImageNet Normalize
  ↓
EfficientNet-B0
```

Thông số normalize:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

## 9. Data augmentation khi train

Chỉ tập `train` được augmentation.

Hiện tại có:

- `RandomHorizontalFlip`
- `RandomAffine`
  - rotation khoảng ±15°
  - translate nhẹ
  - scale nhẹ
- `ColorJitter`
  - brightness
  - contrast
  - saturation
  - hue rất nhỏ

Validation và test **không augmentation**.

Mục đích là giúp model bền vững hơn với:

- góc chụp khác nhau;
- khoảng cách camera;
- ánh sáng;
- vị trí quả trong ảnh.

---

## 10. Kiến trúc EfficientNet-B0

FreshLens sử dụng:

```python
torchvision.models.efficientnet_b0
```

Khởi tạo mặc định bằng pretrained weights:

```text
ImageNet1K V1
```

Classifier gốc được thay bằng:

```text
Dropout(0.25)
Linear(..., 8)
```

Tám output tương ứng với 8 tổ hợp fruit + condition.

---

## 11. Transfer Learning

FreshLens train theo 2 giai đoạn.

### Giai đoạn 1 — Warm-up

- Freeze backbone `model.features`
- Chỉ train classifier mới
- Learning rate classifier cao hơn

Mục đích:

> Cho classifier thích nghi với 8 lớp mới mà chưa phá các đặc trưng ImageNet đã học.

### Giai đoạn 2 — Fine-tuning

- Unfreeze backbone
- Backbone dùng learning rate nhỏ
- Classifier dùng learning rate lớn hơn backbone

Hiện tại:

```text
backbone LR   ≈ 1e-4
classifier LR ≈ 3e-4
```

Do batch size nhỏ, BatchNorm sử dụng pretrained running statistics trong quá trình fine-tune.

---

## 12. Training configuration hiện tại

Cấu hình đã dùng cho model chính:

```text
Architecture        EfficientNet-B0
Input size          224 × 224
Pretrained          ImageNet
Batch size          8
Gradient accumulation 2
Effective batch     16
Warm-up epochs      3
Fine-tune epochs    tối đa 20
Early stopping      patience = 6
Optimizer           AdamW
Weight decay        1e-4
Label smoothing     0.05
Scheduler           CosineAnnealingLR
Gradient clipping   max_norm = 5
Mixed precision     FP16 khi chạy CUDA
Seed                42
```

---

## 13. Train CNN

Ví dụ:

```cmd
cd /d "E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens"
```

Chạy:

```cmd
.\.venv\Scripts\python.exe FreshLens_Buoc3_CNN\TRAIN_CNN.py ^
  --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" ^
  --data "data\cnn_dataset_v3" ^
  --output "models\cnn_efficientnet_b0" ^
  --device cuda ^
  --batch-size 8 ^
  --accumulation 2
```

Không nên train lại vào cùng output nếu muốn giữ model cũ.

Ví dụ candidate mới:

```text
models/cnn_candidate_v2/
```

---

## 14. Cơ chế chọn `best.pt`

Model không lấy epoch cuối.

Checkpoint tốt nhất được chọn theo:

```text
Joint Macro-F1 cao nhất
```

Nếu hai epoch có Macro-F1 gần bằng nhau:

```text
Validation Loss thấp hơn
```

Model hiện tại đạt checkpoint tốt nhất tại:

```text
epoch 9
```

Sau đó early stopping dừng quá trình ở epoch 15.

---

## 15. Kết quả model hiện tại

### Validation tại best checkpoint

Model hiện tại đạt xấp xỉ:

```text
Fruit Accuracy      99.66%
Condition Accuracy  99.49%
Joint Accuracy      99.16%
Joint Macro-F1      99.13%
```

### Internal Test

Kết quả test nội bộ đã ghi nhận:

```text
Fruit Accuracy      99.14%
Joint Accuracy      98.37%
Joint Macro-F1      98.35%
```

### External Test V2

Trên tập ảnh mới ngoài dataset:

```text
Fruit Accuracy      97.53%
Condition Accuracy  93.83%
Joint Accuracy      92.59%
```

Điều này cho thấy model nhận diện **loại quả khá tốt**, nhưng phân loại `fresh / rotten` trên ảnh thực tế vẫn là phần cần cải thiện.

> Không được dùng Validation Accuracy ~99% để khẳng định model đạt 99% trên mọi ảnh camera thực tế.

---

## 16. Đánh giá CNN

Chạy test cố định:

```cmd
.\.venv\Scripts\python.exe FreshLens_Buoc3_CNN\EVALUATE_CNN.py ^
  --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" ^
  --data "data\cnn_dataset_v3" ^
  --checkpoint "models\cnn_efficientnet_b0\best.pt" ^
  --split test ^
  --output "reports\cnn_test_final" ^
  --device cuda
```

Kết quả bao gồm:

```text
metrics.json
test_predictions.csv
test_errors.csv
confusion_matrix.png
```

Các metric quan trọng:

- Accuracy
- Precision
- Recall
- F1-score
- Macro-F1
- Balanced Accuracy
- Confusion Matrix
- Mean group accuracy

---

## 17. Cơ chế Supported / Unsupported

CNN 8 lớp là **closed-set classifier**.

Nếu đưa một ảnh nho vào CNN, CNN vẫn buộc phải chọn một trong:

```text
apple
banana
orange
tomato
```

Để giảm lỗi này, FreshLens có thêm một **open-set gate**.

Gate sử dụng:

- embedding 1280 chiều từ EfficientNet;
- fruit probability;
- top-class margin;
- entropy / certainty;
- maximum joint probability;
- cosine similarity với prototype;
- prototype gap;
- Logistic Regression.

Prototype được xây dựng bằng:

```text
MiniBatchKMeans
```

Gate trả về:

```text
supported = True / False
```

Nếu `False`, ứng dụng không hiển thị loại quả / tình trạng mà báo:

> **Loại quả này hiện chưa được FreshLens hỗ trợ.**

### Giới hạn

Open-set gate hiện là heuristic.

Nó **không thể đảm bảo từ chối mọi ảnh ngoài phạm vi**.

Bộ External V2 đã được dùng một phần để hiệu chỉnh open-set gate nên sau bước calibration đó, bộ này **không còn được xem là untouched final test set**.

---

## 18. Chạy ứng dụng CNN hiện tại

### Cách 1 — CMD

```cmd
cd /d "E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens"
```

Sau đó:

```cmd
.\.venv\Scripts\python.exe -m streamlit run FreshLens_Buoc5_AppThucTe\APP_CNN.py
```

Trình duyệt mở tại:

```text
http://localhost:8501
```

### Cách 2 — file CMD

Chạy:

```text
FreshLens_Buoc5_AppThucTe\RUN_APP.cmd
```

---

## 19. Chức năng ứng dụng

Ứng dụng hiện hỗ trợ:

### Upload ảnh

```text
.jpg
.jpeg
.png
.bmp
.webp
```

### Camera

Streamlit:

```python
st.camera_input(...)
```

### Kết quả

Nếu được hỗ trợ:

```text
Táo
Tình trạng: Tươi

Tin cậy loại quả: 97.2%
Tin cậy tình trạng: 94.8%
```

Nếu ngoài phạm vi:

```text
Loại quả này hiện chưa được FreshLens hỗ trợ.

Hệ thống hiện hỗ trợ:
Táo · Chuối · Cam · Cà chua
```

---

## 20. Baseline xử lý ảnh truyền thống

Phần `src/` hiện tại được giữ lại nhằm:

- làm baseline;
- phục vụ môn Xử lý ảnh;
- giải thích các thuật toán truyền thống;
- so sánh CNN với handcrafted features.

Các kỹ thuật trong baseline gồm:

### CLAHE

Tăng tương phản cục bộ trên kênh sáng.

### Gaussian Blur

Giảm nhiễu cao tần trước một số bước xử lý.

### Canny

Phát hiện biên từ gradient.

> Canny **không phải** thuật toán phát hiện vùng thối.

### HSV / Lab Histogram

Mô tả phân bố màu sắc.

### LBP

Mô tả texture cục bộ bằng quan hệ sáng/tối giữa pixel trung tâm và vùng lân cận.

### HOG

Mô tả phân bố hướng gradient theo cell/block.

### Hu Moments

Mô tả một số đặc điểm hình dạng gần bất biến với biến đổi hình học.

### SVM

Baseline dùng SVM Linear / RBF sau khi scale vector đặc trưng.

---

## 21. CNN và SVM khác nhau thế nào?

### Pipeline SVM

```text
Ảnh
 ↓
Tiền xử lý
 ↓
Con người thiết kế feature
 ↓
HSV/Lab + LBP + HOG + Hu
 ↓
StandardScaler
 ↓
SVM
 ↓
Prediction
```

### Pipeline CNN

```text
Ảnh
 ↓
Resize / Normalize
 ↓
CNN tự học đặc trưng
 ↓
EfficientNet-B0
 ↓
Classifier
 ↓
Prediction
```

CNN không cần con người thiết kế thủ công toàn bộ vector đặc trưng.

---

## 22. Các khái niệm nhóm phải hiểu khi bảo vệ

### CNN

Mạng nơ-ron tích chập dùng kernel để học các đặc trưng không gian từ ảnh.

### Convolution

Một kernel trượt trên ảnh / feature map để tạo feature map mới.

### Transfer Learning

Khởi tạo model bằng kiến thức đã học từ ImageNet rồi fine-tune cho bài toán trái cây.

### Freeze / Unfreeze

- Freeze: không cập nhật backbone.
- Unfreeze: cho phép backbone thích nghi với dataset mới.

### Softmax

Biến logits thành phân phối xác suất tổng bằng 1.

### Cross Entropy

Hàm loss phân loại giữa output model và nhãn thật.

### Label Smoothing

Giảm xu hướng model quá tự tin vào một lớp.

### AdamW

Optimizer thích nghi learning rate và tách weight decay khỏi moment update.

### Dropout

Random bỏ một phần activation khi train để regularize.

### Batch Normalization

Chuẩn hóa feature trong mạng; với batch nhỏ, FreshLens giữ pretrained running statistics trong fine-tune.

### Learning Rate Scheduler

Điều chỉnh learning rate theo quá trình train.

### Early Stopping

Dừng train khi validation không còn cải thiện.

### Gradient Accumulation

Cộng gradient qua nhiều mini-batch trước khi optimizer step, giúp giả lập batch lớn hơn khi VRAM nhỏ.

### Mixed Precision

Dùng FP16 ở các phép toán phù hợp để giảm VRAM và tăng tốc trên GPU.

---

## 23. Accuracy không phải Confidence

Ví dụ app hiển thị:

```text
Tin cậy: 97%
```

không có nghĩa hệ thống có accuracy 97%.

### Confidence

Score của model cho **một ảnh cụ thể**.

### Accuracy

Tỷ lệ dự đoán đúng trên **một tập ảnh có nhãn**.

Khi đánh giá mô hình phải dùng thêm:

- Precision
- Recall
- F1-score
- Macro-F1
- Confusion Matrix

---

## 24. Data leakage

Một vấn đề quan trọng trong bài toán ảnh là nhiều ảnh có thể đến từ cùng một quả.

Ví dụ:

```text
apple_01_angle1.jpg
apple_01_angle2.jpg
apple_01_angle3.jpg
```

Nếu angle1 nằm train nhưng angle2 nằm test thì model có thể nhận diện nền / quả cụ thể đã thấy trước đó.

FreshLens đã có bước group ảnh để giảm nguy cơ này.

Tuy nhiên metadata hiện tại vẫn ghi:

```text
physical_specimen_independence_confirmed = false
```

Điều đó có nghĩa nhóm chưa thể khẳng định tuyệt đối rằng mọi ảnh test đều đến từ một quả vật lý hoàn toàn khác train.

Đây là giới hạn cần nói đúng khi bảo vệ.

---

## 25. Git workflow

Repository:

```text
https://github.com/vandat-snk/FreshLens
```

Baseline hiện tại được đánh tag:

```text
v1-cnn-baseline
```

Khuyến nghị 4 thành viên phát triển theo branch:

```text
feat/data-v2
feat/model-v2
feat/inference-eval
feat/ui-v2
```

Không nên cho cả nhóm sửa trực tiếp `main`.

---

## 26. Hướng nâng cấp tiếp theo

Các hạng mục đang được ưu tiên:

### Dataset V2

Bổ sung ảnh thực tế:

- camera laptop;
- điện thoại;
- nền phức tạp;
- nhiều điều kiện ánh sáng;
- nhiều khoảng cách;
- hỏng nhẹ / hỏng cục bộ;
- nhiều ảnh `other`.

### CNN V2

Thử kiến trúc multi-task:

```text
EfficientNet Backbone
       │
 ┌─────┴──────┐
 │            │
Fruit Head   Condition Head
 │            │
5 class       2 class
 │
apple
banana
orange
tomato
other
```

Nếu fruit là `other`:

```text
không suy luận fresh / rotten
```

### Quality Gate

Từ chối / cảnh báo ảnh:

- quá mờ;
- quá tối;
- quá sáng;
- vật thể quá nhỏ.

### Explainability

Thêm:

```text
Grad-CAM
```

để minh họa vùng hình ảnh ảnh hưởng đến quyết định CNN.

> Grad-CAM là bản đồ đóng góp tương đối, không phải segmentation chính xác vùng hỏng.

### UI V2

Nâng cấp:

- dashboard hiện đại;
- camera guide;
- prediction card;
- latency;
- confidence;
- Grad-CAM;
- model analytics;
- confusion matrix;
- train history;
- error gallery.

---

## 27. Kiểm thử

Baseline:

```powershell
pytest -q
python -m compileall src scripts tests
```

CNN cần kiểm tra tối thiểu:

- ảnh JPEG;
- PNG trong suốt;
- ảnh EXIF xoay;
- ảnh quá tối;
- ảnh quá sáng;
- ảnh mờ;
- ảnh đúng 4 loại;
- ảnh ngoài 4 loại;
- ảnh không phải trái cây;
- ảnh có nhiều vật thể;
- thiếu model;
- checkpoint không đúng;
- gate không đúng model.

---

## 28. Mục tiêu cuối của FreshLens

Mục tiêu production/demo cuối:

```text
Ảnh upload / Camera
       ↓
Quality Check
       ↓
Supported Fruit?
       ↓
 ┌──────────────┬─────────────┐
 │ Có           │ Không       │
 ↓              ↓
Apple/Banana/   Chưa hỗ trợ
Orange/Tomato
 ↓
Fresh / Rotten
 ↓
Confidence
 ↓
Grad-CAM / Explanation
```

Mục tiêu kỹ thuật mong muốn trên **final external test hoàn toàn mới**:

```text
Fruit Accuracy        ≥ 98%
Condition Accuracy    ≥ 95%
Joint Accuracy        ≥ 94–95%
Supported Acceptance  ≥ 98%
Unknown Rejection     ≥ 90%
```

Các mục tiêu này phải được đo trên một tập final test chưa dùng để train, validation hoặc calibration.

---

## 29. Giới hạn hiện tại

FreshLens hiện là:

```text
Image Classification
```

không phải:

```text
Object Detection
Instance Segmentation
Fruit Counting
Food Safety Inspection
```

Hệ thống giả định:

> Trong ảnh có **một vật thể trái cây chính**.

Nếu có nhiều quả / nhiều loại quả trong cùng ảnh, kết quả có thể không đáng tin.

Ngoài ra:

- open-set detection không thể bao phủ mọi ảnh lạ;
- camera domain khác dataset có thể làm accuracy giảm;
- ánh sáng và background ảnh hưởng prediction;
- `fresh/rotten` chỉ dựa vào thông tin thị giác RGB.

---

## 30. Thành viên và phân công

Phân công chi tiết nằm tại:

```text
docs/
```

Bốn mảng chính:

```text
TV1 — Dataset & Image Processing
TV2 — CNN Training & Optimization
TV3 — Inference, Open-set & Evaluation
TV4 — UI, Integration & Testing
```

Mỗi thành viên phải:

- chạy được toàn bộ pipeline;
- hiểu phần mình phụ trách;
- hiểu kiến trúc tổng thể;
- giải thích được ít nhất:
  - một prediction đúng;
  - một prediction sai;
  - một ảnh bị unsupported;
  - một giới hạn của hệ thống.

---

## 31. Tác giả

**FreshLens Team**

Đồ án môn:

```text
Xử lý ảnh
```

Ngôn ngữ chính:

```text
Python
```

Framework / Library chính:

```text
PyTorch
Torchvision
Streamlit
Pillow
NumPy
scikit-learn
Matplotlib
OpenCV
```

---

## 32. Disclaimer

FreshLens là dự án học thuật.

Kết quả `fresh / rotten` là phân loại dấu hiệu bề ngoài dựa trên ảnh và dataset.

Không sử dụng kết quả của FreshLens như kết luận về:

- an toàn thực phẩm;
- vi sinh vật;
- độc tố;
- khả năng ăn được;
- chất lượng dinh dưỡng.
