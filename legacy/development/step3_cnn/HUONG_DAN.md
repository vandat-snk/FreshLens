# FreshLens — Bước 3: fine-tune EfficientNet-B0

Gói này thêm mô hình CNN cho đồ án nhận diện **táo, chuối, cam, cà chua** và hai trạng thái **tươi / thối**. Bước hiện tại là train và kiểm tra validation. Sau khi có mô hình tốt, dùng `predict_image` để nối vào chức năng upload/camera của ứng dụng.

## 1. Đặt đúng thư mục

Giải nén để file hướng dẫn này nằm tại:

```text
E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens\FreshLens_Buoc3_CNN\HUONG_DAN.md
```

Folder `FreshLens_Buoc3_CNN` phải nằm cùng cấp với `.venv`, `src`, `data`, `models`. Nếu giải nén thành hai folder `FreshLens_Buoc3_CNN` lồng nhau, chuyển folder chứa các file `.py` ra đúng cấp trên.

Gói hoạt động trực tiếp với ảnh trên ổ E và manifest đã xuất từ MongoDB. Không cần mở lại MongoDB hoặc tải lại dataset để train.

Các file đầu vào phải có:

```text
data\cnn_dataset_v3\manifest.csv
data\cnn_dataset_v3\final_split_report.json
data\cnn_dataset_v3\dataset_lock.json
```

Dataset đã chốt theo báo cáo của bạn: **3.519 train, 1.187 validation, 1.163 test**, tổng 5.869 ảnh. SHA-256 của manifest:

```text
9be1db4f31ad753a75e54761dceda3aa4081d3add3b67baedccd4253ad83d291
```

Đường dẫn ảnh gốc là:

```text
E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset
```

Giữ nguyên bộ chia này trong quá trình thử mô hình. Code kiểm tra manifest, báo cáo, lock, nhóm ảnh, SHA-256 và nhãn; nếu ảnh bị sửa sau bước chuẩn bị dữ liệu, chương trình dừng và chỉ ra file.

## 2. Cài PyTorch và kiểm tra GPU bằng CMD

Mở **Command Prompt (CMD)** và chạy từng lệnh:

```bat
cd /d "E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens"
FreshLens_Buoc3_CNN\CAI_CNN_GPU.cmd
```

Script cài **torch 2.10.0 + torchvision 0.25.0, bản CUDA 12.8** vào `.venv` hiện có. Cache tải xuống và thư mục tạm được đặt trong `.cache` của dự án trên ổ E. Lần đầu cần mạng và vài GB dung lượng trống. Không cần tự cài CUDA Toolkit để dùng các wheel PyTorch này; cần driver NVIDIA hoạt động.

Đây là cặp phiên bản cố định cho gói, có wheel Windows Python 3.13; không phải tuyên bố về phiên bản mới nhất. Các lệnh pip chính trong script là:

```bat
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r FreshLens_Buoc3_CNN\requirements-cnn.txt
.\.venv\Scripts\python.exe FreshLens_Buoc3_CNN\CHECK_GPU.py
```

Kết quả đạt phải có dạng:

```text
[OK] PyTorch 2.10.0+cu128; torchvision 0.25.0+cu128
[OK] Device: NVIDIA GeForce RTX 2050; forward/backward passed
[OK] Report: ...\reports\cnn_environment.json
```

Tên GPU có thể khác tùy máy. Nếu có `[ERROR]`, giữ toàn bộ log lỗi và gửi lại trước khi train. Chương trình không tự chuyển qua CPU khi CUDA lỗi. `CHECK_GPU.py` chỉ kiểm tra phần mềm/GPU, chưa huấn luyện trên dữ liệu quả.

## 3. Bắt đầu train

Sau khi kiểm tra GPU đạt, chạy trong chính cửa sổ CMD đó:

```bat
.\.venv\Scripts\python.exe FreshLens_Buoc3_CNN\TRAIN_CNN.py --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" --data data\cnn_dataset_v3 --output models\cnn_efficientnet_b0
```

Không chạy `src\train.py` hoặc file batch train cũ để thực hiện bước CNN này. Chúng thuộc luồng mô hình trước đó. Gói mới tự tạo các kết quả CNN trong `models\cnn_efficientnet_b0`.

Đầu tiên chương trình kiểm tra **4.706 ảnh train + validation**. Sau đó tải trọng số ImageNet chính thức của EfficientNet-B0, khoảng 20 MB, vào `.cache\torch`. Khi xuất hiện các dòng dạng sau thì đang train thật:

```text
[EPOCH 1/23] warmup | device=cuda | batch=8 x accumulation=2
  [TRAIN] batch 50/440 | loss ...
[VAL] fruit_acc=... | joint_acc=... | joint_macro_f1=... | BEST SAVED
```

Các dấu `...` chỉ là minh họa, không phải kết quả dự kiến. Mỗi epoch là một lượt qua toàn bộ train, rồi đánh giá toàn bộ validation. Tốc độ phụ thuộc GPU, ổ đĩa và tải máy; xem thời gian epoch thực tế trong `history.csv`.

Thiết lập mặc định cho RTX 2050 4 GB:

| Thành phần | Thiết lập |
|---|---|
| Mạng | EfficientNet-B0, khởi tạo từ ImageNet |
| Nhãn đầu ra | 8 tổ hợp loại quả × trạng thái |
| Ảnh | RGB, sửa chiều EXIF, giữ tỉ lệ và đệm trắng thành 224 × 224 |
| Chuẩn hóa | Mean/std ImageNet |
| Tăng cường ảnh train | Lật ngang, xoay/dịch/đổi kích thước nhẹ, chỉnh màu nhẹ |
| Warmup | 3 epoch: chỉ học lớp phân loại cuối |
| Fine-tune | Tối đa 20 epoch: học thêm backbone, learning rate nhỏ |
| Batch | 8 ảnh/lần, tích lũy gradient 2 lần, tương đương 16 ảnh/bước tối ưu |
| Bộ nhớ/tốc độ | AMP khi train trên CUDA, workers=0 phù hợp Windows |
| BatchNorm | Giữ running statistics của mô hình học sẵn vì batch nhỏ |
| Optimizer | AdamW; learning rate warmup 0,001; fine-tune backbone 0,0001 / head 0,0003 |
| Điều chỉnh | Cosine learning rate, label smoothing 0,05 |
| Chọn mô hình | Joint macro-F1 trên validation; nếu bằng nhau, chọn loss thấp hơn |
| Dừng sớm | 6 epoch fine-tune liên tiếp không cải thiện tiêu chí trên |

Tiền xử lý đệm trắng được chọn để giữ toàn bộ quả, thay cho center crop mặc định của trọng số ImageNet. Validation, test và ảnh upload dùng chung hàm tiền xử lý, không tăng cường ảnh ngẫu nhiên. Đánh giá và dự đoán dùng float32; AMP chỉ dùng khi train.

Code dùng chung quy tắc: cộng điểm tươi/thối để chọn loại quả, sau đó chọn trạng thái trong loại quả đó. Vì vậy loại quả trả ra là loại có điểm tổng cao nhất. `joint accuracy` và `joint macro-F1` được tính theo đúng quy tắc này.

## 4. Đọc kết quả và gửi lại

Sau dòng `[DONE]`, gửi hai file:

```text
models\cnn_efficientnet_b0\training_summary.json
models\cnn_efficientnet_b0\history.png
```

| File | Dùng để làm gì |
|---|---|
| `best.pt` | Mô hình tốt nhất theo validation, dùng cho dự đoán |
| `last.pt` | Trạng thái ở epoch hoàn tất gần nhất để tiếp tục train |
| `training_summary.json` | Epoch tốt nhất, các chỉ số validation, xác nhận chưa đánh giá test |
| `best_validation.json` | Chỉ số và confusion matrix của mô hình tốt nhất |
| `history.csv`, `history.png` | Diễn biến loss và chỉ số validation qua các epoch |
| `validation_predictions.csv` | Dự đoán từng ảnh validation của mô hình tốt nhất |
| `validation_errors.csv` | Những ảnh validation còn nhận sai |
| `validation_confusion_matrix.png` | Các lớp đang nhầm lẫn với nhau |
| `config.json`, `environment.json` | Cấu hình và môi trường để tái lập thí nghiệm |

- **Fruit accuracy**: đúng loại quả, bất kể trạng thái.
- **Condition accuracy**: đúng tươi/thối theo trạng thái trả ra; có thể tính cả ảnh đoán sai loại quả.
- **Joint accuracy**: phải đúng cả loại quả lẫn trạng thái.
- **Joint macro-F1**: trung bình F1 của 8 lớp, giúp theo dõi cả lớp ít ảnh. Dùng chỉ số này chọn checkpoint; vẫn báo riêng accuracy cho đồ án.

Các số lưu dạng từ 0 đến 1; ví dụ `0.92` là 92%. Điểm dự đoán của một ảnh như `fruit_score` là điểm mô hình, **không phải accuracy đã đo** và chưa được hiệu chỉnh xác suất.

Chưa thể cam kết 95% hay 99% trước khi có kết quả. Accuracy validation giúp phát triển mô hình; accuracy test được đo sau khi chốt mô hình. Ảnh điện thoại thực tế cần một bộ kiểm tra riêng có nhãn, đa dạng nền/ánh sáng/góc chụp. Việc gom các cặp gần giống đã giảm nguy cơ rò rỉ nhưng chưa chứng minh mọi nhóm là các quả vật lý độc lập.

## 5. Nếu phải dừng hoặc bị gián đoạn

Có thể nhấn `Ctrl+C`. Nếu đã có `last.pt`, chạy lại đúng lệnh ban đầu và thêm `--resume`:

```bat
.\.venv\Scripts\python.exe FreshLens_Buoc3_CNN\TRAIN_CNN.py --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" --data data\cnn_dataset_v3 --output models\cnn_efficientnet_b0 --resume
```

Tiếp tục từ epoch đã hoàn tất gần nhất; phần epoch đang dở sẽ chạy lại. Giữ nguyên dữ liệu, đường dẫn và các tham số. Nếu bị dừng trước khi epoch đầu tiên hoàn tất thì chưa có `last.pt`; dùng một tên output mới, chẳng hạn `models\cnn_efficientnet_b0_retry1`.

Nếu GPU báo `CUDA out of memory`, đóng ứng dụng đang chiếm GPU và bắt đầu một run mới với batch nhỏ hơn:

```bat
.\.venv\Scripts\python.exe FreshLens_Buoc3_CNN\TRAIN_CNN.py --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" --data data\cnn_dataset_v3 --output models\cnn_efficientnet_b0_b4 --batch-size 4 --accumulation 4
```

Khi tiếp tục run này phải giữ `--batch-size 4 --accumulation 4 --output models\cnn_efficientnet_b0_b4` rồi thêm `--resume`. Sau đó mọi lệnh dự đoán/đánh giá cần trỏ checkpoint vào `models\cnn_efficientnet_b0_b4\best.pt`.

## 6. Thử một ảnh mới sau khi train xong

Ví dụ lưu một ảnh tại `E:\anh_thu\qua.jpg`, rồi chạy:

```bat
.\.venv\Scripts\python.exe FreshLens_Buoc3_CNN\PREDICT_CNN.py --image "E:\anh_thu\qua.jpg" --checkpoint models\cnn_efficientnet_b0\best.pt
```

Mặc định dự đoán trên CPU, có thể thêm `--device cuda`. Kết quả JSON gồm `fruit`, `condition`, điểm từng loại quả và điểm từng trạng thái. File ảnh cần tồn tại đúng đường dẫn bạn nhập.

Hàm `predict_image(model, image_bytes, device)` dùng được cho cả ảnh upload và ảnh chụp camera. Ứng dụng hiện tại chưa tự chuyển sang CNN chỉ bằng việc train; bước tích hợp UI sẽ nạp `best.pt` rồi gọi hàm này cho ảnh mà người dùng chọn.

Phạm vi hiện tại là **một loại quả chính trong ảnh, thuộc 4 loại đã học**. Mô hình chưa được huấn luyện để từ chối ảnh đồ vật, loại quả lạ hoặc phân tích nhiều quả khác loại trong cùng ảnh. Với những ảnh đó, mô hình vẫn có thể trả một nhãn trong 4 loại. Cần thêm dữ liệu và đánh giá riêng để mở rộng phạm vi; không dùng một ngưỡng điểm tùy ý để tuyên bố đã nhận biết được mọi ảnh lạ.

## 7. Đánh giá test sau khi chốt mô hình

Lệnh train không đọc ảnh test, không dùng test để chọn epoch hay điều chỉnh tham số. Khi hoàn tất quyết định dựa trên validation, dùng riêng lệnh sau:

```bat
.\.venv\Scripts\python.exe FreshLens_Buoc3_CNN\EVALUATE_CNN.py --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" --data data\cnn_dataset_v3 --checkpoint models\cnn_efficientnet_b0\best.pt --split test --output reports\cnn_test_final
```

Kết quả `reports\cnn_test_final\metrics.json` mới là số đo test của mô hình đã chọn. Kèm danh sách từng dự đoán, ảnh sai và confusion matrix. Không chọn nhiều mô hình theo điểm test rồi dùng chính test đó báo cáo như một lần đánh giá độc lập.

## 8. Cơ sở và kiểm thử gói

- Lệnh cài PyTorch chính thức: https://pytorch.org/get-started/previous-versions/
- Wheel torch Windows/Python 3.13: https://download.pytorch.org/whl/cu128/torch/
- Wheel torchvision Windows/Python 3.13: https://download.pytorch.org/whl/cu128/torchvision/
- EfficientNet-B0: https://docs.pytorch.org/vision/0.25/models/generated/torchvision.models.efficientnet_b0.html
- AMP: https://docs.pytorch.org/docs/2.10/amp.html
- Transfer learning: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

Xem `QA_REPORT.json` để biết những phần đã được chạy kiểm thử. Kiểm thử tại môi trường chuẩn bị gói sử dụng CPU và ảnh giả lập; chưa thay thế việc chạy `CHECK_GPU.py` trên Windows/RTX 2050 của bạn, và không tạo ra accuracy cho bộ ảnh thật.
