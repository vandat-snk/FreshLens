# FreshLens

FreshLens là đồ án môn Xử lý ảnh bằng Python: phân loại một ảnh có một vật thể chính thành **táo, chuối, cam, cà chua**, sau đó dự đoán nhãn dấu hiệu **tươi/hỏng**. Ứng dụng có cơ chế `other` và ngưỡng từ chối để không ép ảnh số, chữ, đồ vật hoặc thực phẩm chưa học vào một trong bốn loại.

> `fresh/rotten` chỉ là nhãn dấu hiệu bề ngoài theo dữ liệu. Hệ thống không kết luận trái cây ăn được hay an toàn thực phẩm.

## 1. Cấu trúc thư mục

```text
FreshLens/
├─ src/
│  ├─ app.py                 # Giao diện upload/camera, học thuật toán, đánh giá
│  ├─ config.py              # Cấu hình dùng chung và đường dẫn dataset
│  ├─ image_processing.py    # Pillow/EXIF/BGR/letterbox/CLAHE/Gaussian/Canny
│  ├─ features.py            # màu, LBP, hình dạng/Hu, HOG và ablation
│  ├─ dataset.py             # quét dữ liệu, group split, manifest
│  ├─ check_dataset.py       # quality gate trước train
│  ├─ model.py               # SVM, xác suất, open-set decision, artifact
│  ├─ train.py               # baseline/candidate/ablation và lưu mô hình
│  ├─ predict.py             # dự đoán dùng chung cho CLI và UI
│  └─ metrics.py             # confusion matrix, F1, coverage, false acceptance
├─ scripts/generate_demo_dataset.py
├─ tests/
├─ docs/TEAM_ASSIGNMENT.md
├─ data/demo/                # dữ liệu giả lập để smoke-test
├─ artifacts/                # model joblib được tạo sau train
└─ reports/                  # manifest report, metrics, lỗi từng ảnh
```

## 2. Cài đặt trên Windows

Mở PowerShell tại thư mục `FreshLens`:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dataset hoàn toàn có thể lưu ở ổ E. Tạo cấu trúc thư mục:

```powershell
.\setup_dataset.ps1
$env:FRESHLENS_DATASET_ROOT = "E:\FreshLens\dataset"
```

Biến môi trường chỉ có hiệu lực trong cửa sổ PowerShell hiện tại. Muốn đặt cố định, cấu hình biến môi trường người dùng trong Windows hoặc dùng tham số `--root` ở các lệnh bên dưới.

## 3. Cấu trúc dataset thật

```text
E:\FreshLens\dataset\
├─ raw\
│  ├─ apple\fresh\*.jpg
│  ├─ apple\rotten\*.jpg
│  ├─ banana\fresh\*.jpg
│  ├─ banana\rotten\*.jpg
│  ├─ orange\fresh\*.jpg
│  ├─ orange\rotten\*.jpg
│  ├─ tomato\fresh\*.jpg
│  ├─ tomato\rotten\*.jpg
│  └─ other\numbers|objects|other_food\*.jpg
└─ groups.csv              # khuyến nghị nếu một quả có nhiều góc chụp
```

Tên ảnh dạng `apple_001__view02.jpg` sẽ được gom cùng `group_id=apple_001` nếu chưa có `groups.csv`. Tốt nhất tạo `groups.csv`:

```csv
relative_path,group_id,source
raw/apple/fresh/apple_001__view01.jpg,apple_001,phone_A
raw/apple/fresh/apple_001__view02.jpg,apple_001,phone_A
```

Không đưa ảnh cùng quả/cùng buổi chụp sang nhiều tập. `other` không có nhãn `fresh/rotten`.

### Số lượng thực tế nên thu thập

Để một mô hình nhỏ nhưng có cơ sở đánh giá, nên có tối thiểu khoảng 30–50 **group độc lập** cho mỗi tổ hợp loại × tình trạng và 50–100 ảnh `other` đa dạng. Tốt hơn là 80–120 group cho mỗi tổ hợp. Một group là một quả/buổi chụp, không phải nhiều crop của cùng một ảnh. Nếu ít dữ liệu hơn, vẫn chạy được nhưng phải ghi rõ giới hạn và không dùng accuracy demo để khẳng định khả năng tổng quát.

## 4. Chạy bằng demo dataset ngay

Nếu chưa có ảnh thật, chạy smoke-test ở thư mục dự án:

```powershell
python scripts\generate_demo_dataset.py
python -m src.dataset --root data/demo
python -m src.check_dataset --root data/demo
python -m src.train --root data/demo --manifest data/demo/manifest.csv --output artifacts/freshlens_model.joblib
python -m src.predict data/demo_uploads/apple_fresh_demo.jpg
streamlit run src/app.py
```

Demo dataset là ảnh tổng hợp để kiểm tra luồng phần mềm, **không được dùng để công bố độ chính xác nhận dạng ảnh chụp đời thực**.

## 5. Train dữ liệu thật trên ổ E

```powershell
python -m src.dataset --root "E:\FreshLens\dataset"
python -m src.check_dataset --root "E:\FreshLens\dataset" --manifest "E:\FreshLens\dataset\manifest.csv"
python -m src.train --root "E:\FreshLens\dataset" --manifest "E:\FreshLens\dataset\manifest.csv" --ablation
python -m src.metrics --root "E:\FreshLens\dataset" --manifest "E:\FreshLens\dataset\manifest.csv" --split test
streamlit run src/app.py
```

Lệnh train mặc định so sánh:

* baseline: màu + SVM tuyến tính;
* candidate: màu + texture + shape + HOG + SVM RBF.

Thêm `--ablation` để chạy thêm màu+texture và màu+texture+shape. Scaler nằm trong pipeline SVM. Tham số được chọn trên train folds; ngưỡng `probability` và `margin` chọn trên validation; test chỉ dùng sau khi đã khóa cấu hình.

## 6. Thử ảnh upload/camera

Trong giao diện:

1. Chọn `Chẩn đoán`.
2. Upload ảnh hoặc chụp camera.
3. Đọc trạng thái: `Đã chấp nhận`, `Chưa đủ tin cậy`, `Ngoài phạm vi hỗ trợ`, `Tệp ảnh lỗi`.
4. Chỉ khi loại quả được chấp nhận, hệ thống mới hiển thị tình trạng.
5. Mở `Xem ảnh trung gian` để xem letterbox, CLAHE, Gaussian, Canny, LBP và gradient.

CLI JSON và giao diện lấy nhãn/xác suất/trạng thái từ cùng `FreshLensPredictor`; không có bước UI tự chọn nhãn khác.

## 7. Kiểm thử

```powershell
pytest -q
python -m compileall src scripts tests
```

Ma trận kiểm thử cần có: ảnh xám, PNG trong suốt, EXIF xoay, tên đường dẫn tiếng Việt, ảnh quá tối/quá sáng, ảnh số/chữ, đồ vật, thực phẩm khác, ảnh nhiều vật thể và mô hình sai số chiều. Test phần mềm chỉ chứng minh luồng chạy; năng lực nhận dạng phải được đo trên ảnh thật có nhãn và chia group độc lập.

## 8. Giới hạn cần nói khi bảo vệ

HOG/LBP/Canny mô tả cấu trúc/biên/kết cấu chứ không tự chứng minh vùng hỏng. Lớp `other` hữu hạn và không bao phủ mọi ảnh lạ. Một xác suất cao trên một ảnh không phải accuracy của cả tập. Một ảnh có nhiều vật thể nằm ngoài phạm vi phiên bản này; đây là phân loại toàn ảnh, chưa phải detection/counting.
