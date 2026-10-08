# TV3 — Sửa inference, evaluation và Grad-CAM

Bản sửa dùng CNN 8 lớp và gate production hiện có. Mục tiêu là thống nhất hành vi giữa app, CLI và evaluator, đồng thời báo cáo đúng cả ảnh bị từ chối. Chưa có kết luận accuracy camera mới từ các kiểm tra phần mềm.

## 1. Luồng dùng chung

`APP_CNN_V2.py`, `PREDICT_CNN_V2.py` và `EVALUATE_TV3.py` gọi `FreshLensPredictor` trong `freshlens_ai/inference/cnn_predict.py`.

Predictor gọi `load_model()` để kiểm tra checkpoint; `load_gate()` để kiểm tra gate tương ứng; `analyze_bytes()` để dùng nguyên EXIF → RGB/alpha trên nền trắng → letterbox → normalization → CNN → fruit-first → gate production. Không tự viết lại decoder hoặc sáu đặc trưng gate.

Gate thiếu, sai checkpoint hoặc lỗi suy luận sẽ báo lỗi. Loader còn kiểm tra ngưỡng hữu hạn trong [0,1], ngưỡng NPZ/JSON khớp nhau, kích thước các mảng, scaler dương và prototype hợp lệ. Ngưỡng lấy từ artifact. Không fallback thành “supported”. Model và artifact gate không được thay đổi trong bản sửa này.

Kết quả có ba trạng thái:

- `accepted`: hiển thị loại quả và tình trạng.
- `quality_rejection`: yêu cầu chụp lại, kèm lý do.
- `openset_rejection`: chưa đủ cơ sở nhận diện trong phạm vi hỗ trợ.

`gate_supported` là quyết định riêng của gate; `supported`/`is_supported` là quyết định cuối sau chính sách chất lượng. Các trường nhãn và xác suất CNN vẫn được giữ để đánh giá khi bị từ chối; UI không trình bày chúng như kết luận được chấp nhận.

## 2. Chạy dự đoán

Tại thư mục gốc repo, dùng Python của virtualenv đã cài requirements và PyTorch phù hợp:

```bash
python PREDICT_CNN_V2.py --image "path/to/fruit.jpg" --device cpu
```

CLI hiện trả thêm trạng thái gate, điểm hỗ trợ, ngưỡng, quality diagnostics và latency. Đây là thay đổi có chủ đích so với CLI cũ chỉ phân loại. Gate mặc định nằm cạnh checkpoint; có thể chỉ rõ `--gate-npz` và `--gate-meta`.

App chạy bằng:

```bash
python -m streamlit run APP_CNN_V2.py
```

Kiểm tra chất lượng đang tắt mặc định. Có checkbox thử nghiệm trong sidebar; đổi chính sách sẽ xóa kết quả cũ. CLI có `--quality` để bật bộ ngưỡng thử nghiệm. App, CLI và evaluator mặc định cùng đọc `config/quality.json`.

## Cấu hình quality dùng chung

`config/quality.json` chứa đúng bốn trường `min_size`, `blur_threshold`, `dark_threshold`, `bright_threshold`. Các giá trị hiện tại vẫn là ngưỡng thử nghiệm; tệp cấu hình không phải bằng chứng calibration.

- CLI và evaluator nhận `--quality-config` để chọn tệp khác.
- Evaluator vẫn cho ghi đè từng ngưỡng qua các tham số cũ. Sau khi chạy, cấu hình hiệu lực được xuất vào `quality_config.json` trong thư mục kết quả.
- App dùng biến môi trường `FRESHLENS_QUALITY_CONFIG` nếu cần chọn tệp khác. Nếu không đặt, app dùng tệp mặc định của repo.
- Kết quả CLI/app và báo cáo evaluator có `quality_config_sha256`, được tính từ nội dung cấu hình hiệu lực. Dùng hash để đối chiếu cấu hình giữa các nơi.
- Khi nội dung tệp thay đổi, app nạp runtime tương ứng và xóa kết quả cũ ở lần rerun kế tiếp.

Sau khi evaluator đã tạo `eval_results/tv3_dev_quality/quality_config.json`, có thể dùng đúng tệp đó cho app. Lệnh Linux:

```bash
FRESHLENS_QUALITY_CONFIG=eval_results/tv3_dev_quality/quality_config.json python -m streamlit run APP_CNN_V2.py
```

PowerShell trên Windows:

```powershell
$env:FRESHLENS_QUALITY_CONFIG = "eval_results/tv3_dev_quality/quality_config.json"
python -m streamlit run APP_CNN_V2.py
```

Đường dẫn trên chỉ tồn tại sau khi đã chạy evaluator vào thư mục đó. Để chạy ngay với cấu hình mặc định, dùng `python -m streamlit run APP_CNN_V2.py` như bình thường.

## 3. Manifest đánh giá

Evaluator yêu cầu các cột `path,fruit,status,split`. Đường dẫn ảnh tương đối với `--root`.

```csv
path,fruit,status,split,group_id,sha256,quality_label
known/apple_01.jpg,apple,fresh,dev,apple_01,,good
unknown/mango_01.jpg,other,,dev,mango_01,,good
known/blur_01.jpg,banana,rotten,dev,banana_01,,bad
```

`other`, `unknown`, `unsupported` biểu thị ảnh ngoài phạm vi. Không dùng folder chứa `::`. `quality_label` tùy chọn là nhãn người đánh giá `good` hoặc `bad`, không phải nhãn tự sinh từ ngưỡng. Cần quy tắc gán nhãn chất lượng nhất quán trước khi đo.

Nên cung cấp `group_id` và `sha256`: evaluator kiểm tra group không chạy qua nhiều split và đối chiếu hash nếu có. Khi thiếu, báo cáo ghi rõ số ảnh thiếu metadata; CSV kết quả luôn lưu hash thực tế. Những kiểm tra này không chứng minh mọi group là một quả vật lý độc lập.

```bash
python EVALUATE_TV3.py --root "path/to/images" --manifest "path/to/manifest.csv" --split dev --device cpu --output "eval_results/tv3_dev_baseline"
```

Đánh giá chính sách chất lượng riêng trên cùng tập development:

```bash
python EVALUATE_TV3.py --root "path/to/images" --manifest "path/to/manifest.csv" --split dev --device cpu --quality --min-size 100 --blur-threshold 100 --dark-threshold 40 --bright-threshold 215 --output "eval_results/tv3_dev_quality"
```

Các ngưỡng trên là điểm bắt đầu thử nghiệm, chưa phải ngưỡng được chứng minh phù hợp. Không chọn ngưỡng bằng final test.

Đầu ra:

- `metrics.json`: CNN-only, coverage, accepted-only accuracy, end-to-end success, gate outcomes, FAR, quality diagnostics, ECE, latency trung bình/p95; hash checkpoint/gate/manifest và cấu hình.
- `predictions.csv`: mỗi ảnh một dòng, gồm nhãn thật/dự đoán, confidence, điểm gate, quality score, lý do từ chối, hash và đường dẫn ảnh trong gallery.
- `confusion_matrix.png`: tám lớp, trên toàn bộ known, nếu tập có ảnh known.
- `error_gallery/`: ảnh known sai hoặc bị từ chối, ảnh unknown được nhận nhầm; tên file hợp lệ trên Windows.
- `manifest.csv`: bản sao manifest đầu vào để đối chiếu.

Ảnh bị từ chối vẫn nằm trong mẫu số CNN-only và end-to-end. Nếu thiếu unknown thì FAR/rejection là `null`, không phải 0% hoặc 100%. Thiếu nhãn good/bad thì các tỷ lệ chất lượng tương ứng cũng là `null`. Lỗi đọc ảnh hoặc sai hash làm dừng đánh giá, không âm thầm bỏ ảnh. Thư mục đầu ra phải mới để tránh ghi đè bằng chứng cũ.

Macro-F1 lấy trung bình trên danh sách lớp cố định; balanced accuracy tính trên các lớp có ground truth. Khi lớp vắng mặt cần đọc cả support, không so macro-F1 giữa các bộ dữ liệu khác thành phần một cách máy móc.

Latency gồm decode, quality diagnostics và toàn bộ CNN/gate, kể cả ảnh quality-rejected, để giữ dự đoán thô cho việc đánh giá. Loại một lượt warm-up, đồng bộ CUDA khi dùng GPU. Không bao gồm thời gian khởi động model hoặc vẽ UI. Chạy CPU và CUDA vào hai thư mục khác nhau nếu máy có GPU phù hợp.

## 4. Grad-CAM

```bash
python GENERATE_GRADCAM_DEMO.py --image "path/to/fruit.jpg" --device cpu --output "eval_results/cam_apple.png"
```

Dùng ảnh quả thật, không dùng screenshot giao diện. Target là joint class do fruit-first decoder chọn. Heatmap phủ lên đúng canvas letterbox 224×224; file JSON bên cạnh lưu tên lớp và hash. Grad-CAM diễn giải dự đoán CNN, không xác nhận gate chấp nhận và không phải segmentation vùng hỏng. Hook được gỡ sau khi dùng.

## 5. TTA và final test

TTA tắt; gọi `use_tta=True` báo rõ chưa được xác thực với gate production. Chỉ triển khai lại thành một thí nghiệm riêng sau khi xác định cách kết hợp score/embedding và hiệu chỉnh gate cho luồng đó.

Evaluator không huấn luyện model, không chọn threshold và không tự chứng nhận tập ảnh là final test độc lập chỉ vì split có tên `test`. Nhóm cần khóa riêng model, gate, quality policy và manifest trước khi chạy final test. External V2 đã dùng calibration không được dùng làm bằng chứng final độc lập.

## 6. Kiểm tra trước khi bàn giao

```bash
python -m pytest tests/tv3 -q
python -m tests.refactor.TEST_INFERENCE_REFACTOR
python -m tests.refactor.TEST_PREDICT_ENTRYPOINT_REFACTOR
python -m tests.refactor.TEST_APP_REFACTOR
```

Full suite đã thêm test TV3. Nó vẫn cần ảnh dataset gốc cho bước data:

```bash
python -m tests.integration.TEST_FULL_PIPELINE_V2 --root "path/to/original_dataset" --data "data/cnn_dataset_v3"
```

Kiểm tra thủ công: upload ảnh thật; đổi ảnh; đổi checkbox chất lượng; ảnh lỗi; ảnh quá mờ/tối; camera; đối chiếu cùng ảnh qua CLI. Test tổng hợp xác nhận tính đúng của code, không đo accuracy trái cây hoặc khả năng tổng quát hóa.

## 7. Kết quả kiểm tra bản sửa

Đã chạy trên môi trường local CPU:

- 30 test TV3 (sau lần bổ sung cấu hình chung và validation gate): preprocessing/gate tương đương production; ảnh EXIF/alpha/xám/WEBP; quality rejection vẫn giữ dự đoán thô; lỗi gate không bị nuốt; mẫu số và bốn trường hợp CNN/gate; manifest và tên file Windows; xuất JSON/CSV/CM/gallery; Grad-CAM thật với checkpoint production; trạng thái Streamlit khi đổi ảnh, đổi chính sách và file hỏng.
- Các nhóm regression model, training, evaluation, orchestration, inference, gate builder, CLI và app đã pass.
- Audit import production và smoke test checkpoint + gate đã pass.

Chưa chạy bước data của full suite vì workspace không có ảnh dataset gốc. Chưa đo trên bộ final/camera độc lập, chưa kiểm tra camera trình duyệt thực tế và chưa xác nhận trên máy Windows/GPU. Test tên file kiểm tra tính hợp lệ về ký tự, không thay thế việc chạy trên Windows. Các tỷ lệ accuracy trong báo cáo cũ không được cập nhật từ những test tổng hợp này.


## Những việc cần dữ liệu thật để nghiệm thu

1. TV1/TV3 thống nhất manifest development/calibration và một final test độc lập; có known, unknown và nhãn chất lượng good/bad. Rà ảnh trùng và group trước khi khóa.
2. Chạy baseline quality OFF, rồi thử ngưỡng quality trên development/calibration. Đọc coverage, end-to-end success, good false rejection và bad rejection cùng nhau; không chỉ tối ưu một chỉ số.
3. Bàn giao file quality đã chọn và hash cho TV4; dùng đúng model/gate/config đó khi chạy final test. Chưa chỉnh `config/quality.json` thành “đã calibrated” khi chưa có dữ liệu chứng minh.
4. Grad-CAM: chọn ảnh thật cho ca đúng, ca sai và ca bị gate từ chối; giải thích vì sao heatmap không phải mặt nạ vùng hỏng.
5. Chạy benchmark CPU, và GPU nếu máy demo có hỗ trợ, vào hai thư mục riêng. TTA vẫn là thí nghiệm chưa thực hiện; nhóm cần làm thí nghiệm hoặc thống nhất hoãn hạng mục này.

Không có báo cáo final thực tế mới trong bản sửa. Chỉ số sidebar đã đổi tên thành `known_calibration_accept_rate`; metadata cũ vẫn giữ trường alias `known_validation_accept_rate` để tương thích lịch sử, nhưng không dùng alias đó làm bằng chứng validation độc lập.
