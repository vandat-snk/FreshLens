# Bàn giao FreshLens CNN

Bản nộp dùng EfficientNet B0 tám lớp, fruit-first, gate prototype/logistic và quality rejection.
Giữ nguyên checkpoint và ngưỡng đã lưu. Kiểm thử phần mềm không thay thế đánh giá camera độc lập.

## Cài và chạy

Windows có Python 3.13 và Internet: mở RUN_FRESHLENS.bat. Nếu thiếu .venv, script cài CPU,
chạy kiểm thử runtime rồi mở app. Lần sau dùng lại môi trường. Hoặc:

```powershell
powershell -File SETUP_FRESHLENS.ps1 -Device cpu
.\run_app.bat
```

Máy NVIDIA tương thích có thể chọn -Device cu128. Chẩn đoán không cần MongoDB/Cloudinary;
công cụ cloud dùng requirements-db.txt riêng. Không cần đường dẫn máy cá nhân.

## Demo và kiểm thử

1. Tải ảnh hoặc chụp camera, nhấn Phân tích; tải JSON để giữ bằng chứng.
2. Đổi ảnh A sang B: kết quả và Grad-CAM cũ phải biến mất.
3. Thử file lỗi, ảnh tối/mờ/nhỏ và ảnh ngoài bốn loại hỗ trợ.
4. Mở Xem ảnh xử lý và Grad-CAM, tạo heatmap cho ảnh hiện tại và tải PNG.
5. Mở Đánh giá, Giải thích, About. Ảnh trên trang Giải thích được ghi rõ là demo cố định.

Chạy tự động: `python -m pytest tests/tv3 tests/submission -q`.
Full integration cần ảnh gốc: xem TESTING.md. GitHub Actions chạy CPU trên Windows;
chỉ coi workflow pass sau khi GitHub đã chạy xong. Camera vật lý cần thử trên máy demo.

## Bộ ảnh camera thật

CSV nhãn thủ công có cột path,fruit,status,split,group_id,source,quality_label.
Fruit là apple/banana/orange/tomato hoặc other. Status là fresh/rotten cho known,
để trống cho other. Split là dev/calibration/final. Group theo quả vật lý, không
phải từng ảnh; cùng quả không được vào nhiều split. Quality label good/bad hoặc trống.

```cmd
python PREPARE_CAMERA_MANIFEST.py --root "D:\camera" --labels "D:\camera\labels.csv" --output "reports\camera_audit_01"
python EVALUATE_TV3.py --root "D:\camera" --manifest "reports\camera_audit_01\manifest.csv" --split final --quality --device cpu --output "reports\camera_final_cpu_01"
```

Công cụ xuất hash, thống kê, quality metadata và kiểm tra giao với manifest gốc.
Không tự suy ra độc lập theo quả vật lý hay lịch sử tuning. Khóa ảnh final, model,
gate và quality config trước đánh giá. Không dùng kết quả final để chỉnh ngưỡng.

## Benchmark checkpoint đã có

JSON candidate là danh sách mục có name,checkpoint,gate_npz,gate_meta. Đường dẫn
tương đối theo file JSON; gate phải khớp model, cùng hợp đồng preprocessing production.
Ví dụ một baseline (thêm candidate thật sau khi đã huấn luyện):

```json
[{"name":"baseline","checkpoint":"models/cnn_efficientnet_b0/best.pt","gate_npz":"models/cnn_efficientnet_b0/open_set_gate.npz","gate_meta":"models/cnn_efficientnet_b0/open_set_gate.json"}]
```

```cmd
python BENCHMARK_MODELS.py --candidates "candidates.json" --root "D:\camera" --manifest "reports\camera_audit_01\manifest.csv" --split dev --device cpu --output "reports\comparison_cpu_01"
```

Chạy cùng ảnh và quality policy; xuất report từng candidate và comparison.json gồm
fruit/condition/joint, per-class và macro-F1, coverage/FAR, latency mean/p95.
Chạy riêng --device cuda nếu có GPU. Không tự promote model, không tạo số cho candidate chưa có.

## Trạng thái nghiệm thu

Đã có pipeline CNN chung, manifest khóa, quality/gate, evaluator, live Grad-CAM,
JSON/PNG export, kiểm thử, cài CPU và phân công đúng tên. TTA bị tắt do chưa kiểm định
với gate. Chưa có lớp other học trực tiếp, benchmark Candidate A/B mới và camera final
độc lập đạt KPI. External V2 là calibration. Ảnh demo/tổng hợp không thay thế camera thật.

## Kiểm tra lại ngày 20 tháng 10 năm 2026

Đã tích hợp pipeline TV1 từ nhánh feat/data-v2: inventory camera/other, kiểm tra
provenance và duplicate, candidate builder có human review, khóa manifest và adapter.
Hướng dẫn chi tiết ở dataset_v2/README.md. Các snapshot V5/camera trên máy tác giả
không tự xuất hiện trong clone; dùng PREPARE_CAMERA_MANIFEST.py cho bộ ảnh mới,
hoặc tạo candidate baseline từ V3 với CLI dataset_v2 trước khi dùng workflow V5.

Bản chạy lại External V2 gồm 81 known và 22 unknown nằm ở
`eval_results/submission_external_v2_cpu/run` (metrics, predictions, CM và manifest).
CNN joint accuracy: 75/81 = 92,59%. Khi bật quality như app: nhận 53/81 known,
đúng đồng thời và được nhận 51/81 = 62,96%; 25 known bị quality từ chối và 3 bị gate
từ chối. Unknown được từ chối 22/22 trên bộ đã dùng calibration; không suy rộng
thành hiệu năng ảnh lạ ngoài thực tế. Không có nhãn good/bad thủ công nên chưa
kết luận 25 ảnh quality-rejected là từ chối đúng hay sai. Ngưỡng quality cần hiệu
chuẩn trên development camera riêng, không thay tùy ý để tăng số trên final test.

Benchmark tool đã chạy được với baseline production trên cùng 103 ảnh (CPU).
Đây là kiểm tra công cụ một candidate, chưa phải bằng chứng V2 tốt hơn V1.
