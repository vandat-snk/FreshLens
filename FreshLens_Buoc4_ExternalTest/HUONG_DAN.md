# FreshLens – Bước 4: External / Real-world Test

Mục tiêu: kiểm tra `models\cnn_efficientnet_b0\best.pt` trên ảnh **hoàn toàn mới**, không dùng để train/validation/test nội bộ.

## 1. Đặt gói vào project

Sau khi giải nén, cần có:

`E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens\FreshLens_Buoc4_ExternalTest\EXTERNAL_TEST.py`

Gói dùng trực tiếp các module CNN trong `FreshLens_Buoc3_CNN`, nên không train lại.

## 2. Tạo bộ ảnh ngoài

Nhấp đúp `TAO_THU_MUC.cmd`, hoặc tạo các thư mục sau trong:

`E:\VanDat_\XuLyAnh\FreshLens_external_test`

- apple_fresh
- apple_rotten
- banana_fresh
- banana_rotten
- orange_fresh
- orange_rotten
- tomato_fresh
- tomato_rotten
- unknown

Đợt đầu nên có **5 ảnh mới cho mỗi lớp đã học = 40 ảnh**. Nên ưu tiên ảnh tự chụp bằng điện thoại/camera; nếu lấy Internet thì không lấy từ nguồn dataset cũ. Mỗi lớp nên có thay đổi về góc chụp, nền, khoảng cách và ánh sáng. Một ảnh chỉ nên có một quả chính.

`unknown` là tùy chọn; nên cho khoảng 10 ảnh như xoài, nho, lê, chai nước, bàn tay, đồ vật. Kết quả ở đây chỉ để xem model có đoán bừa với độ tự tin cao hay không. Model hiện chưa có lớp `other/unknown`, vì vậy không được tính accuracy cho thư mục này.

Không copy ảnh từ dataset nội bộ sang bộ external. Script sẽ kiểm tra SHA-256 để bắt bản trùng byte chính xác.

## 3. Chạy đánh giá

Mở CMD:

```cmd
cd /d "E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens"
.\.venv\Scripts\python.exe .\FreshLens_Buoc4_ExternalTest\EXTERNAL_TEST.py --root "E:\VanDat_\XuLyAnh\FreshLens_external_test" --checkpoint "models\cnn_efficientnet_b0\best.pt" --data "data\cnn_dataset_v3" --output "reports\external_test_v1" --device cuda
```

Nếu `reports\external_test_v1` đã tồn tại, không ghi đè. Hãy dùng tên mới như `external_test_v2`.

## 4. File cần gửi lại

Sau khi chạy xong, gửi:

1. `reports\external_test_v1\metrics.json`
2. `reports\external_test_v1\predictions.csv`
3. `reports\external_test_v1\confusion_matrix.png`

Nếu có lỗi, gửi nguyên màn hình CMD.

## 5. Cách đọc kết quả

- `known_metrics.joint_accuracy`: đúng đồng thời loại quả và fresh/rotten trên ảnh mới.
- `exact_internal_duplicate_images`: phải bằng 0. Nếu >0, bộ external không còn độc lập hoàn toàn.
- `errors.csv`: chỉ các ảnh có nhãn biết trước nhưng model dự đoán sai.
- `unknown_predictions.csv`: model buộc phải chọn một trong 8 lớp; dùng để chẩn đoán khả năng đoán bừa, không phải bằng chứng về OOD detection.
- `review_heuristic`: chỉ là cờ xem lại theo ngưỡng điểm 0.70; đây không phải detector unknown đã được huấn luyện hay hiệu chỉnh.

Sau khi external test ổn, bước kế tiếp là nối CNN vào giao diện Streamlit và kiểm tra camera/upload trực tiếp.
