# Câu hỏi bảo vệ trọng tâm

## 1. Vì sao phải resize và padding?

Mô hình cần vector có cùng số chiều. Letterbox đưa cạnh dài về 224, giữ tỷ lệ và bù phần thiếu thay vì kéo méo vật thể. Padding vẫn đi vào histogram/HOG nên phải dùng giống nhau ở train và upload.

## 2. Vì sao có RGB/BGR và EXIF?

Pillow đọc ảnh theo RGB còn hợp đồng nội bộ FreshLens là BGR `uint8`. Đổi kênh đúng một lần. EXIF có thể lưu hướng xoay của ảnh điện thoại; `ImageOps.exif_transpose` được gọi trước khi tạo NumPy array.

## 3. CLAHE và Gaussian làm gì?

CLAHE điều chỉnh tương phản cục bộ trên kênh sáng Lab với `clipLimit=2`, lưới `8×8`. Gaussian 3×3 là tích chập có trọng số để giảm nhiễu nhưng có thể làm mất đốm nhỏ. Code có ablation để đo bật/tắt, không mặc định bước nào tốt chỉ vì tên thuật toán.

## 4. Canny có tìm được vùng hỏng không?

Không. Canny tìm biên do biến thiên độ sáng; biên có thể đến từ quả, nền, bóng hoặc chữ. Giao diện gọi đúng là ảnh biên, không gọi là mặt nạ vùng hỏng.

## 5. Bốn nhóm đặc trưng khác nhau thế nào?

Màu mô tả phân bố HSV/Lab nhưng mất vị trí; LBP mô tả quan hệ sáng tối lân cận; hình dạng dùng hình học và Hu trên một foreground ước lượng; HOG tổng hợp hướng gradient theo ô và vị trí. Số chiều lớn không tự chứng minh nhóm đó quan trọng — phải xem ablation trên cùng split.

## 6. Vì sao scaler phải nằm trong Pipeline?

StandardScaler phải học trung bình/độ lệch chuẩn từ phần train của từng fold. Nếu fit trên toàn bộ dữ liệu, thông tin validation/test lọt vào train và làm kết quả lạc quan giả.

## 7. Vì sao ảnh số có thể bị nhận là chuối với xác suất cao?

SVM với các lớp hữu hạn vẫn phải chọn một lớp nếu không có cơ chế từ chối. FreshLens thêm lớp `other`, dùng ngưỡng xác suất và khoảng cách hai lớp đầu. Đây là giảm rủi ro có đo lường, không phải cam kết từ chối mọi ảnh lạ.

## 8. Accuracy khác confidence thế nào?

Confidence là đầu ra trên một ảnh; accuracy là tỷ lệ đúng trên cả tập có nhãn. Phải báo thêm macro-F1, coverage trong phạm vi và false acceptance rate của ảnh ngoài phạm vi.

## 9. Vì sao không dự đoán tình trạng cho ảnh bị từ chối?

Nếu loại quả chưa được chấp nhận thì nhãn fresh/rotten không có ngữ cảnh đáng tin. Nghiệp vụ của app chỉ chạy bộ phân loại tình trạng sau trạng thái `ACCEPTED`.

## 10. Dataset ít có vấn đề gì?

Nhiều ảnh cùng một quả hoặc cùng nền làm test quá dễ do rò rỉ. Cần group theo vật thể/buổi chụp, nguồn ảnh độc lập và ảnh ngoài phạm vi đa dạng. Nếu chưa đo một tình huống, nhóm phải nói rõ chưa đo.

