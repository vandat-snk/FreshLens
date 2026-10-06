# Phân công bốn thành viên FreshLens

## TV1 — Dữ liệu và tiền xử lý

Phụ trách `legacy/svm/src/dataset.py`, `legacy/svm/src/check_dataset.py`, `legacy/svm/src/image_processing.py` và phần đường dẫn trong `legacy/svm/src/config.py`.

Bàn giao: manifest có `path, fruit, status, source, group_id, sha256, split`; thống kê 8 tổ hợp loại × tình trạng; ảnh `other`; bảng lỗi/ảnh loại; bằng chứng ảnh xám, PNG alpha, EXIF, đường dẫn Unicode; so sánh bật/tắt CLAHE/Gaussian.

## TV2 — Đặc trưng và thí nghiệm xử lý ảnh

Phụ trách `legacy/svm/src/features.py`.

Bàn giao: giải thích vector màu 132, texture 21, shape 12, HOG 6.084; thứ tự vector cố định; minh họa histogram, LBP, Canny, gradient; bảng ablation cùng một manifest và mã lần chạy. Không gọi Canny/HOG/LBP là vùng hỏng hoặc vùng mô hình chú ý.

## TV3 — Huấn luyện và quyết định nhận dạng

Phụ trách `legacy/svm/src/model.py`, `legacy/svm/src/train.py`, `legacy/svm/src/predict.py` và requirements.

Bàn giao: baseline/candidate; scaler trong pipeline; SVM linear/RBF; chọn tham số trên fold giữ nhóm; lớp `other`; ngưỡng xác suất + margin chọn trên validation; artifact lưu model, classes, feature order, preprocessing, threshold, run_id, version; không suy diễn status khi fruit bị từ chối.

## TV4 — Giao diện, nghiệp vụ và đánh giá độc lập

Phụ trách `legacy/svm/src/app.py`, `legacy/svm/src/metrics.py`, `legacy/svm/tests/` và tổng hợp demo.

Bàn giao: upload/camera; trạng thái thiếu model/chưa có ảnh/đang xử lý/chấp nhận/chưa đủ tin cậy/ngoài phạm vi/tệp lỗi; JSON khớp kết quả; confusion matrix, precision/recall/F1, macro-F1, coverage, accuracy trên phần được nhận, false acceptance rate; test ảnh thuận lợi, ảnh khó, ảnh ngoài phạm vi và ảnh nhiều vật thể.

## Kiểm tra chéo

* Vòng 1: TV1 ↔ TV2; TV3 ↔ TV4.
* Vòng 2: TV1 ↔ TV3; TV2 ↔ TV4.
* Mỗi người phải chạy được toàn bộ pipeline và giải thích một ca đúng, một ca sai, một ca bị từ chối.
* Nếu chưa đo một trường hợp, nói rõ chưa đo và nêu thí nghiệm sẽ dùng; không lấy ảnh hình học hoặc demo tổng hợp để tuyên bố accuracy ảnh đời thực.

