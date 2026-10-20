# Phân công FreshLens CNN

Căn cứ: tài liệu phân công nâng cấp bốn thành viên và bộ slide 20 phút.

| Thành viên | Vai trò | Module chính | Slide |
| --- | --- | --- | --- |
| Lương Thị Như Huỳnh — N23DCCN096 | TV1 dữ liệu và xử lý ảnh | freshlens_ai/data, inference/quality.py, PREPARE_CAMERA_MANIFEST.py | 01–07 |
| Võ Văn Đạt — N23DCCN078 | TV2 huấn luyện CNN | models, training, TRAIN_CNN_V2.py, BENCHMARK_MODELS.py | 08–13 |
| Hứa Như Quỳnh — N23DCCN118 | TV3 suy luận và đánh giá | inference, evaluation, EVALUATE_TV3.py, Grad-CAM | 14–20 |
| Nguyễn Thị Yến Nhi — N23DCCN112 | TV4 giao diện và tích hợp | ui, APP_CNN_V2.py, launcher, cài đặt, QA và tài liệu | 21–26 |

Slide 27–29 là phụ lục. SVM trong legacy/svm chỉ là lịch sử/baseline.
Phân công là trách nhiệm kiểm tra và bàn giao, không tự chứng nhận thí nghiệm đã hoàn thành.

## Bàn giao

- Huỳnh: nhãn ảnh camera, group theo quả vật lý, nguồn ảnh, báo cáo trùng, ảnh trước/sau tiền xử lý; tách dev/calibration/final.
- Đạt: cấu hình, seed, checkpoint/hash, learning curves; chỉ công bố nâng cấp tốt hơn sau khi so cùng protocol. Candidate multi-task và resolution mới là thí nghiệm, chưa là production.
- Quỳnh: JSON/CSV/CM, error gallery, coverage/FAR, quality và latency. External V2 đã calibration không phải final test độc lập.
- Nhi: bốn trang chạy được, đổi ảnh xóa kết quả và heatmap, ảnh lỗi không crash, CPU fallback, cài mới, demo upload và camera thật.

Mỗi người kiểm tra công thức và giáo trình thuộc phần mình. Nhi tổng hợp báo cáo;
cả nhóm xác nhận bằng chứng. Xem docs/SUBMISSION.md để chạy lại công cụ.
