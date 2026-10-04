# Nhật ký thí nghiệm FreshLens

Mỗi dòng phải giữ nguyên `run_id`, manifest, seed, feature config, preprocessing, kernel, C/gamma, threshold và phiên bản thư viện.

| Run ID | E | Feature config | Kernel | Split/group | Val macro-F1 | Coverage | False acceptance other | Test macro-F1 | Kết luận |
|---|---|---|---|---|---:|---:|---:|---:|---|
|  | E0/E1/E2/E3/E4 |  |  |  |  |  |  |  |  |

## Bảng lỗi bắt buộc

Lưu `reports/latest_predictions.csv`, sau đó chọn mẫu theo:

* chuối ↔ cam;
* táo ↔ cà chua;
* nền phức tạp, sáng/tối, crop sát;
* số/chữ, màn hình, đồ vật, thực phẩm khác;
* ảnh có nhiều loại trong cùng ảnh;
* ảnh đúng bị từ chối và ảnh ngoài phạm vi bị nhận nhầm.

Không sửa nhãn hoặc đổi ngưỡng chỉ để làm đẹp một ca demo. Nếu dùng lỗi test để chỉnh tiếp, bộ đó trở thành dữ liệu phát triển và phải có test độc lập mới.

