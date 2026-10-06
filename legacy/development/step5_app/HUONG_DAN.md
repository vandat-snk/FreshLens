# FreshLens - Bước 5: app thực tế + từ chối ảnh ngoài phạm vi

## Mục tiêu

Giữ nguyên CNN `models/cnn_efficientnet_b0/best.pt` đã train. Bước này **không train lại CNN**.

Luồng cuối:

1. Upload ảnh hoặc chụp bằng camera.
2. Open-set gate kiểm tra ảnh có đủ giống Táo / Chuối / Cam / Cà chua hay không.
3. Nếu không: hiện `Loại quả này hiện chưa được FreshLens hỗ trợ`.
4. Nếu có: CNN trả loại quả và Tươi / Hỏng.

Gate dùng embedding EfficientNet + prototype của tập train, sau đó hiệu chỉnh bằng validation đã khóa và thư mục `FreshLens_external_test_v2/unknown`. Vì thế đây là cơ chế open-set thực dụng, không phải cam kết toán học 100% cho mọi ảnh lạ.

## Chạy

Đặt thư mục `FreshLens_Buoc5_AppThucTe` ngay trong project:

`E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens\FreshLens_Buoc5_AppThucTe`

### Lần đầu

Nhấp đúp:

`SETUP_STEP5.cmd`

Nó cài Streamlit nếu thiếu và tạo:

- `models/cnn_efficientnet_b0/open_set_gate.npz`
- `models/cnn_efficientnet_b0/open_set_gate.json`

Việc tạo gate có thể mất vài phút vì phải chạy ảnh train/validation qua EfficientNet một lần. `best.pt` không bị thay đổi.

### Các lần sau

Nhấp đúp:

`RUN_APP.cmd`

Trình duyệt sẽ mở Streamlit. Chọn **Tải ảnh** hoặc **Chụp bằng camera**, rồi bấm **Phân tích ảnh**.

## Nếu setup lỗi

Mở CMD trong project và chạy:

```cmd
.\.venv\Scripts\python.exe .\FreshLens_Buoc5_AppThucTe\BUILD_OPENSET_GATE.py --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" --data "data\cnn_dataset_v3" --checkpoint "models\cnn_efficientnet_b0\best.pt" --unknown "E:\VanDat_\XuLyAnh\FreshLens_external_test_v2\unknown" --device cuda --batch 8
```

Gửi toàn bộ output nếu có `[ERROR]`.
