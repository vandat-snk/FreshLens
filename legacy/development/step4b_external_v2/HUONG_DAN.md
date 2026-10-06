# FreshLens – Bước 4B: Mở rộng External Test

Mục tiêu: tăng bộ external từ 5 ảnh/lớp lên **ít nhất 10 ảnh/lớp** và tăng `unknown` lên **ít nhất 20 ảnh**, trong khi vẫn giữ nguyên External Test V1 làm bằng chứng độc lập.

## Nguyên tắc quan trọng

- **Không sửa/xóa** `E:\VanDat_\XuLyAnh\FreshLens_external_test` (V1).
- Không đưa các ảnh external V1/V2 vào train trước khi chốt kết quả V2.
- Không copy ảnh từ dataset nội bộ.
- Ảnh phải là ảnh mới thật sự; ưu tiên ảnh tự chụp hoặc ảnh Internet khác nguồn dataset.
- Ảnh mơ hồ về fresh/rotten không ép nhãn: cho vào `ambiguous_review` và không tính accuracy.

## Chuẩn nhãn sử dụng ở Bước 4B

**Fresh:** không thấy mốc, mục, mô mềm nhũn/sụp, vùng thối ướt rõ. Vết sẹo nhỏ, xước vỏ, khác màu tự nhiên nhưng không có dấu hiệu phân hủy vẫn có thể là fresh.

**Rotten:** có dấu hiệu phân hủy nhìn thấy rõ như mốc, vùng thâm/mục lan rộng, mô mềm sụp/ướt, vết thối rõ.

Nếu không chắc: `ambiguous_review`.

## 1. Cài gói

Giải nén thư mục này vào project sao cho có:

`E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens\FreshLens_Buoc4B_MoRongExternalTest\RUN_V2.cmd`

## 2. Tạo V2 mà không phá V1

Nhấp đúp `PREPARE_V2.cmd`.

Nó copy V1 sang:

`E:\VanDat_\XuLyAnh\FreshLens_external_test_v2`

## 3. Bổ sung ảnh

Mỗi class known cần **>=10 ảnh tổng cộng**, nghĩa là nếu V1 đã có 5 thì thêm ít nhất 5 ảnh mới:

- apple_fresh
- apple_rotten
- banana_fresh
- banana_rotten
- orange_fresh
- orange_rotten
- tomato_fresh
- tomato_rotten

`unknown` cần **>=20 ảnh tổng cộng**. Nên chia cân bằng:

- ~10 ảnh **quả khác**: grape, mango, pear, watermelon, dragon fruit, guava, pineapple...
- ~10 ảnh **không phải quả**: chai nước, cốc, bàn tay, hộp, điện thoại, cây lá, đồ vật...

### Mỗi class known nên có độ đa dạng

Trong 5 ảnh mới, cố gắng có:

1. 1 ảnh nền đơn giản.
2. 1 ảnh nền phức tạp.
3. 1 ảnh chụp gần.
4. 1 ảnh chụp xa hoặc góc nghiêng.
5. 1 ảnh ánh sáng khác bình thường / tự chụp bằng camera điện thoại.

Không cố tình chọn toàn ảnh “đẹp” hoặc toàn ảnh “khó”.

## 4. Chạy

Sau khi đủ ảnh, nhấp đúp `RUN_V2.cmd`.

Script sẽ:

1. kiểm tra số ảnh;
2. kiểm tra ảnh trùng byte bên trong V2;
3. chỉ khi đạt mới chạy `EXTERNAL_TEST.py`;
4. lưu kết quả mới vào `reports\external_test_v2`.

## 5. Gửi lại 3 file

- `reports\external_test_v2\metrics.json`
- `reports\external_test_v2\predictions.csv`
- `reports\external_test_v2\confusion_matrix.png`

Sau đó sẽ so sánh V1 vs V2 và quyết định có cần tạo CNN V2 với lớp `other/unknown` hay chỉ cần cải thiện dữ liệu fresh/rotten.
