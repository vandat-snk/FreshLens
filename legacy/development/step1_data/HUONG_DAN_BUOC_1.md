FreshLens — Bước 1: sửa và kiểm tra luồng dữ liệu

Gói này áp dụng cho bản FreshLens_complete_v2.rar đã được xem xét. Mục tiêu là bảo toàn metadata và ngăn dữ liệu không hợp lệ đi vào train. Đây chưa phải bản CNN và chưa cung cấp model được huấn luyện trên dataset thật của bạn.

Lộ trình tiếp theo: kiểm tra dataset thật → thêm CNN và sửa đánh giá validation → huấn luyện/đánh giá → nối vào upload/camera.

1. Mở đúng thư mục dự án.

   Trong PyCharm, mở thư mục FreshLens có `src`, `scripts`, `requirements.txt` và file `.env` hiện có. Mở Terminal ở thư mục này. Các lệnh dưới đây dùng PowerShell trên Windows; không cần kích hoạt virtualenv.

   Để xác nhận đúng vị trí, chạy:

   ```powershell
   Test-Path .\src\train.py
   Test-Path .\.venv\Scripts\python.exe
   ```

   Cả hai nên trả về `True`. Nếu dòng thứ nhất là `False`, chuyển vào đúng thư mục FreshLens. Nếu chỉ dòng thứ hai là `False`, dùng interpreter đã chọn trong PyCharm; hoặc tạo môi trường Python 3.12 ở đúng thư mục bằng `py -3.12 -m venv .venv`. Trường hợp vừa tạo môi trường mới, cài thêm `requirements.txt` trước khi dùng toàn bộ ứng dụng.

2. Đặt gói cập nhật vào dự án.

   Giải nén ZIP, lấy thư mục `FreshLens_Buoc1_DuLieu` đặt cạnh `src` và `scripts`. Khi đó file công cụ phải nằm tại `FreshLens\FreshLens_Buoc1_DuLieu\APPLY_STEP1.py`. Tránh lồng thêm hai thư mục cùng tên do tùy chọn giải nén của Windows.

   Chạy:

   ```powershell
   .\.venv\Scripts\python.exe .\FreshLens_Buoc1_DuLieu\APPLY_STEP1.py --project .
   ```

   Kết quả đúng có dòng `[OK] Da cap nhat ... file` và `[BACKUP] ...`. Công cụ sao lưu những file cũ vào `backups\step1_...` trước khi thay. Chạy lại gói đã áp dụng sẽ báo đã cập nhật đầy đủ.

   Nếu hiện `[DUNG]` cùng danh sách file khác bản RAR, chưa có file nào bị thay. Gửi lại thông báo đó để ghép đúng phiên bản code. Đừng tự chép đè thư mục `payload` trong trường hợp này.

3. Cài thư viện cho dữ liệu.

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements-db.txt
   ```

   File này cài PyMongo, dnspython, python-dotenv, Cloudinary và Pillow. Luồng thống kê hiện tại không cần cài PyTorch/CUDA. Nếu vừa tạo virtualenv mới và muốn chạy bộ kiểm thử toàn dự án, cài thêm `requirements.txt`.

4. Đọc thống kê MongoDB.

   Dùng cấu hình MongoDB hiện có trong `.env` tại thư mục FreshLens. Hai tên cấu hình mà code đọc là `MONGODB_URI` và `MONGODB_DB`; collection hiện tại là `images`.

   ```powershell
   .\.venv\Scripts\python.exe scripts\export_manifest_from_db.py --stats --report reports\step1_mongodb_summary.json
   ```

   Lệnh này chỉ đọc metadata MongoDB và tạo một báo cáo JSON trên máy. Nó không tạo index, chia tập, sửa bản ghi hay tải ảnh. Khi thành công, có dòng `[OK] Da luu bao cao: reports\step1_mongodb_summary.json`.

   Gửi `reports\step1_mongodb_summary.json` để quyết định cách chuẩn bị ảnh thật cho bước kế tiếp. Nếu lệnh lỗi trước khi tạo báo cáo, gửi ảnh thông báo lỗi.

Các trường trong báo cáo có ý nghĩa như sau:

| Trường | Ý nghĩa |
|---|---|
| `total_images` | Số bản ghi ảnh đọc từ collection |
| `split_counts` | Số bản ghi train/val/test/unsplit |
| `label_counts` | Số ảnh theo tổ hợp loại quả và tình trạng |
| `unique_groups` | Số group_id đã khai báo, chưa xác minh mỗi nhóm thực sự là một quả/buổi chụp |
| `with_cloud_url` | Số bản ghi có URL; chưa xác minh URL còn hoạt động |
| `with_local_path_reference` | Số bản ghi có đường dẫn; chưa xác minh file có trên máy |
| `metadata_error_count` | Số lỗi cấu trúc, nhãn, group/split hoặc SHA đã khai báo |
| `metadata_error_counts` | Phân loại những lỗi trên |

`metadata_error_count = 0` không có nghĩa accuracy đã tốt. Còn phải mở ảnh, kiểm tra ảnh gần trùng, nguồn dữ liệu và tập ảnh chụp thật độc lập. Các dòng `unsplit` được thống kê để biết trạng thái hiện tại; lệnh `--stats` không tự sửa chúng.

Các thay đổi trong gói:

| File | Thay đổi và lý do |
|---|---|
| `scripts/db_manager.py` | Upload lại không ghi đè split, nhóm/nguồn đã có hay URL bằng giá trị rỗng; chặn cùng nội dung nhưng nhãn mâu thuẫn; chỉ tạo index khi thao tác upload yêu cầu; không in URI kết nối |
| `scripts/upload_dataset_to_db.py` | Dùng chung bộ quét local để đọc groups.csv và SHA; kiểm tra metadata; báo lỗi nếu upload chưa hoàn tất |
| `src/data_integrity.py` | Kiểm tra nhãn/nhóm/split và tổ chức nhóm ảnh trùng byte; chia lần đầu theo nhóm 60/20/20; giữ các tập đã có |
| `scripts/export_manifest_from_db.py` | Thống kê chỉ đọc; giữ split trong DB/manifest cũ; dừng khi ảnh tải lỗi; lưu manifest sau kiểm tra; tải qua file tạm có timeout; tránh trùng tên ảnh từ URL |
| `src/check_dataset.py` | Manifest rỗng/split không hợp lệ là lỗi; tính lại SHA từ file thật; chặn ảnh trùng byte giữa các tập; không âm thầm tạo lại manifest bị thiếu |
| `src/train.py` | Kiểm tra dataset trước khi fit; lưu lỗi vào reports/dataset_check_before_training.json; phần thuật toán/validation SVM sẽ được xử lý ở bước sau |
| `src/config.py` | Đọc .env nhất quán cho luồng local; biến môi trường đã khai báo vẫn được ưu tiên |
| `run_train.bat` | Dùng Python của .venv và thư mục dự án; không tạo lại manifest từ thư mục raw |
| `requirements.txt`, `requirements-db.txt` | Tách rõ thư viện dữ liệu, bỏ điều kiện extra khiến pip bỏ qua gói cần thiết |
| `tests/test_step1_data.py` | 27 kiểm tra hồi quy cho đồng bộ dữ liệu, chia nhóm, tải ảnh, bảo toàn manifest và chặn train lỗi |

Chính sách chia tập của bản sửa:

- Lần đầu chưa có split, dùng cách chia 60/20/20 theo group đã có trong dự án. Nếu một lớp quá ít nhóm, không nhân bản ảnh để lấp tập trống.
- Khi đã có split, giữ nguyên phân công. Ảnh mới thuộc một nhóm cũ kế thừa split của nhóm. Nhóm mới hoàn toàn được thêm vào train, để bộ validation/test dùng so sánh được giữ cố định.
- Các nhóm có ảnh giống hệt về SHA được giữ trong cùng tập. Nếu metadata cũ đã đặt chúng ở nhiều tập, exporter dừng để rà soát, không âm thầm chuyển ảnh.
- Khi export lần sau, dùng lại split trong manifest đầu ra đang có. Giữ manifest này cùng checkpoint để tái lập thí nghiệm; không xóa rồi tạo lại mỗi lần train.
- Export mặc định không ghi DB. Chỉ `--write-splits` mới ghi trường split vào MongoDB sau khi dữ liệu local đã được kiểm tra. `--force-resplit` tạo một bộ chia mới và làm mất tính so sánh trực tiếp với kết quả cũ; không dùng trong bước thống kê hiện tại.
- SHA chỉ tìm ảnh trùng byte. Ảnh xoay/crop/nén khác nhau, hoặc nhiều góc chụp của cùng một quả, vẫn cần group chính xác và kiểm tra ảnh gần trùng ở bước dữ liệu tiếp theo.
- Upload lại bảo toàn group/source đã có trong DB. Nếu các trường cũ vốn sai, cần rà soát và sửa có chủ đích; gói này không tự suy luận nhóm thực tế từ tên ảnh.

Phần kiểm tra đã thực hiện trước khi gửi:

- 27 kiểm tra hồi quy chạy với MongoDB/Cloudinary/network giả lập, không kết nối database thật.
- 6 hàm kiểm tra sẵn có của dự án chạy qua.
- Kiểm tra cú pháp 22 file Python trong src/scripts/tests và kiểm tra CLI `--help`.
- Phần xử lý ảnh chạy bằng nhánh Pillow/NumPy vì môi trường kiểm tra không có OpenCV. Chưa chạy Streamlit trên Windows hoặc huấn luyện với dữ liệu thật.

Để tự chạy các kiểm tra hồi quy sau khi cài thư viện dự án:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_step1_data.py -v
```

Kết thúc bước hiện tại ở báo cáo MongoDB. Bước tiếp theo phụ thuộc số lớp, số ảnh tươi/hỏng, tình trạng split và vị trí ảnh được báo cáo; sau đó mới chốt dataset cho CNN.
