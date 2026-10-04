**FreshLens — Bước 2B: gom các ảnh nghi liên quan vào cùng nhóm đánh giá**

Gói này tiếp nối kết quả tại `data\cnn_dataset_v2`. Mục tiêu là chốt cách chia
dữ liệu để bắt đầu thử train CNN, xử lý các liên kết nghi cùng nguồn đã phát
hiện giữa train/validation/test. Đây chưa phải bước huấn luyện.

**Kết quả xem ảnh `pairs_01.jpg`**

- Cặp 1–2: các ảnh cà chua có ngoại hình khác nhau. Ảnh mang nhãn fresh có bề
  mặt tương đối nguyên vẹn; ảnh mang nhãn rotten có vết hư hỏng nhìn thấy.
  Ảnh ghép không cung cấp cơ sở để đổi nhãn. Perceptual hash gần nhau không
  đồng nghĩa hai ảnh giống hệt hoặc một nhãn bị sai.
- Cặp 3–8: các cặp táo rất giống nhau, có vẻ là cùng quả hoặc các góc chụp/
  biến thể gần nhau. Giữ chúng cùng một nhóm đánh giá là lựa chọn thận trọng.
- Chỉ 8 cặp trong ảnh ghép đã được xem bằng mắt. 463 cặp còn lại không được
  đánh dấu đã xác nhận trùng; chúng được gom nhóm để hạn chế nguy cơ rò rỉ.

**Script sẽ làm gì?**

1. Kiểm tra manifest, report và CSV các cặp nghi trùng có khớp nhau không.
2. Đọc lại ảnh thật, đối chiếu SHA-256, pixel và tính lại danh sách cặp nghi
   trùng. Nếu kết quả khác lần trước, dừng để kiểm tra nguyên nhân.
3. Giữ nguyên nhóm cũ; nối các nhóm có liên kết trong toàn bộ 471 cặp. Liên
   kết có tính bắc cầu: A liên quan B, B liên quan C thì cả A/B/C ở một nhóm.
   Các thành viên khác thuộc nhóm cũ cũng đi cùng, dù không xuất hiện trong CSV.
4. Giữ đủ 5.869 ảnh cùng nhãn hiện có. Gộp nhóm không có nghĩa xóa ảnh hoặc
   đổi nhãn. Một nhóm có thể chứa cả fresh và rotten nếu chưa biết quan hệ
   vật thể thật; nhóm như vậy được giữ trong train, không dùng để chấm điểm
   validation/test trong thí nghiệm này.
5. Chia các nhóm chỉ có một nhãn theo khoảng 60/20/20, seed 42. Đảm bảo mọi
   cặp đã phát hiện nằm trong cùng nhóm, cùng tập; kiểm tra đủ 8 nhãn ở cả
   ba tập. Ghi thành một phiên bản mới tại `data\cnn_dataset_v3`.

Cách này cố ý thận trọng: hai ảnh khác nhau nhưng giống hình dáng có thể bị
gom cùng nhóm. Đổi lại, các liên kết đã phát hiện không còn băng qua ranh giới
train/validation/test. Đây là quy tắc chia tập, không phải xác nhận mọi cặp
là ảnh trùng. Không được trình bày 4.786 nhóm là 4.786 quả độc lập đã xác minh.

**Chạy bằng CMD**

1. Giải nén, đặt thư mục chứa `STEP2B_GROUP.py` vào project. Đường dẫn đúng:

```text
E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens\FreshLens_Buoc2B_GomNhom\STEP2B_GROUP.py
```

Giữ `step2_core.py` và `VISUAL_REVIEW.json` trong cùng thư mục với script.

2. Chuyển đến project:

```bat
cd /d "E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens"
```

3. Dán nguyên dòng lệnh:

```bat
.\.venv\Scripts\python.exe .\FreshLens_Buoc2B_GomNhom\STEP2B_GROUP.py --input-dir "data\cnn_dataset_v2" --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" --output-dir "data\cnn_dataset_v3"
```

`--root` là thư mục có `raw`, còn `--input-dir` chứa ba file đã được tạo bởi
bước 2: `manifest.csv`, `split_report.json`, `near_duplicates.csv`.
Script dùng Pillow/NumPy của môi trường đã chạy được bước 2. Dòng `[CHECK]`
hiển thị tiến độ đọc ảnh; `[AUDIT]` là kiểm tra các cặp tương tự. Chưa dùng GPU.

**Kết quả dự kiến khi dữ liệu vẫn giống bản đã gửi**

| Chỉ tiêu | Trước | Sau |
| --- | ---: | ---: |
| Số ảnh | 5.869 | 5.869 |
| Số nhóm theo quy tắc hiện tại | 5.101 | 4.786 |
| Các cặp đã phát hiện nằm khác tập | 244 | 0 |
| Ảnh train | 3.529 | 3.519 |
| Ảnh validation | 1.179 | 1.187 |
| Ảnh test | 1.161 | 1.163 |

Các số trên được tính trước từ manifest và danh sách cặp bạn đã gửi; chưa
phải kết quả đọc lại ảnh thật. Lệnh trên sẽ kiểm chứng ảnh trên máy bạn trước
khi xuất kết quả. Cả 8 nhãn đều có trong từng tập ở bản tính trước.

Nhóm cà chua liên quan hai cặp đầu có 4 ảnh sau khi tính cả thành viên của
nhóm cũ. Chúng giữ nguyên nhãn và cùng chuyển vào train. Không tự quy các
ảnh fresh thành rotten hoặc ngược lại.

Sau khi chạy, các dòng chính dự kiến là:

```text
[IMAGES] 5869; labels changed: 0; images removed: 0
[GROUPS] 5101 -> 4786
[CROSS SPLIT PAIRS] 244 -> 0
[READY_FOR_PILOT_TRAINING] Send final_split_report.json for the CNN step.
```

Gửi lại file:

```text
data\cnn_dataset_v3\final_split_report.json
```

Đây là báo cáo cần dùng cho bước CNN tiếp theo. Gói này không khởi chạy
`run_train.bat` cũ hoặc huấn luyện SVM.

**Ý nghĩa các file đầu ra**

| File | Công dụng |
| --- | --- |
| `manifest.csv` | Danh sách ảnh với nhóm và split mới, sẽ dùng cho CNN. |
| `final_split_report.json` | Thống kê, điều kiện bắt đầu train thử và các giới hạn còn lại. |
| `pair_actions.csv` | Cách xử lý từng cặp, tập mới và việc cặp đó đã được xem bằng mắt hay chưa. |
| `group_mapping.csv` | Đối chiếu nhóm/tập cũ với nhóm/tập mới. |
| `dataset_lock.json` | Mã băm manifest/report để kiểm tra đúng phiên bản khi train. |
| `source_split_report.json` | Bản sao nguyên vẹn báo cáo đầu vào để truy vết. |

Trường `decision` trong `pair_actions.csv` là giá trị từ CSV đầu vào; các
trường `action`, `new_group`, `new_split_a`, `new_split_b` ghi hành động mới.
`visually_inspected` chỉ đúng với các cặp khớp quan sát và SHA ảnh đã lưu.

Script chỉ ghi thư mục mới. Không sửa ảnh, CSV/JSON cũ, MongoDB hoặc source
ứng dụng. Thư mục đã tồn tại sẽ không bị ghi đè.

**Nếu gặp thông báo khác**

- `Output directory already exists`: giữ kết quả đã có. Nếu lần trước có
  report, gửi report đó trước. Nếu cần chạy lại sau khi sửa dữ liệu, dùng tên
  thư mục mới như `data\cnn_dataset_v3_retry`.
- `manifest does not match split_report.json`: kiểm tra đang dùng nguyên bộ
  đầu ra bước 2 trong `data\cnn_dataset_v2`; không dùng manifest cũ 15.537 ảnh.
- `current image audit differs` hoặc `SHA-256 mismatch`: gửi nguyên thông báo;
  không sửa số liệu report hoặc CSV bằng tay để bỏ qua kiểm tra.
- `REVIEW_REQUIRED`: kiểm tra `missing_label_split_buckets` trong report.
  Không nhân bản ảnh sang tập khác để lấp một nhãn còn thiếu.

**Phạm vi của kết quả**

`ready_for_pilot_training: true` có nghĩa các điều kiện tự động của bước này
cho phép bắt đầu thử train, không phải chứng nhận hết mọi dạng rò rỉ dữ liệu.
`physical_specimen_independence_confirmed` vẫn là `false` vì chưa có mã quả/
buổi chụp. Hash có thể bỏ sót những góc xoay lớn hoặc crop mạnh của cùng quả.
Việc gom nhóm này cũng không chứng minh mọi nhãn trong dataset đều đúng.

Giữ nguyên manifest này trong quá trình chọn mô hình và tham số. Không chia
lại liên tục để chọn lần có accuracy đẹp. Bắt đầu CNN từ trọng số pretrained;
không lấy mô hình đã học cách chia cũ làm kết quả kiểm chứng cho cách chia mới.

Để đánh giá mục tiêu upload/camera, cần thêm một tập ảnh chụp thực tế riêng,
có nhãn và độc lập với dữ liệu phát triển. Chỉ đánh giá cuối sau khi đã chốt
mô hình. Accuracy hiện chưa được đo ở bất kỳ bước chuẩn bị dữ liệu nào.

**Kiểm thử**

Gói đã qua 16 kiểm thử về nhóm bắc cầu, bảo toàn nhóm cũ, giữ nhãn, kiểm tra
nội dung/nguồn báo cáo, phát hiện danh sách cặp bị thiếu, giữ nguyên file đầu
vào và từ chối ghi đè. Kiểm thử có dùng ảnh tổng hợp và bản nén lại; không thay
thế việc chạy trên ảnh thật trong máy bạn. Runtime kiểm thử là Python 3.12 trên
Linux, chưa chạy trực tiếp Windows/Python 3.13.

Nếu muốn chạy kiểm thử từ thư mục project:

```bat
.\.venv\Scripts\python.exe -m unittest discover -s FreshLens_Buoc2B_GomNhom -p "test_*.py" -v
```
