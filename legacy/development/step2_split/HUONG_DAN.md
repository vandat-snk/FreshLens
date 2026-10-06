**FreshLens — Bước 2: sửa cách chia dữ liệu trước khi train CNN**

Gói này chuẩn bị một phiên bản dữ liệu mới cho thí nghiệm CNN. Chạy bằng CMD
trong môi trường `.venv` hiện tại. Chưa cần cài PyTorch ở bước này.

**Vì sao cần làm bước này?**

Hai file bạn gửi cho thấy có 15.537 bản ghi. Báo cáo cũ có `is_valid: true`,
nghĩa là các kiểm tra file, SHA-256 và `group_id` đã khai báo đều qua.
Tuy nhiên, script kiểm tra trước chưa nhận ra quan hệ ảnh gốc — ảnh biến thể
thể hiện trong tên file.

Phân tích tên file cho thấy 1.452 nhóm ảnh `Screen Shot ...` có các biến thể
ở nhiều tập, liên quan 10.013 bản ghi. Ví dụ nhóm
`Screen Shot 2018-06-08 at 4.59.36 PM` của apple/fresh:

| Tên biến thể | Tập cũ |
| --- | --- |
| `rotated_by_15_...` | val |
| `rotated_by_30_...` | test |
| `rotated_by_45_...` | train |

Đây là bằng chứng từ tên file; chưa phải kết quả đối chiếu pixel các ảnh thật.
Nếu chúng là các biến thể của cùng ảnh gốc như tên mô tả, cách chia này làm
việc đánh giá trên ảnh mới thiếu tin cậy. Không nên dùng accuracy từ cách chia
cũ để kết luận khả năng nhận diện ảnh camera.

**Cách xử lý trong gói**

1. Tạm loại khỏi thí nghiệm 9.310 ảnh có tên biến thể xoay/lật/dịch/nhiễu và
   358 ảnh `Banana__Healthy_augmented_*` chưa biết ảnh cha. Không xóa ảnh gốc
   trên ổ đĩa. Việc tạo biến đổi ngẫu nhiên sẽ được thực hiện chỉ trên tập
   train trong bước CNN.
2. Kiểm tra 5.869 ảnh còn lại trên máy bạn: file có tồn tại, khớp SHA-256,
   giải mã được, và có trùng toàn bộ pixel sau xử lý EXIF/alpha hay không.
   Ảnh không có dấu augmentation trong tên chỉ là ứng viên ảnh gốc; tên file
   không chứng minh đó là một ảnh chụp độc lập.
3. Giữ các liên kết `group_id` cũ; bổ sung liên kết theo tên gốc, tên giống
   nhau khác đuôi/hoa thường và nội dung giống hệt. Chỉ giữ một đại diện cho
   các bản có toàn bộ pixel giống hệt nhau. Nếu cùng nội dung có nhãn mâu
   thuẫn, script dừng để sửa nhãn trước.
4. Chia lại khoảng 60% train / 20% validation / 20% test theo nhóm và nhãn,
   với seed 42. Tỷ lệ tính theo số nhóm, nên số ảnh từng tập có thể lệch nhẹ.
   Mỗi nhóm được giữ nguyên trong một tập. Lần chia này bắt đầu thí nghiệm
   mới; không thể so trực tiếp accuracy với mô hình dùng cách chia cũ.
5. Tìm các cặp ảnh có perceptual hash gần nhau. Đây là danh sách cần xem lại,
   không phải khẳng định chúng trùng. Script không tự gộp các cặp này. Cặp
   khác nhóm nằm ở hai tập khác nhau, cặp khác nhãn, nhóm có nhiều nhãn,
   thiếu nhãn trong một tập hoặc quá nhiều cặp cần rà soát sẽ bật
   `review_required: true`.

Số ứng viên trước kiểm tra pixel:

| Loại quả | Fresh | Rotten |
| --- | ---: | ---: |
| Apple | 921 | 841 |
| Banana | 624 | 814 |
| Orange | 748 | 745 |
| Tomato | 602 | 574 |
| Tổng | 2.895 | 2.974 |

Số ảnh cuối cùng có thể nhỏ hơn 5.869 sau khi bỏ các bản giống hệt pixel.

**Cách chạy từng bước**

1. Giải nén và đặt thư mục chứa `STEP2_SPLIT.py` vào project. Đường dẫn đúng là:

```text
E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens\FreshLens_Buoc2_ChiaTap\STEP2_SPLIT.py
```

2. Mở CMD và chuyển đến project:

```bat
cd /d "E:\VanDat_\XuLyAnh\FreshLens_complete_v2\FreshLens"
```

3. Chạy nguyên dòng lệnh sau:

```bat
.\.venv\Scripts\python.exe .\FreshLens_Buoc2_ChiaTap\STEP2_SPLIT.py --manifest "data\cnn_dataset\manifest.csv" --root "E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset" --output-dir "data\cnn_dataset_v2"
```

Đường dẫn sau `--root` là thư mục có thư mục con `raw`, không phải thư mục
project. File đầu vào sau `--manifest` là manifest đã export từ MongoDB.
Script dùng Pillow và NumPy đã có trong requirements của project, không dùng
GPU và không truy cập MongoDB. Dòng `[CHECK] ...` biểu thị đang đọc/kiểm tra
ảnh, chưa phải đang train.

4. Khi lệnh hoàn tất, gửi lại:

```text
data\cnn_dataset_v2\split_report.json
data\cnn_dataset_v2\near_duplicates.csv
```

Nếu có thư mục `data\cnn_dataset_v2\review_images`, gửi thêm ảnh
`pairs_01.jpg` trong đó. Đây là ảnh ghép các cặp cần xem lại; số Pair trên ảnh
trùng với `pair_id` trong CSV. Tối đa 48 cặp được tạo ảnh xem nhanh, còn toàn
bộ cặp phát hiện được nằm trong CSV.

**Các file được tạo**

| File | Ý nghĩa |
| --- | --- |
| `manifest.csv` | Danh sách ảnh và tập mới, cần đọc kèm trạng thái trong report trước khi train. |
| `split_report.json` | Số lượng, kết quả kiểm tra và trạng thái cần xem lại. |
| `excluded.csv` | Những bản ghi bị tạm loại và lý do; ảnh trên ổ đĩa vẫn còn. |
| `lineage.csv` | Đối chiếu toàn bộ bản ghi trước/sau, gồm `included` và split của nhóm. |
| `image_fingerprints.csv` | SHA file, SHA pixel, perceptual hash và kích thước ảnh đã kiểm tra. |
| `near_duplicates.csv` | Các cặp khác nhóm có vẻ giống nhau, chưa kết luận trùng. |
| `review_images/pairs_*.jpg` | Ảnh ghép giúp xem lại các cặp ưu tiên. |

Chỉ `manifest.csv` là danh sách cho chương trình train. `excluded.csv` và
`lineage.csv` không phải danh sách train. Bản ghi bị loại có thể vẫn có split
của nhóm để truy vết; trường `included` mới xác định nó có nằm trong manifest
mới hay không.

Script không sửa source code ứng dụng, ảnh, manifest cũ hoặc dữ liệu MongoDB.
Nó ghi một thư mục kết quả mới và từ chối ghi đè thư mục đã tồn tại. Sau bước
này, không chạy lại `run_train.bat` cũ để train SVM; bước tiếp theo sẽ cấu hình
CNN dùng manifest mới khi kết quả rà soát cho phép.

**Đọc kết quả và xử lý thông báo**

- `[OK] Saved: ...`: đã tạo bộ file đầu ra, chưa có nghĩa dữ liệu đã hết mọi
  nguy cơ trùng hoặc mô hình đã được train.
- `[REVIEW_REQUIRED]`: có trường hợp cần xem lại. Gửi report, CSV và ảnh ghép
  để xử lý tiếp; không đổi thành `false` bằng tay.
- `[READY_FOR_PILOT_TRAINING]`: các kiểm tra tự động hiện tại cho phép bắt đầu
  thử train. Vẫn cần xác minh nguồn/chùm chụp và một tập ảnh camera riêng để
  đánh giá khả năng tổng quát hóa. `evaluation_independence_confirmed` luôn
  là `false`, vì script không biết mỗi ảnh có cùng quả/cùng buổi chụp hay không.
- `Output directory already exists`: kết quả trước được giữ nguyên. Nếu cần
  chạy lại sau khi sửa dữ liệu, dùng một tên mới, ví dụ
  `--output-dir "data\cnn_dataset_v2_retry"`. Khi đã bắt đầu đánh giá mô hình,
  giữ nguyên split; không đổi seed liên tục để tìm accuracy cao hơn.
- `File changed since manifest export`: nội dung ảnh đã khác SHA-256 trong
  manifest. Gửi nguyên thông báo để kiểm tra trước khi export lại.
- `Filename and metadata labels disagree` hoặc `Same content has conflicting
  labels`: cần rà soát nhãn. Script không tự đoán nhãn đúng.
- `Image file not found`: kiểm tra `--root`; ví dụ ảnh `raw/apple/fresh/x.png`
  phải có tại `<root>\raw\apple\fresh\x.png`.
- `Missing dependency`: chỉ nếu gặp dòng này, cài các thư viện cần thiết bằng:

```bat
.\.venv\Scripts\python.exe -m pip install "Pillow>=10,<12" "numpy>=1.26,<3"
```

**Giới hạn cần hiểu để báo cáo đồ án đúng**

Perceptual hash chỉ là bộ lọc để rà soát: có thể đánh dấu nhầm hai ảnh khác
nhau, cũng có thể bỏ sót ảnh crop/xoay hoặc ảnh cùng vật thể. Ngưỡng đang dùng
là khoảng cách Hamming tối đa 4 trên 63 bit DCT bỏ thành phần DC. Nếu vượt
20.000 cặp, báo cáo ghi audit chưa đầy đủ và yêu cầu xem lại.

Tên số như `freshApple (1)` không cho biết nguồn dataset hay vật thể thật.
Nếu còn dữ liệu về ảnh gốc, nguồn tải, cùng quả hoặc cùng buổi chụp, giữ lại
để bổ sung nhóm. Số ảnh nhiều không đồng nghĩa số mẫu độc lập nhiều.

Hiện bộ dữ liệu chỉ có 4 loại quả và 2 trạng thái, chưa có dữ liệu nền/ngoài
phạm vi để học từ chối các ảnh khác. Chưa thể hứa nhận diện đúng ảnh bất kỳ
hoặc biết trước accuracy. Tập ảnh camera thực tế cần tách riêng, có nhãn,
không dùng để chỉnh mô hình sau khi xem kết quả đánh giá cuối.

Tài liệu gốc:

- [scikit-learn: data leakage và nguyên tắc tách train/test](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [PyTorch: transfer learning, augmentation cho train và đánh giá riêng](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

**Kiểm thử gói**

Gói có các kiểm thử bằng ảnh tạo cho mục đích kiểm thử: giữ nhóm, loại bản
augmentation, bảo toàn file đầu vào, trùng pixel dù khác byte file, nhãn mâu
thuẫn, cặp ảnh nén lại, thứ tự CSV thay đổi, giới hạn rà soát và từ chối ghi
đè. Chúng không thay thế kiểm tra ảnh thật trên máy bạn và không đo accuracy.

Nếu muốn tự chạy các kiểm thử, dùng lệnh sau từ project:

```bat
.\.venv\Scripts\python.exe -m unittest discover -s FreshLens_Buoc2_ChiaTap -p "test_*.py" -v
```

`PHAN_TICH_MANIFEST_CU.json` là kết quả phân tích tên từ hai file bạn đã gửi;
không phải kết quả kiểm tra pixel hay báo cáo huấn luyện.
