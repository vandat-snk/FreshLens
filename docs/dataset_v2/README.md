# Dataset V2 — TV1 data contract và cách tái tạo

Tên deliverable là **Dataset V2 (TV1)**. Thư mục output dùng **`cnn_dataset_v4`**, vì `cnn_dataset_v3` đã là baseline cố định và `cnn_dataset_v2` xuất hiện trong lịch sử step 2. Bản đầu hiện tại chỉ là metadata snapshot: chưa có raw images hoặc ảnh bổ sung. Không dùng snapshot này để tuyên bố đã đạt nghiệm thu dữ liệu.

## Công cụ dùng lại

- Production `freshlens_ai/data/image_io.py`: decode một frame, EXIF transpose, RGB, alpha compositing trên trắng.
- Production `freshlens_ai/data/locked_dataset.py`: xác minh baseline manifest/report/lock, tám nhãn và leakage theo metadata.
- `freshlens_ai/utils/file_io.py`: SHA256 và ghi JSON/CSV nguyên tử.
- Legacy `step2b_grouping/step2_core.py`: `UnionFind` và `assign_splits`; module được đọc từ vị trí tương đối trong repository. Không sửa hoặc chạy luồng legacy loại ảnh/resplit trên V3.
- Legacy step1/check_dataset dùng schema SVM gồm `other` và preprocessing khác CNN, nên không gọi như quality gate production.

`freshlens_ai.data` trì hoãn import Dataset/transforms cho tới lúc được yêu cầu. API import hiện tại được giữ; metadata audit cần Pillow/NumPy, không cần PyTorch.

## Cấu trúc output

| File | Vai trò |
|---|---|
| `manifest.csv` | Chỉ tám supported combinations; bảy cột production giữ nguyên, thêm capture và quality metadata |
| `other_manifest.csv` | `category`: `other_fruit`, `non_fruit`, `multiple`; `subcategory` lưu loại quả/vật thể/cảnh |
| `image_quality.csv` | Hash thực đo, decode, resolution, brightness, blur, format, mode, channels, EXIF, size |
| `dataset_report.json` | Audit, lineage hash của ba file V3, split policy, trạng thái thiếu dữ liệu |
| `dataset_lock.json` | SHA256 của toàn bộ output metadata; **không phải** production V3 lock |

Supported manifest vẫn có `path,fruit,status,source,group_id,sha256,split`. Tên validation trong code là **`val`**. Không đưa `other` vào `CLASSES`, training label index hoặc prediction contract.

**Loader production chưa chấp nhận lock mới.** Không tạo `final_split_report.json` giả định đã kiểm chứng perceptual candidates. TV2/TV3 cần thống nhất loader candidate riêng/bước tích hợp sau khi đủ ảnh, review grouping và khóa protocol. Không sửa loader production để bỏ các gate hiện tại.

## Thu thập ảnh thật

1. Khôi phục raw baseline vào một image root chứa `raw/...` đúng các path manifest V3. Giữ byte gốc để SHA khớp. Không lưu lại qua trình chỉnh ảnh hoặc tự sửa hash baseline.
2. Bổ sung ảnh mới vào cùng root, ví dụ `camera/session_01/...` và `unknown/source_01/...`. Đường dẫn CSV tương đối với root; không lưu đường dẫn ổ đĩa cá nhân vào CSV/code.
3. Copy `capture_inventory.csv` ra file inventory local và điền **một hàng cho mỗi ảnh thật**. File mẫu chỉ có header; không có dữ liệu giả. Supported: `data_kind=supported`, fruit/status hợp lệ, category/subcategory để trống. Unknown: `data_kind=other`, fruit/status để trống, category/subcategory bắt buộc.
4. `source` mô tả nguồn cụ thể, không chỉ `local`; `provenance` lưu dataset ID/URL/capture log; `license` ghi quyền dùng hoặc `not_verified`. Không thêm ảnh không rõ quyền sử dụng vào bản public.
5. Với camera, điền `capture_device=phone|laptop_camera|other_camera`, `specimen_id`, `specimen_evidence`, `session_id`, `lighting`, `background`, `viewpoint`, `distance`. `specimen_id` duy nhất toàn bộ inventory, dùng cùng ID khi cùng quả thay góc, nền, buổi chụp hoặc chuyển fresh sang rotten. `group_id` phải giống nhau cho các ảnh ấy; `specimen_evidence` dẫn tới nhật ký thu thập thật. Không tạo ID riêng cho từng file rồi tuyên bố độc lập.
6. Multiple objects: group theo cùng cảnh/collection specimen; ghi rõ các vật thể liên quan trong evidence. Nếu cảnh chia sẻ quả với supported ảnh khác thì dùng cùng group hoặc rà soát liên kết trước split.
7. `hard_case_tags` dùng dấu `;`, ví dụ `yellow_light;shadow;hand_held;small_damage`. Không suy ra các tag này từ tên file/brightness một cách tự động.
8. `split` có thể để trống cho group mới. Nếu đã có protocol/frozen split, khai báo rõ; mọi hàng cùng group/SHA phải nhất quán. Không thêm ảnh để tối ưu một final test đã được xem.

### Ma trận thu thập camera

Thu đủ apple/banana/orange/tomato × fresh/rotten bằng nhiều **quả vật lý thật**. Ưu tiên apple fresh/rotten, orange fresh, tomato rotten. Có điện thoại và camera laptop, ánh sáng vàng/yếu, bóng đổ, nền phức tạp, tay cầm, góc nghiêng, gần/xa, quả nhỏ trong frame, hỏng nhẹ/vết hỏng nhỏ. Mục tiêu development ban đầu 20–40 ảnh khó là kế hoạch thu thập, **chưa phải số đã có**. Cần đủ group mới để chia; không nhân bản quả/ảnh để lấp bucket.

### Ma trận thu thập unknown

Other fruit: grape, mango, pear, watermelon, dragon fruit, lemon, strawberry và loại khác. Non-fruit: phone, laptop, keyboard, cup, book, shoe, bag và vật thể khác. Multiple: multiple fruits, fruit+object, multiple objects. Thu qua nhiều nguồn, nền, ánh sáng, góc và khoảng cách. Kiểm tra tỷ trọng từng `subcategory`; không coi tập chủ yếu một loại quả là đủ đa dạng. Tool luôn ghi `OTHER DATA INCOMPLETE` cho tới khi nhóm nghiệm thu tính đa dạng bằng protocol có bằng chứng.

Ảnh/nhãn mơ hồ cần review chung: lưu hàng inventory trong danh sách local chờ review; không tự ép fresh/rotten. Nhóm phải thống nhất biểu hiện nhìn thấy, không suy ra an toàn ăn uống từ ảnh.

## Split và duplicate policy

- Giữ toàn bộ bảy field baseline, mọi hàng và split V3. Không tự bỏ duplicate.
- Union tất cả liên kết `group_id` và SHA **xuyên supported + other** trước chia tập. Group mới trùng byte với baseline kế thừa split baseline.
- Component liên kết nhiều split đã khóa: dừng và báo review, không chuyển hàng.
- Component mới thuần nhãn: reuse 60/20/20 theo **group** và seed 42. Component nhãn hỗn hợp mới vào train. Giữ nguyên group ID đã khai báo; coassignment SHA không chứng minh cùng vật thể.
- Với ít group, val/test có thể trống; tool không tạo thêm dữ liệu. Kiểm tra report trước bàn giao.
- SHA chỉ phát hiện trùng byte. Crop/rotate/recompress hoặc góc khác cùng quả cần evidence và perceptual/human review. Legacy `phash63`, `HashTree`, `find_candidates` có thể dùng cho review riêng trên ảnh thật; chưa chạy trên dataset này vì không có raw.
- Chưa xác minh physical specimen independence. `FINAL TEST NOT YET FROZEN`: internal test V3 đã dùng đánh giá và External V2 đã dùng calibration; không gọi chúng là final camera test mới độc lập.

## Quality audit

Đo trên RGB đã EXIF-correct ở kích thước gốc, trước letterbox/augmentation. Brightness là trung bình Pillow luminance `L` thang 0–255. Blur score là variance của Laplacian bốn lân cận, boundary reflect, không resize. Candidate thresholds: cạnh nhỏ nhất <96 px, brightness <35, blur score <50. Đây là ứng viên diagnostic chưa được hiệu chỉnh trên dữ liệu thật; texture/resolution/background ảnh hưởng blur score. Không đưa vào inference, không loại ảnh tự động.

Ảnh decode thành công: `OK`/`LOW_QUALITY`; decode lỗi: `DECODE_ERROR`. Không có raw: `quality_status=NOT_AVAILABLE`, decode `NOT_AVAILABLE` hoặc `MISSING_FILE`, các phép đo để trống. SHA khai báo không được trình bày như SHA đã tính lại từ ảnh. Các số `null` trong JSON là **NOT AVAILABLE**, không phải 0.

## Commands

Chạy từ root repository với Python có Pillow và NumPy. Dùng Python của `.venv` nếu nhóm đã có môi trường hợp lệ.

```powershell
# Audit baseline metadata và kiểm tra có file ở root đã chỉ định không
python scripts/dataset_v2.py audit --data data/cnn_dataset_v3 --root . --report reports/tv1_baseline_audit.json --quality-csv reports/tv1_baseline_quality.csv

# Snapshot metadata đầu tiên (inventory header-only, không cần raw root)
python scripts/dataset_v2.py build --baseline data/cnn_dataset_v3 --inventory docs/dataset_v2/capture_inventory.csv --output data/cnn_dataset_v4

# Khi có raw + camera/other: tạo VERSION MỚI, không ghi đè snapshot v4
python scripts/dataset_v2.py build --baseline data/cnn_dataset_v3 --root '<IMAGE_ROOT>' --inventory '<INVENTORY_CSV>' --output data/cnn_dataset_v5 --seed 42

# Duplicate + group/specimen leakage + quality nằm trong cùng một audit
python scripts/dataset_v2.py audit --data data/cnn_dataset_v5 --root '<IMAGE_ROOT>' --report reports/tv1_v5_audit.json --quality-csv reports/tv1_v5_quality.csv

python -m unittest tests.test_dataset_v2 -v
python -m unittest discover -s legacy/development/step2_split -p test_step2.py
python -m unittest discover -s legacy/development/step2b_grouping -p test_step2b.py
```

`declared_identity.duplicate_path`, `duplicate_sha256`, `cross_split_sha256`, `cross_split_group_id` là các kết quả audit metadata. `actual_byte_identity` chỉ có khi đã đọc được byte ảnh. `camera.grouping_violations` kiểm tra specimen across groups/splits. `quality` và quality CSV chứa phép đo thật khi raw sẵn có.

Exit code: **0** metadata và toàn bộ image bytes/decode đạt; **1** lỗi metadata/integrity/decode; **2** chưa kiểm chứng toàn bộ ảnh. Exit 0 vẫn không chứng minh đủ camera/other, specimen independence hoặc final test được khóa. Audit outputs phải ở ngoài thư mục dataset để không thay đổi lock/input.

Mọi version output là immutable: tool từ chối directory đã tồn tại. `data/`, `reports/`, raw/cache và inventory local không commit. Metadata snapshot v4 có thể tái tạo từ baseline đã tracked + code + header inventory.

## Preprocessing và augmentation đang dùng

Production decode `rgb_from_bytes`: một frame → EXIF transpose → alpha trên trắng nếu có → RGB. `Letterbox`: `ImageOps.pad` giữ aspect ratio, BICUBIC, trắng, center, 224×224. `ToTensor`: RGB HWC uint8 → CHW float 0–1. ImageNet mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]`, công thức `(x-mean)/std` theo channel.

Train: letterbox → horizontal flip p=0.5 → affine ±15°, translate 0.04, scale 0.9–1.02, BILINEAR, fill trắng → ColorJitter brightness/contrast 0.12, saturation 0.08, hue 0.01 → tensor/normalize. Val/test/inference: letterbox → tensor/normalize, không augmentation ngẫu nhiên. Không thay config; chưa chạy ablation mới hoặc kết luận cải thiện accuracy.

## Bàn giao

TV2 dùng report, supported manifest và quality sidecar để lên thí nghiệm khi raw đã có; giữ production training V3 hiện tại. TV3 dùng other manifest riêng cho thiết kế development/calibration/evaluation sau khi thống nhất protocol. Cả hai cần thống nhất image root, SHA256 byte gốc, split `val`, specimen evidence, candidate loader/version, quality measurement và final test governance. Nhóm xác nhận shared interface trước merge; TV1 không tự nhắn/merge branch khác.
