# TV1 Dataset V2 & Image Processing — FreshLens

Audit ngày 06/10/2026, branch `feat/data-v2`. **Công cụ và metadata snapshot đã có; Dataset V2 thực tế chưa đạt nghiệm thu vì chưa có raw/camera/hard cases/other.** Không tạo ảnh hoặc số liệu thay thế. Tài liệu phân công Word được đọc như context; những đề xuất model/UI trong đó không mở rộng phạm vi người dùng cho TV1.

## 1. Baseline

Baseline: `data/cnn_dataset_v3/`, 8 supported classes. Số dưới đây được đếm trực tiếp từ manifest, không phải số ảnh raw đã kiểm chứng trong phiên này.

| Mục | Bản ghi |
|---|---:|
| Tổng | 5869 |
| Train | 3519 |
| Validation (`val`) | 1187 |
| Test | 1163 |
| Unique group | 4786 |
| Source `local` | 5869 |

Fruit: apple 1762, banana 1438, orange 1493, tomato 1176. Status: fresh 2895, rotten 2974. Groups train/val/test = 2871/958/957. Bảy field bắt buộc không thiếu giá trị; metadata nguồn chỉ là `local`, không đủ source dataset/device/license/specimen provenance.

Manifest SHA256: `9be1db4f31ad753a75e54761dceda3aa4081d3add3b67baedccd4253ad83d291`.
Report SHA256: `936dd0587808bd29ad6da664c8a81822a70213c49e1038a625b4752c1b28ca53`.
Cả hai khớp `dataset_lock.json`. Ba file baseline được fingerprint trước/sau build và giữ nguyên.

**RAW DATA NOT AVAILABLE.** Không có `raw/...` được manifest tham chiếu ở repository root: 5869/5869 file không tìm thấy tại root đã kiểm tra. Điều này không chứng minh ảnh không tồn tại ở máy/cloud khác. Repo chỉ có metadata V3, không có image root được người dùng xác nhận. Con số `local_image_files_verified=5869` trong report V3 là bằng chứng lịch sử, không phải kiểm chứng lại hiện tại.

## 2. Dataset V2

Deliverable TV1 gọi Dataset V2, output riêng `data/cnn_dataset_v4/` theo version storage hiện tại. Snapshot đầu tiên giữ 5869 supported metadata rows từ V3; **raw V2 total = NOT AVAILABLE**. Không nhân bản dữ liệu pixel và không thay đổi baseline.

| Mục | Metadata hiện có | Ảnh thật mới |
|---|---:|---|
| Supported | 5869 | NOT AVAILABLE |
| Camera | 0 | NOT AVAILABLE |
| Hard cases được gắn metadata | 0 | NOT AVAILABLE |
| Other | 0 | NOT AVAILABLE |

`manifest.csv` supported có đủ bảy field production, capture metadata và quality metadata. `other_manifest.csv` chỉ có header vì chưa có ảnh unknown. `image_quality.csv` có 5869 hàng trạng thái chưa đo. `dataset_report.json` ghi lineage, incomplete status, split policy; `dataset_lock.json` fingerprint các file output. `ready_for_pilot_training=false`, `production_loader_compatible=false`.

## 3. Supported distribution

| Fruit × status | Baseline total | V2 metadata | Train | Val | Test |
|---|---:|---:|---:|---:|---:|
| Apple Fresh | 921 | 921 | 555 | 185 | 181 |
| Apple Rotten | 841 | 841 | 495 | 172 | 174 |
| Banana Fresh | 624 | 624 | 376 | 126 | 122 |
| Banana Rotten | 814 | 814 | 489 | 163 | 162 |
| Orange Fresh | 748 | 748 | 449 | 153 | 146 |
| Orange Rotten | 745 | 745 | 445 | 151 | 149 |
| Tomato Fresh | 602 | 602 | 365 | 119 | 118 |
| Tomato Rotten | 574 | 574 | 345 | 118 | 111 |

Chưa có ảnh mới để kết luận cải thiện domain shift. Class imbalance lớn nhất/nhỏ nhất = 921/574 ≈1.60; TV1 chỉ báo phân bố, không tự đổi loss/sampler.

## 4. Other distribution

**OTHER DATA INCOMPLETE**.

| Category | Metadata rows | Real images |
|---|---:|---|
| Other fruits | 0 | NOT AVAILABLE |
| Non-fruit | 0 | NOT AVAILABLE |
| Multiple | 0 | NOT AVAILABLE |
| Other tổng | 0 | NOT AVAILABLE |

Inventory mẫu và collection guide yêu cầu nhiều fruit/object types, backgrounds, lighting, viewpoints, distances và sources. Chưa có bằng chứng tính đa dạng. Không tạo `other` thành production class thứ chín.

## 5. Split

Snapshot V2 metadata giữ train/val/test = 3519/1187/1163. Toàn bộ group ID và split V3 được bảo toàn.

Ảnh mới: union group/SHA xuyên hai manifest; component có split đã khóa kế thừa split đó; conflicting frozen splits dừng build. Component mới thuần nhãn reuse group split 60/20/20 seed 42; mixed-label components mới vào train. Không tự xóa ảnh/đổi nhãn. Với ít independent groups, báo bucket trống, không nhân bản để lấp split.

**FINAL TEST NOT YET FROZEN.** Giữ internal test V3 cố định không có nghĩa nó là final camera test mới độc lập. External V2 đã dùng gate calibration theo README. Chưa có protocol, collection log hoặc dataset final mới được khóa. Không dùng test để chọn augmentation, hyperparameters hoặc threshold.

## 6. Duplicate audit

| Check | V3 metadata | V2 metadata | Recomputed raw bytes |
|---|---:|---:|---|
| Duplicate path keys | 0 | 0 | NOT AVAILABLE |
| Duplicate SHA256 keys | 0 | 0 | NOT AVAILABLE |
| Cross-split SHA256 keys | 0 | 0 | NOT AVAILABLE |

SHA được đọc từ manifest và lock xác nhận tính toàn vẹn metadata. Không gọi đó là hash đã tính lại từ ảnh trong phiên này. Khi raw có, audit tính actual SHA; mismatch dừng gate/build. Duplicate cùng split được giữ và báo cáo; conflicting labels hoặc frozen split conflict yêu cầu review. Không xóa hàng loạt.

## 7. Leakage audit

Cross-split group keys = **0** cho V3 và V2 theo metadata. Audit kiểm tra thêm specimen ID xuất hiện nhiều group/split. Inventory camera yêu cầu evidence, một specimen ID cùng group; chưa có camera rows để kiểm chứng grouping thực tế.

**Physical specimen independence NOT VERIFIED.** V3 có 4786 group, không thể suy ra 4786 quả vật lý khác nhau. Report V3 cũng ghi `physical_specimen_independence_confirmed=false`. Exact SHA không phát hiện góc chụp khác, crop, rotate hoặc re-encoding. Perceptual review mới chưa thực hiện vì không có raw; không sao chép các cờ review V3 sang candidate mới.

## 8. Image quality

| Metric | Kết quả dữ liệu thật |
|---|---|
| Decode errors | NOT AVAILABLE |
| Low resolution | NOT AVAILABLE |
| Low brightness | NOT AVAILABLE |
| Blur | NOT AVAILABLE |
| Dimensions/formats/channels/EXIF | NOT AVAILABLE |

Audit có width/height, aspect ratio, brightness, Laplacian variance, file size, original mode/channels, format, EXIF presence/orientation và hash status. Đo sau EXIF/RGB tại native resolution, trước resize/augmentation. Candidate thresholds: min side 96 px, brightness 35 trên 0–255, Laplacian variance 50. Chưa calibrate trên ảnh thật; không áp dụng production inference và không tự loại ảnh. `NOT_AVAILABLE`/`MISSING_FILE` không được biến thành `OK` hoặc count 0 cho lỗi chưa đo.

## 9. Camera

Camera phone/laptop: **NOT AVAILABLE**. Lighting/background/viewpoint/distance/hard cases: **NOT AVAILABLE**. Camera grouping thực tế: **NOT VERIFIED**.

Kế hoạch thu: đủ tám tổ hợp, ưu tiên apple fresh/rotten, orange fresh, tomato rotten; 20–40 ảnh development khó là mục tiêu ban đầu, chưa phải số đã thu. Yêu cầu nhiều physical specimens, ánh sáng vàng/yếu, shadow, nền phức tạp, hand-held, tilted, close/far, small-in-frame, small damage/mild rot. Nhật ký specimen dùng để gắn group cho mọi view và lần chụp. Xem `docs/dataset_v2/README.md` và header inventory.

## 10. Preprocessing và kiến thức xử lý ảnh

Production thực tế được reuse và giữ nguyên:

`bytes → single-frame decode → EXIF transpose → RGB (alpha composite white) → white letterbox 224×224/BICUBIC → tensor → ImageNet normalization`.

- EXIF orientation là metadata yêu cầu xoay/lật ảnh từ camera; transpose trước đo kích thước và đưa vào model.
- Pillow/torchvision dùng RGB. OpenCV mặc định BGR; đảo convention sai làm sai kênh màu. Audit dùng RGB production decoder, không thêm BGR conversion vào CNN.
- Resize ép trực tiếp về hình vuông có thể méo aspect ratio. Letterbox resize giữ aspect ratio rồi pad; implementation production dùng `ImageOps.pad`, center và fill trắng.
- Interpolation BICUBIC dùng trong letterbox; BILINEAR trong random affine. Không tự đổi sang kernel khác hay input size khác.
- ToTensor tạo float CHW, scale 0–1; normalize `(x-mean)/std` với mean `[0.485,0.456,0.406]`, std `[0.229,0.224,0.225]` đúng checkpoint contract.
- Data leakage là liên kết nội dung/specimen xuất hiện ở các split đánh giá và train. Group split gán toàn bộ group vào cùng split; SHA duplicate là byte identity, không chứng minh specimen identity.
- Domain shift là khác biệt nguồn/camera/ánh sáng/nền giữa train và demo, cần dữ liệu thực tế và đánh giá độc lập. Chưa có experiment để nói đã giảm domain shift.
- Class imbalance có thể ảnh hưởng recall từng lớp; TV2 quyết định sampler/loss sau protocol phù hợp.
- CLAHE tăng contrast cục bộ, Gaussian Blur làm trơn, Sobel ước lượng gradient và Canny phát hiện cạnh. Chỉ diagnostic/baseline trong legacy; không thêm vào CNN production. Canny không xác định vùng hỏng; tăng contrast/blur có thể làm sai màu hoặc mất vết hỏng nhỏ, chưa có experiment chứng minh cải thiện.

## 11. Augmentation

Train hiện tại: letterbox → RandomHorizontalFlip (p=0.5) → RandomAffine degrees=15, translate=(0.04,0.04), scale=(0.9,1.02), BILINEAR, trắng → ColorJitter brightness=0.12, contrast=0.12, saturation=0.08, hue=0.01 → tensor/normalize.

Val/test/inference dùng `training=False`: không flip/affine/jitter ngẫu nhiên. Đã đọc train/eval/inference call sites. Không đổi augmentation, không bật MixUp/CutMix, không chạy augmentation ablation hoặc tuyên bố cải thiện accuracy. Bộ test production tensor-equivalence chưa thực thi vì thiếu torch và raw images.

## 12. Other / Unsupported

Other dùng manifest riêng với `path,category,subcategory,source,group_id,sha256,split` và capture/quality sidecars. Không có fruit/status labels production. Hai manifest được kiểm tra group/SHA cùng nhau; không đánh giá độc lập hai tập rồi bỏ qua leakage giữa chúng. Schema/lock mới chỉ phục vụ chuẩn bị dữ liệu; integration model/gate do TV2/TV3 thống nhất sau.

## 13. Reproducibility và testing

Commands chạy từ repo root, xem chi tiết và placeholders image root tại `docs/dataset_v2/README.md`:

```powershell
python scripts/dataset_v2.py audit --data data/cnn_dataset_v3 --root . --report reports/tv1_baseline_audit.json --quality-csv reports/tv1_baseline_quality.csv
python scripts/dataset_v2.py build --baseline data/cnn_dataset_v3 --inventory docs/dataset_v2/capture_inventory.csv --output data/cnn_dataset_v4
python scripts/dataset_v2.py audit --data data/cnn_dataset_v4 --report reports/tv1_v4_audit.json --quality-csv reports/tv1_v4_quality.csv
```

Cùng lệnh `audit` kiểm tra duplicate, group/specimen leakage và quality. Khi có raw, thêm `--root '<IMAGE_ROOT>'`; khi thêm dữ liệu, dùng inventory thật và version output mới. Exit **2** cho incomplete verification, **1** cho errors, **0** cho metadata/byte/decode đã kiểm chứng; không dùng exit 0 để chứng minh acceptance đầy đủ.

Các report audit metadata chia sẻ tại `docs/dataset_v2/baseline_audit.json` và `docs/dataset_v2/candidate_audit.json`. Snapshot data và quality CSV dưới `data/`/`reports/` là local ignored outputs, không commit raw/cache/artifacts.

Tests TV1: kiểm tra giữ baseline bytes, immutable outputs, lock tampering, split deterministic khi đổi thứ tự input, transitive group/SHA inheritance, conflicting labels/splits, global supported-other grouping, physical specimen violations, paths unsafe, EXIF/alpha/grayscale, dark/blur/small/broken/missing/hash mismatch. Pixel test fixtures chỉ có trong thư mục tạm, không được dùng như dữ liệu đời thực.

TV1 suite: 16 tests PASS. Legacy split suite: 24 tests PASS. Legacy grouping suite: 16 tests PASS. Tổng 56 tests PASS. Production `tests.refactor.TEST_DATA_REFACTOR` không chạy được: `ModuleNotFoundError: torch`; raw sample cũng chưa có. Lần đầu các suite dùng sandbox TEMP bị PermissionError; chạy lại với TEMP/TMP trong workspace và không đổi test cũ đã pass. Chưa phát hiện regression TV1; không tuyên bố toàn bộ production suite pass.

## 14. Limitations và OUT OF SCOPE

- Raw baseline unavailable, chưa đo quality/decode/EXIF hoặc tính lại image hashes.
- Camera/hard cases/other chưa có; other diversity và specimen independence chưa xác minh.
- Candidate hiện chỉ chứa baseline metadata, không có dữ liệu mới; chưa production-loader compatible hoặc ready for training.
- Final camera test chưa được thu/khóa; External V2 đã calibration.
- Quality thresholds chưa được thử nghiệm; near-duplicate review và augmentation ablation chưa thực hiện.
- Môi trường Python thiếu torch/torchvision, nên production tensor-equivalence/integration chưa chạy.

| OUT OF SCOPE — file | Problem | Reason | Suggested owner |
|---|---|---|---|
| `scripts/sync_cnn_dataset.py`, `scripts/publish_cnn_dataset.py` | Import `scripts.db_manager`, nhưng module đó không có trong `scripts/` hiện tại; helper chỉ có trong legacy | Khôi phục cloud infrastructure/credentials không thuộc offline dataset audit này; không tự di chuyển legacy/cloud code | TV4/integration + người quản lý dữ liệu cloud |
| `freshlens_ai/data/locked_dataset.py` và training consumers | V3 loader yêu cầu report `2.1.0`, recomputed candidates và pilot-ready; candidate lock mới không đáp ứng | Không được giả định evidence hoặc nới gate để train candidate chưa hoàn thiện | TV1 phối hợp TV2/TV3 khi đủ dữ liệu |
| `docs/TEAM_ASSIGNMENT.md` | Nội dung phân công cũ còn tập trung SVM, khác yêu cầu CNN hiện tại | Không refactor tài liệu toàn nhóm theo tài liệu cũ | TV4/docs coordinator |

## 15. Handoff và acceptance

TV2: baseline V3 vẫn là production data; đọc report + `data/cnn_dataset_v4/manifest.csv`, quality sidecar và guide để chuẩn bị candidate experiment sau khi có raw. Không train snapshot hiện tại như dataset đã hoàn thiện. TV3: dùng schema `other_manifest.csv` và capture metadata cho future development/calibration, giữ tám class và prediction contract. Không merge/cherry-pick branch TV3.

Shared contract cần thống nhất: image root tương đối, byte SHA, `val`, supported/other separation, global group/specimen ID, evidence/provenance, loader/lock candidate, quality measurement version và quyền giữ final test độc lập. Trước merge phải review thay đổi data API lazy import.

- [x] Baseline metadata preserved; snapshot độc lập; đủ tám supported metadata combinations.
- [x] Manifest/SHA/group metadata; zero declared cross-split SHA/group; reproducible audit/build.
- [x] Quality tooling + unavailable status; preprocessing/augmentation review; model/training/inference/UI giữ nguyên.
- [ ] Raw baseline restored and image hashes/decode/quality verified.
- [ ] Real camera images, hard cases, diverse other and physical specimen evidence collected.
- [ ] New perceptual/grouping review; final independent camera test frozen.
- [ ] Candidate loader handoff agreed; production data-equivalence test executed with valid environment/raw.

TV1 **chưa đạt nghiệm thu dữ liệu đầy đủ**; phần công cụ, audit metadata và documentation sẵn sàng để tiếp tục sau khi thu/khôi phục ảnh thật.
