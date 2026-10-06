# TV1 Dataset V2 & Image Processing — FreshLens

Cập nhật 06/10/2026 trên branch `feat/data-v2`. **Đã audit ảnh thật và build candidate baseline-only; chưa hoàn thành nghiệm thu camera/hard cases/other.** Không tạo ảnh giả. Tài liệu đính kèm được dùng làm context; phạm vi thực hiện tuân theo yêu cầu người dùng.

## 1. Baseline

`data/cnn_dataset_v3`: 5869 bản ghi, đủ 8 class; **5869 ảnh đã đọc byte, hash và decode thực tế** trong phiên này.

| Mục | Baseline | Candidate V5 |
|---|---:|---:|
| Total supported | 5869 | 5869 |
| Train | 3519 | 3519 |
| Validation (val) | 1187 | 1187 |
| Test | 1163 | 1163 |
| Other | 0 | 0 |
| Unique group | 4786 | 4786 |

Fruit apple/banana/orange/tomato = 1762/1438/1493/1176. Fresh/rotten = 2895/2974. Groups train/val/test = 2871/958/957. Source `local`: 5869; bảy field bắt buộc không thiếu. Device, source dataset, license, specimen evidence và hard-case metadata chưa được xác minh.

### Mapping và tính bất biến

Raw hiện có: `data/raw/cnn_v3` gồm 15629 ảnh + README; `data/raw/cnn_v3_originals` gồm 5869 ảnh + matched catalog. Trạng thái hiện tại **RAW DATA AVAILABLE**; kết luận thiếu raw ở report cũ đã được thay bằng audit thực tế.

Manifest dùng `raw/fruit/status/file`, originals root không chứa prefix `raw`. 75 path còn khác tên file với manifest: catalog đã chọn file byte-identical khác tên. Audit dùng `--strip-prefix raw --path-map-csv data/raw/cnn_v3_originals/matched_originals.csv`, kiểm tra coverage và mọi metadata catalog trước SHA/decode. Không copy, rename, sửa raw hoặc tạo manifest trung gian. Lock V5 lưu 75 overrides.

5855 physical filenames thuộc dạng candidate_original; 14 có transform marker theo legacy parser. “Verified originals” ở đây nghĩa là **khớp SHA V3**, không bảo đảm đều là ảnh chụp nguyên thủy độc lập. Không tự đổi label/group/split dựa trên tên.

Fingerprint trước/sau audit/build vẫn bằng baseline:
- manifest.csv: `9be1db4f31ad753a75e54761dceda3aa4081d3add3b67baedccd4253ad83d291`
- final_split_report.json: `936dd0587808bd29ad6da664c8a81822a70213c49e1038a625b4752c1b28ca53`
- dataset_lock.json: `ce0a29333867905d09e489179c3b94fbb90b988a89b005ad5cc378cc03ae7945`

## 2. Dataset V2

Tên deliverable Dataset V2; storage candidate mới `data/cnn_dataset_v5`. Đã kiểm tra lock V4: snapshot metadata-only, ready=false; giữ nguyên V4. V5 reuse originals root, không nhân bản pixels.

- Supported: 5869 ảnh thật; added supported = 0.
- Camera có metadata/evidence: 0.
- Hard cases có tag được khai báo: 0.
- Other: 0; không admit unmatched.
- Build status: `VERIFIED_BASELINE_ONLY`; ready_for_pilot_training=true cho baseline-only candidate đã kiểm chứng ảnh.
- production_loader_compatible=false: không thay production V3 loader hoặc training entrypoint.

V5 có manifest, other manifest header-only, quality sidecar, report và lock. Adapter `load_candidate_dataset` kiểm tra lock/hash/metadata, map path trong bộ nhớ, giữ manifest_path và target tám nhãn. Readiness không chứng minh cải thiện accuracy/domain shift hoặc physical independence.

## 3. Supported distribution

| Fruit × status | Baseline | V5 | Train | Val | Test |
|---|---:|---:|---:|---:|---:|
| Apple Fresh | 921 | 921 | 555 | 185 | 181 |
| Apple Rotten | 841 | 841 | 495 | 172 | 174 |
| Banana Fresh | 624 | 624 | 376 | 126 | 122 |
| Banana Rotten | 814 | 814 | 489 | 163 | 162 |
| Orange Fresh | 748 | 748 | 449 | 153 | 146 |
| Orange Rotten | 745 | 745 | 445 | 151 | 149 |
| Tomato Fresh | 602 | 602 | 365 | 119 | 118 |
| Tomato Rotten | 574 | 574 | 345 | 118 | 111 |

Class imbalance lớn nhất/nhỏ nhất = 921/574 ≈1.60. Báo phân bố, không đổi loss/sampler hoặc training.

## 4. Other distribution

| Category | V5 images |
|---|---:|
| Other fruits | 0 |
| Non-fruit | 0 |
| Multiple | 0 |
| Total | 0 |

**OTHER DATA INCOMPLETE**. Chưa có ảnh other được review label/source/license/group. Chưa chứng minh diversity về loại quả/vật thể/nền/ánh sáng/góc/khoảng cách.

## 5. Split và final test

Giữ nguyên toàn bộ bảy field baseline: path, fruit, status, source, group_id, sha256, split. So sánh đủ 5869 hàng V5 với V3 đã pass. Không resplit hoặc sửa final test V3.

Group mới: union group+SHA toàn supported/other trước chia, seed42; component nối nhiều frozen splits bị từ chối. Pure-label component mới theo group 60/20/20; mixed mới vào train. Không ảnh mới trong candidate hiện tại nên không tạo split mới.

**FINAL TEST NOT YET FROZEN** cho final test camera mới độc lập. Internal test V3 kế thừa vẫn giữ nguyên lock/split; không coi nó là camera test mới. Không train/tune/model select trên test.

## 6. Duplicate audit và unmatched provenance

Baseline/V5: duplicate path = 0; duplicate SHA khai báo/thực đo = 0; cross-split SHA = 0. Decoded native RGB pixel identity cũng không có duplicate hoặc cross-split duplicate.

Raw partition được đối chiếu đủ filesystem và SHA:
- Matched originals: 5869.
- Duplicate originals: 75 file trùng baseline; không phải sample mới.
- Unmatched: 9685.
- Đã rehash cả 5869 originals, 5944 matched/duplicate raw và 9685 unmatched. Không sửa/xóa raw.

| Unmatched classification | Số ảnh |
|---|---:|
| Exact duplicate | 17 |
| Probable augmentation | 6502 |
| Probable new image | 0 |
| Unknown | 3166 |
| Corrupt | 0 |

17 exact duplicates là SHA-match với unmatched khác trong thứ tự canonical, không phải match baseline. 6502 probable augmentation có explicit transform filename và named parent cùng folder label; chưa chứng minh derivation bằng kiểm tra người. 3166 unknown gồm 2791 transform-marked không có named parent, 358 generic augmented không rõ parent, 17 unmarked thiếu provenance. Không mặc định unmatched là augmentation; **0 ảnh được admit**.

pHash legacy 63 DCT bits, Hamming ≤4: 920 unmatched có candidate gần baseline. Không dùng pHash để tự merge group, đổi label hoặc xác nhận specimen. Không có named-parent candidate hoặc pHash candidate set nối nhiều baseline splits trong review này; đây không chứng minh mọi near duplicate đã được tìm thấy. Crop/rotation có thể bị bỏ sót.

## 7. Leakage và physical specimen

Cross-split group = 0; cross-split SHA thực đo = 0; cross-split decoded-pixel = 0. Kiểm tra trên đủ 5869 ảnh.

Camera grouping: **NOT AVAILABLE** vì chưa có camera evidence. Không có specimen violations trong các hàng khai báo hiện tại không đồng nghĩa đã xác minh physical independence. Group ID khác nhau chưa chứng minh quả vật lý khác nhau; SHA/pHash không thể thay nhật ký thu thập.

## 8. Image quality

Đo đủ 5869 ảnh; missing/read error/SHA mismatch/decode error đều 0.

| Flag | Ảnh |
|---|---:|
| Low resolution, min side <96 | 3 |
| Low brightness, mean L <35 | 12 |
| Blur candidate, Laplacian variance <50 | 1891 |
| LOW_QUALITY, hợp các flag | 1896 |
| OK | 3973 |

Flag có thể chồng nhau. Brightness là mean Pillow L 0–255; blur là variance Laplacian bốn lân cận, reflect boundary, trên native oriented RGB trước resize. Threshold diagnostic chưa được hiệu chỉnh bằng experiment; texture/nền/resolution ảnh hưởng blur. Không loại ảnh hoặc đổi inference quality threshold.

| Class | Small | Dark | Blur flag | LOW_QUALITY | Brightness median | Blur median |
|---|---:|---:|---:|---:|---:|---:|
| Apple Fresh | 0 | 0 | 317 | 317 | 140.45 | 313.23 |
| Apple Rotten | 0 | 0 | 409 | 409 | 176.57 | 55.10 |
| Banana Fresh | 0 | 0 | 185 | 185 | 135.71 | 181.22 |
| Banana Rotten | 0 | 0 | 235 | 235 | 152.00 | 79.27 |
| Orange Fresh | 0 | 0 | 342 | 342 | 147.25 | 83.75 |
| Orange Rotten | 0 | 0 | 242 | 242 | 142.31 | 93.89 |
| Tomato Fresh | 0 | 0 | 50 | 50 | 124.02 | 440.45 |
| Tomato Rotten | 3 | 12 | 111 | 116 | 137.14 | 252.32 |

Width 80–4160, height 95–4160. Formats JPEG 3247, PNG 2612, WEBP 10. Original modes RGB 4740, RGBA 1127, P 2; channels 3/4/1 tương ứng. EXIF present 1336; orientation thiếu 4662, 1:1197, 6:2, 3:8. Mode/channels là trước RGB conversion. JSON/CSV quality summary có min/quantiles/mean và resolution bins từng class.

## 9. Camera và hard cases

Ảnh camera/source/lighting/background/hard-case có evidence: **NOT AVAILABLE**. Không suy ra camera từ EXIF hoặc small/dark flags. Header inventory và guide đã có, chưa có ảnh mới.

Kế hoạch development 20–40 ảnh khó là mục tiêu thu, chưa phải số đã có. Thu đủ tám tổ hợp, ưu tiên apple fresh/rotten, orange fresh, tomato rotten. Phone/laptop, ánh sáng vàng/yếu, bóng đổ, nền phức tạp, tay cầm, góc nghiêng, gần/xa, quả nhỏ, hỏng nhẹ/vết hỏng nhỏ. Thu nhiều physical specimens; dùng cùng group cho cùng quả ở mọi góc/session. Initial development train/val; final camera test riêng cần protocol và khóa trước tuning.

## 10. Preprocessing và kiến thức Xử lý ảnh

Production pipeline giữ nguyên: image bytes → một frame → EXIF orientation correction → alpha composite trắng → RGB → letterbox224 → tensor → ImageNet normalize.

- EXIF lưu orientation của camera; transpose trước đo/resize tránh hiển thị và input khác hướng.
- RGB là thứ tự channel PIL/torch; OpenCV thường BGR, cần chuyển khi đi qua OpenCV. Không đổi convention CNN.
- Resize đổi kích thước; resize ép vuông có thể méo aspect ratio. Letterbox bằng ImageOps.pad giữ tỷ lệ và thêm nền trắng, center.
- BICUBIC cho letterbox, BILINEAR affine; interpolation ảnh hưởng chi tiết cục bộ, không tự đổi baseline.
- Tensor chuyển HWC uint8 thành CHW float 0–1; ImageNet normalize (x-mean)/std, mean [0.485,0.456,0.406], std [0.229,0.224,0.225].
- Data leakage gồm byte/near duplicate hoặc cùng specimen qua splits. Group split giữ nhóm chung nhưng chỉ mạnh bằng evidence grouping.
- Duplicate SHA là byte-identical; decoded-pixel equality phát hiện encoding khác nhưng RGB giống; pHash chỉ tạo candidate.
- Domain shift là khác camera/nền/ánh sáng/khoảng cách so với nguồn training. Candidate hiện chưa thêm domain camera mới.
- Class imbalance là phân bố nhãn không đều, báo tỷ trọng để TV2 cân nhắc experiment.

CLAHE tăng local contrast; Gaussian Blur làm mượt; Canny/Sobel biểu diễn biên/gradient, có thể giữ diagnostic/baseline/visualization. Không đưa vào CNN pipeline; Canny không xác định vùng hỏng; chưa có bằng chứng tăng contrast tăng accuracy.

## 11. Augmentation

Production train giữ nguyên: letterbox → horizontal flip p0.5 → affine ±15°, translate0.04, scale0.9–1.02, BILINEAR trắng → ColorJitter brightness/contrast0.12, saturation0.08, hue0.01 → tensor/normalize.

Val/test/inference: letterbox → tensor/normalize, không augmentation ngẫu nhiên. Không bật MixUp/CutMix, không đổi image size/normalization hoặc tune augmentation trên test. Không tạo experiment augmentation mới khi chưa có ảnh camera.

## 12. Other / Unsupported và shared contract

Other là data category riêng, không class thứ chín. Không sửa CLASSES, output dimension, loss, inference threshold, prediction contract.

TV2 dùng V5 supported manifest/quality/lock và adapter trong `freshlens_ai/data/dataset_v2.py`, image root originals; production training cần integration TV2 riêng, không tự nhận lock candidate. TV3 dùng other_manifest riêng (hiện empty), provenance report và camera protocol khi có dữ liệu. Nhóm cần thống nhất version/root/mapping, tám label indices, split val, group/specimen evidence, source/license, development vs final-test protocol. Không merge/cherry-pick hoặc nhắn branch khác.

## 13. Reproducibility, testing và limitations

Commands đầy đủ audit/build/provenance/duplicate/group/quality tại `docs/dataset_v2/README.md`. Duplicate/leakage/quality dùng chung audit CLI, không tạo script trùng chức năng.

Artifacts local:
- reports/tv1_v3_originals_audit.json
- reports/tv1_v3_originals_quality.csv
- reports/tv1_unmatched_provenance.json
- reports/tv1_unmatched_provenance.csv
- reports/tv1_quality_summary.json
- reports/tv1_quality_summary.csv
- reports/tv1_leakage_audit.json

V5 local: manifest.csv, other_manifest.csv, image_quality.csv, dataset_report.json, dataset_lock.json. Snapshot audit review được cập nhật trong docs/dataset_v2/baseline_audit.json và candidate_audit.json. Không commit raw/candidate/reports/cache/checkpoint.

Testing trên Python C314:
- `python -m pytest tests/test_dataset_v2.py -q`: **27 passed**.
- Legacy step2_split: **24 passed**; step2b_grouping: **16 passed**.
- `python -m pytest -q`: **29 passed, 1 error**.
- Candidate smoke: đủ baseline fields 5869 hàng giữ nguyên; lock V3/V4/V5 đạt; 12 mẫu đủ tám labels, ba splits và alias tạo tensor 3×224×224 finite, validation/test deterministic.

**OUT OF SCOPE**: tests/refactor/TEST_TRAINING_REFACTOR.py::test_optimizer, thiếu fixture stage khi pytest collect helper như test. Owner đề xuất TV2 (training/refactor). Lỗi có sẵn, không do TV1; giữ nguyên test, không tạo fixture giả và không sửa training/model. Không có REGRESSION DETECTED còn tồn tại trong test TV1.

Giới hạn còn lại:
- Không có camera/hard-case/other mới; diversity chưa đạt.
- Không kiểm chứng physical specimen independence hoặc chất lượng label bằng review người.
- 14 SHA-matched physical filenames có transform marker; giữ nguyên baseline, không kết luận độc lập.
- 3166 unmatched còn unknown; probable augmentation cũng cần review trước sử dụng.
- Không có final camera test mới độc lập/frozen; không claim cải thiện accuracy/domain shift.
- Quality threshold là candidate diagnostic; giữ 1896 flagged images, không tự xóa.
- Root/image bytes và catalog cần còn sẵn có để reproduce; V5 metadata lock không phải production V3 lock.

TV1 đã hoàn tất công cụ/audit/candidate baseline-only/report và data adapter. Acceptance camera, hard cases, other diversity, specimen evidence và final-test protocol **chưa hoàn tất**.
