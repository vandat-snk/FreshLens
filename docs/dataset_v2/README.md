# TV1 Dataset V2 — pipeline và bàn giao

## Trạng thái và phạm vi

TV1 có audit SHA/decode/quality/provenance, inventory camera/OTHER, review gate, builder candidate từ V5, split theo group+content+specimen, CLI và Dataset adapter dùng preprocessing production. Đây là **code pipeline**, chưa phải final Dataset V2 có đủ camera/hard cases/other.

V3/final test và 5869 baseline originals bất biến. V4 metadata snapshot cũ giữ nguyên. **V5** là candidate baseline-only đã kiểm chứng đủ 5869 ảnh: train3519/val1187/test1163. Các file V5, bảy field baseline và splits không bị sửa. Images ở data/raw/cnn_v3_originals; mapping V5 bỏ raw prefix và có 75 alias SHA-identical. Không copy/rename originals.

camera_v1 hiện 22 ảnh (banana fresh3/rotten19), nguồn/specimen chưa biết, tất cả UNASSIGNED và không eligible. OTHER chưa có ảnh. Không tải web hoặc tạo pixel data; ảnh synthetic chỉ xuất hiện trong fixture test tạm.

## Workflow

1. Thêm ảnh thật vào camera/external/OTHER root.
2. Điền metadata path tương đối root, source và provenance; cùng physical specimen/cảnh dùng chung group_id toàn bộ nguồn.
3. Chạy camera/other. Xem quality, duplicate, pHash, group và provenance warnings.
4. Human review nhãn/source/license/specimen/duplicates. Chỉ bật eligibility trong **metadata input**, không sửa generated manifest.
5. Chạy builder dry-run. Record chưa eligible bị loại khỏi danh sách nhận; record eligible nhưng thiếu evidence gây lỗi, không im lặng admit.
6. Chỉ khi đã review, chạy build với **output mới**. TV2 chủ động dùng adapter/experiment riêng, không auto thay production dataset.
7. Final camera test độc lập cần protocol riêng; builder hiện không thêm record mới vào test.

## Commands

Chạy từ repository root với Python có Pillow/NumPy. Dataset adapter cần torch/torchvision.

```powershell
python scripts/dataset_v2.py --help
python scripts/dataset_v2.py camera --help
python scripts/dataset_v2.py build --help

# Một manifest/report, scan lại được và không xóa low-quality/duplicate
python scripts/dataset_v2.py camera --root data/camera_v1 --baseline data/cnn_dataset_v5 --metadata docs/dataset_v2/camera_metadata.csv --output reports/camera_v1
python scripts/dataset_v2.py other --root data/other --baseline data/cnn_dataset_v5 --metadata docs/dataset_v2/other_metadata.csv --output reports/other

# Audit aliases: --root/metadata cho phép đo mới; bỏ --root chỉ đọc saved report có hash manifest
python scripts/dataset_v2.py audit --data reports/camera_v1 --root data/camera_v1 --metadata docs/dataset_v2/camera_metadata.csv
python scripts/dataset_v2.py duplicates --data reports/camera_v1 --root data/camera_v1 --metadata docs/dataset_v2/camera_metadata.csv
python scripts/dataset_v2.py leakage --data reports/camera_v1 --root data/camera_v1 --metadata docs/dataset_v2/camera_metadata.csv
python scripts/dataset_v2.py quality --data reports/camera_v1 --root data/camera_v1 --metadata docs/dataset_v2/camera_metadata.csv

# V5: tự đọc prefix/alias từ lock nếu không override
python scripts/dataset_v2.py audit --data data/cnn_dataset_v5 --root data/raw/cnn_v3_originals --fingerprints --report reports/tv1_v5_audit.json --quality-csv reports/tv1_v5_quality.csv

# Review candidate, không tạo output
python scripts/dataset_v2.py build --baseline data/cnn_dataset_v5 --root data/raw/cnn_v3_originals --sources-json docs/dataset_v2/candidate_sources.json --dry-run

# Sau khi có eligibility + evidence hợp lệ, dùng VERSION MỚI
python scripts/dataset_v2.py build --baseline data/cnn_dataset_v5 --root data/raw/cnn_v3_originals --sources-json docs/dataset_v2/candidate_sources.json --output data/cnn_dataset_v6_candidate

# Audit candidate nhiều image roots
python scripts/dataset_v2.py audit --data data/cnn_dataset_v6_candidate --root data/raw/cnn_v3_originals --sources-json docs/dataset_v2/candidate_sources.json --report reports/tv1_v6_audit.json

python -m pytest tests/test_dataset_v2.py -q
python -m pytest tests/test_camera_dataset.py -q
python -m pytest tests/test_reviewed_candidate.py -q
python -m pytest -q
```

V3 audit/provenance/build CLI cũ vẫn có. V3 build là baseline-only để không bypass review; ảnh mới dùng V5 sources builder. Không chạy build vào V5 đã tồn tại.

```powershell
python scripts/dataset_v2.py audit --data data/cnn_dataset_v3 --root data/raw/cnn_v3_originals --strip-prefix raw --path-map-csv data/raw/cnn_v3_originals/matched_originals.csv --fingerprints --report reports/tv1_v3_originals_audit.json --quality-csv reports/tv1_v3_originals_quality.csv --quality-summary reports/tv1_quality_summary.json --quality-summary-csv reports/tv1_quality_summary.csv --leakage-report reports/tv1_leakage_audit.json

python scripts/dataset_v2.py provenance --baseline data/cnn_dataset_v3 --originals-root data/raw/cnn_v3_originals --raw-root data/raw/cnn_v3 --original-quality-csv reports/tv1_v3_originals_quality.csv --matched-csv data/raw/cnn_v3_audit/matched_originals.csv --duplicates-csv data/raw/cnn_v3_audit/duplicate_originals.csv --unmatched-csv data/raw/cnn_v3_audit/unmatched_images.csv --report reports/tv1_unmatched_provenance.json --csv reports/tv1_unmatched_provenance.csv
```

Raw provenance vẫn phân biệt exact duplicate/probable augmentation/probable new/unknown/corrupt; filename/pHash không tự chứng minh augmentation. 9685 unmatched chưa được admit.

## Chọn nguồn

candidate_sources.json là danh sách nguồn explicit: id duy nhất, kind=supported|other, root, metadata. Root/metadata tương đối **repository cwd**, không theo vị trí JSON. Không hard-code đường dẫn cá nhân. Mặc định chọn camera_v1 và OTHER để review, nhưng không tự nhận mọi ảnh.

Supported external dùng cùng tám folder labels như camera; đặt source=web|external|dataset_external trong metadata. Có thể dùng camera command với root external riêng. OTHER root:
- data/other/fruit/mango/file.jpg
- data/other/fruit/pear/file.jpg
- data/other/non_fruit/phone/file.jpg
- data/other/multiple/fruit_and_object/file.jpg

OTHER root chưa tồn tại được xem là empty, không tạo ảnh/folder giả. OTHER có label_type=OTHER, fruit/status=UNKNOWN, category=other_fruit|non_fruit|multiple và subcategory; **luôn eligible_for_training=false**. eligible_for_evaluation=true chỉ nhận vào other_manifest riêng sau review; dù split development là train/val, adapter tám nhãn không bao giờ lấy OTHER.

## Review gate

Metadata thiếu → UNKNOWN; phép đo không có → NOT_AVAILABLE. group_id=UNKNOWN nếu chưa biết specimen; không tự tạo specimen ID theo hash/tên file.

Supported muốn được nhận phải eligible_for_training=true, review_status=APPROVED, reviewer, review_evidence, label_evidence, group_id, specimen_evidence, source và source_name rõ ràng; SHA/decode phải đạt. Camera cần specimen_id/capture_device/capture_context/session_id. Web/external cần source_url/license. Tool kiểm tra khai báo, không thay human/license review.

pHash candidates chưa có exact byte/pixel evidence cần perceptual_review_status=APPROVED và perceptual_review_evidence. pHash là candidate, không tự merge groups. Unknown group/evidence hoặc SHA/decode lỗi → builder dừng trước khi tạo output.

requested_split=UNASSIGNED|train|val. test bị từ chối cho record mới. Camera manifest split vẫn UNASSIGNED; requested_split chỉ lưu đề xuất reviewer.

## Split/duplicate và candidate artifacts

Union toàn bộ group_id, specimen_id (nếu có), SHA và native decoded RGB pixel SHA xuyên supported/OTHER/baseline. Content label conflict bị từ chối; component nối nhiều frozen splits bị từ chối; overlap baseline test bị từ chối. Không xóa duplicate. Specimen ID ở nhiều group phải review, không tự giả định độc lập.

Giữ baseline splits. Nhóm mới chưa đề xuất split dùng hash seed42 + global group ID, deterministic 80/20 train/val. Không đảm bảo stratification hoặc val đủ lớp khi ít groups. Cùng component luôn cùng split. Không augment trước split. Không tạo final test mới hoặc claim independent physical specimens.

Candidate riêng gồm manifest.csv, other_manifest.csv, image_quality.csv, dataset_report.json, dataset_lock.json. Schema reviewed lock khác V3/V5; production loader không chấp nhận. Baseline quality được reuse từ V5 lock **sau rehash từng baseline image**; new image đo/hash/decode mới. Bảy field baseline được so sánh nguyên trạng. dry-run kiểm tra cùng gates mà không ghi output.

Dataset report có totals/distribution/splits, missing/decode/SHA, path/SHA/pixel duplicate, group/specimen leakage, brightness/blur/quality, source, hard cases, provenance completeness và pHash candidate counts mới. Không flag quality thành quyết định loại dữ liệu.

Candidate mới lưu root_id và image_path; không copy pixels và không lưu absolute machine path vào manifest. TV2 cung cấp image-root mapping khi đọc:

```python
from freshlens_ai.data.reviewed_candidate import make_reviewed_dataset
roots = {
    "baseline": "data/raw/cnn_v3_originals",
    "camera_v1": "data/camera_v1",
    "other": "data/other",
}
train = make_reviewed_dataset("data/cnn_dataset_v6_candidate", roots, "train", training=True)
val = make_reviewed_dataset("data/cnn_dataset_v6_candidate", roots, "val")
test = make_reviewed_dataset("data/cnn_dataset_v6_candidate", roots, "test")
```

V5 adapter load_candidate_dataset vẫn giữ API cũ. New adapter xác minh reviewed lock, đọc đúng byte+SHA qua production image_io, chỉ lấy supported rows. Bật training augmentation cho val/test gây lỗi. Không sửa training entrypoint.

## Preprocessing/augmentation/quality

Reuse production decoder: một frame → EXIF transpose → alpha composite trắng → RGB. Letterbox ImageOps.pad giữ tỷ lệ, BICUBIC, center trắng 224×224; tensor CHW 0–1; normalize ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225]. Không đổi image contract.

Train giữ flip p0.5, affine ±15°, translate0.04, scale0.9–1.02, BILINEAR trắng; jitter brightness/contrast0.12, saturation0.08, hue0.01. Val/test/inference không random augmentation. Split seed42 reproducible; augmentation dùng PyTorch RNG: TV2 đặt torch.manual_seed và worker seeding trong experiment của mình. TV1 test tái tạo augmentation trong single process, không claim đã cấu hình multiworker training.

Audit native RGB trước resize: mean Pillow L; variance Laplacian bốn lân cận, reflect boundary. Diagnostic thresholds min side96, brightness35, blur50; inventory thêm mean brightness>220 là HIGH_BRIGHTNESS_CANDIDATE, không xác nhận glare. Hard-case useful sample có thể LOW_QUALITY và vẫn được giữ/review. Không dùng Canny/Sobel như vùng hỏng, không thêm CLAHE/blur vào CNN hoặc đổi inference threshold.

## Handoff/giới hạn

TV2 dùng reviewed candidate supported manifest, quality, lock và multi-root adapter sau review. TV3 dùng OTHER manifest/provenance và evaluation protocol riêng. Cần thống nhất source/license, label review, global specimen/group IDs, eligibility, root IDs, split val và governance final test.

Camera chưa đủ tám lớp, source/group của 22 ảnh chưa xác minh. OTHER DATA INCOMPLETE. FINAL TEST NOT YET FROZEN cho camera evaluation mới độc lập; internal test V5/V3 kế thừa giữ nguyên. Không claim accuracy/domain shift đã cải thiện.

Không sửa model/training/inference/UI/backend hoặc test legacy. Deletion TV1_DATASET_V2_REPORT.md đã stage từ trước được giữ nguyên. Không commit/push. Không tạo temp/backup/debug artifacts trong repo; fixture test dùng TemporaryDirectory ngoài repo.

Xem [CAMERA_V1.md](CAMERA_V1.md) cho schema và thu thập/review. Snapshot baseline audit cũ trong docs là bằng chứng lịch sử, không bị viết lại thành dataset mới.
