# Camera V1 / OTHER — collection, audit và review

## Chạy

```powershell
python scripts/dataset_v2.py camera --root data/camera_v1 --baseline data/cnn_dataset_v5 --metadata docs/dataset_v2/camera_metadata.csv --output reports/camera_v1
python scripts/dataset_v2.py other --root data/other --baseline data/cnn_dataset_v5 --metadata docs/dataset_v2/other_metadata.csv --output reports/other
```

Mỗi collection chỉ có manifest.csv + audit_report.json. Chạy lại được, nhưng không overwrite output khác schema/kind hoặc chứa file không liên quan. Root trống/OTHER chưa có vẫn xuất header/count0. Không xóa, lọc, chia split hoặc train ảnh.

Supported: apple/banana/orange/tomato × fresh/rotten. OTHER: fruit|other_fruit, non_fruit, multiple / subcategory / image. Fruit/status folder là label cần review, không phải kết luận tự động từ filename.

## Thêm ảnh và metadata

Ảnh supported đặt fruit/status/filename. Điền path tương đối image root trong metadata CSV. Header templates không chứa sample giả. Ảnh không có metadata vẫn audit; source và specimen UNKNOWN.

Cùng physical fruit qua góc/ánh sáng/session/fresh→rotten phải chung group_id toàn bộ nguồn. Không biết specimen → group_id=UNKNOWN. SHA/pixel equality chỉ là content identity; không tạo group/specimen “độc lập” từ mỗi filename. Specimen ID có evidence ở nhiều group bị báo conflict.

Source camera chỉ dùng khi nhóm tự chụp có evidence. Web/external/dataset_external phải khai báo nguồn/license; không suy camera từ thư mục hoặc EXIF.

Hard-case annotations: lighting(normal/low_light/dark/uneven/overexposed), distance(far/small_object/near), angle(tilted), background(complex), occlusion(hand_held/partial), multiple_objects(true/false), noise/blur/glare(true/false), severity(small_defect/severe_rotten). Missing → UNKNOWN. hard_case_tags chỉ map annotation rõ ràng sang normal/noisy/low_light/overexposed/glare/uneven_lighting/tilted/far/small_object/blurry/complex_background/multiple_objects/hand_held/occlusion/small_defect/severe_rotten. Không suy các tag này từ quality flags, không tạo class mới.

## Schema

- Identity: path, fruit, status, label_type, category, subcategory, source, group_id, group_evidence_status, sha256, expected_sha256, sha256_status, split.
- Quality: width, height, brightness, blur_score, quality_flags, decode_ok, decode_status, quality_status, format, original_mode, channels, exif_present, exif_orientation, exif (JSON).
- Content review: pixel_sha256, phash63, duplicate_internal, duplicate_with_baseline, duplicate_of, duplicate_type, duplicate_matches (JSON), perceptual_candidates (JSON), perceptual_status.
- Hard case: lighting, distance, angle, background, occlusion, multiple_objects, noise, blur, glare, severity, hard_case_tags.
- Provenance: source_name/source_url/license/license_url/author/attribution/download_url/retrieved_date/label_evidence/provenance; specimen_id/specimen_evidence/session_id; capture_device/capture_context.
- Review: eligible_for_training, eligible_for_evaluation, requested_split, review_status, reviewer, review_evidence, perceptual_review_status, perceptual_review_evidence.

Split luôn UNASSIGNED trong collection. Eligibility mặc định false. Người dùng bật eligibility trong **metadata input** sau review; tool ghi lại đề xuất, builder kiểm tra thêm evidence/integrity. OTHER luôn false cho training, dùng eligibility evaluation riêng.

## Duplicate/leakage/quality

SHA + native EXIF-corrected RGB pixel SHA so camera/OTHER với nội bộ và V5; flag mọi thành viên, không xóa. pHash legacy Hamming≤4 so với baseline và nội bộ là candidate human review. Duplicate references phân biệt camera:path và baseline:path (camera: ở đây chỉ collection namespace; source thực vẫn độc lập). Không gọi pHash là proof.

Manifest path duplicate/case conflict bị từ chối; metadata path trùng không được silently overwrite. Report có duplicate path/SHA/pixel, references, source và requested group/SHA/specimen leakage. UNASSIGNED không bị gọi là split; UNKNOWN groups được đếm riêng, không giả gộp thành physical specimen.

Quality reuse dataset_audit/image_io: native oriented RGB trước resize, mean L 0–255 và variance Laplacian. Flags TOO_SMALL(<96), DARK(<35), BLUR_CANDIDATE(<50), HIGH_BRIGHTNESS_CANDIDATE(mean>220), MISSING_FILE, DECODE_ERROR, SHA256_MISMATCH. Brightness cao không tự chứng minh glare; noise/defect/occlusion cần annotation. Low-quality có thể là hard-case hữu ích.

Missing path chỉ biết nếu có expected metadata row; không thể phát hiện ảnh chưa khai báo và chưa từng scan. Phép đo không làm được → NOT_AVAILABLE. Quality flags không đổi fruit/status và không đổi threshold production.

## Review để build

Supported: eligible_for_training=true + APPROVED review, reviewer/review_evidence, label_evidence, group/specimen_evidence, source/source_name; camera cần specimen/capture/session, web/external cần URL/license. pHash-only candidates cần review evidence riêng. requested_split=UNASSIGNED/train/val; không thêm test.

OTHER: eligible_for_evaluation=true và gates tương tự; vào other_manifest riêng, không có target production. Human review cần kiểm tra source/licensing và label ambiguity; code không chứng nhận license hoặc physical independence.

Dùng candidate_sources.json chọn collection explicit, rồi dry-run/build output mới theo [README](README.md). Builder rehash baseline và new samples, giữ V5/final test, union group/SHA/pixel/specimen, chặn component cross-split/test-overlap. Nhóm mới chưa đề xuất split deterministic train/val, không sinh test mới.

## Kết quả dữ liệu hiện tại

Snapshot 07/10/2026: 22 supported images, banana fresh3/rotten19; sáu tổ hợp còn lại0. Decode22 OK; duplicate SHA/pixel nội bộ và V5 0; pHash baseline candidates0; diagnostic quality flags0. Source/group UNKNOWN22; eligibilityfalse22; hard-case annotations UNKNOWN. Cần run lại khi thêm ảnh.

OTHER root chưa có ảnh: manifest rỗng. Không tải web trong task này vì yêu cầu mới tập trung code; không tạo provenance giả. Camera dataset chưa hoàn chỉnh, OTHER DATA INCOMPLETE, FINAL TEST NOT YET FROZEN cho camera test mới.

Thu development nhiều specimens thật cho tám tổ hợp, ưu tiên apple fresh/rotten, orange fresh, tomato rotten: phone/laptop, vàng/yếu/bóng, tay cầm, nghiêng, gần/xa/quả nhỏ, nền phức tạp, hỏng nhẹ/nặng. Final evaluation cần collection/protocol khóa riêng trước tuning. Không làm image augmentation để giả ảnh camera mới.
