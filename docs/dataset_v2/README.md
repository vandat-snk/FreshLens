# Dataset V2 — TV1: data contract và tái tạo

Dataset V3 là baseline bất biến. V4 là snapshot metadata cũ, giữ nguyên. Candidate có ảnh thật được lưu tại `data/cnn_dataset_v5`; dùng lại ảnh trong `data/raw/cnn_v3_originals`, không copy dataset.

## Mapping ảnh

Manifest V3 dùng `raw/fruit/status/file`, nhưng originals root dùng `fruit/status/file`. Audit bỏ prefix chỉ lúc đọc ảnh. 75 tên file khác manifest được resolve từ catalog SHA-match đã có. Tool kiểm tra coverage, label, group, split, source và SHA; mỗi byte ảnh vẫn được hash lại. Không đổi đường dẫn trong V3 hoặc tạo manifest trung gian.

Candidate lock lưu `image_path_mapping.strip_prefix` và `overrides`. Adapter TV1 map path trong bộ nhớ và giữ `manifest_path` gốc. Không suy ra provenance/camera/specimen độc lập từ tên file hoặc hash match.

## Commands

Chạy từ repository root với Python có NumPy/Pillow; adapter Dataset cần torch/torchvision.

```powershell
# Audit, duplicate, group leakage và quality cùng một luồng
python scripts/dataset_v2.py audit --data data/cnn_dataset_v3 --root data/raw/cnn_v3_originals --strip-prefix raw --path-map-csv data/raw/cnn_v3_originals/matched_originals.csv --fingerprints --report reports/tv1_v3_originals_audit.json --quality-csv reports/tv1_v3_originals_quality.csv --quality-summary reports/tv1_quality_summary.json --quality-summary-csv reports/tv1_quality_summary.csv --leakage-report reports/tv1_leakage_audit.json

# Review toàn bộ raw partition; không tự thêm unmatched vào dataset
python scripts/dataset_v2.py provenance --baseline data/cnn_dataset_v3 --originals-root data/raw/cnn_v3_originals --raw-root data/raw/cnn_v3 --original-quality-csv reports/tv1_v3_originals_quality.csv --matched-csv data/raw/cnn_v3_audit/matched_originals.csv --duplicates-csv data/raw/cnn_v3_audit/duplicate_originals.csv --unmatched-csv data/raw/cnn_v3_audit/unmatched_images.csv --strip-prefix raw --workers 4 --report reports/tv1_unmatched_provenance.json --csv reports/tv1_unmatched_provenance.csv

# Chỉ chạy khi output chưa tồn tại; không có overwrite mode
python scripts/dataset_v2.py build --baseline data/cnn_dataset_v3 --root data/raw/cnn_v3_originals --strip-prefix raw --path-map-csv data/raw/cnn_v3_originals/matched_originals.csv --inventory docs/dataset_v2/capture_inventory.csv --output data/cnn_dataset_v5 --require-images

python -m pytest tests/test_dataset_v2.py -q
python -m unittest discover -s legacy/development/step2_split -p test_step2.py
python -m unittest discover -s legacy/development/step2b_grouping -p test_step2b.py
```

Build từ chối thư mục đã tồn tại, thiếu ảnh, SHA mismatch, decode lỗi, label conflict và leakage group/SHA. Version sau phải dùng output mới. Audit output phải ở ngoài dataset. Exit 0: metadata/byte/decode đã kiểm chứng; 1: lỗi integrity/metadata/decode; 2: chưa kiểm chứng đủ ảnh.

## Manifest và loader

Supported: `path,fruit,status,source,group_id,sha256,split`, thêm capture và quality. Validation dùng `val`. Eight-class mapping giữ nguyên. Other tách `other_manifest.csv`: category `other_fruit|non_fruit|multiple`, subcategory mô tả cảnh. Không thêm class thứ chín.

```python
from freshlens_ai.data.dataset_v2 import load_candidate_dataset
from freshlens_ai.data.dataset import FruitDataset

rows, identity = load_candidate_dataset("data/cnn_dataset_v5")
train = FruitDataset("data/raw/cnn_v3_originals",
                     [r for r in rows if r["split"] == "train"], training=True)
validation = FruitDataset("data/raw/cnn_v3_originals",
                          [r for r in rows if r["split"] == "val"], training=False)
```

Adapter xác minh lock/hash/metadata, không thay production loader hoặc training entrypoint. TV2 phải chủ động tích hợp candidate trong experiment riêng. Test dùng `training=False` và không dùng cho tuning. `ready_for_pilot_training` chỉ nói candidate baseline-only có đủ ảnh đã kiểm chứng, không chứng minh domain shift giảm hoặc specimen độc lập. `production_loader_compatible=false`.

## Reuse và image processing

Reuse `image_io.py` để decode một frame, EXIF transpose, alpha composite trắng, RGB. Reuse production transform: `ImageOps.pad` giữ aspect ratio, BICUBIC, trắng, center, 224×224; tensor RGB CHW 0–1; ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225].

Train: flip p=0.5, affine ±15°, translate 0.04, scale 0.9–1.02, BILINEAR trắng; jitter brightness/contrast 0.12, saturation 0.08, hue 0.01. Val/test/inference không có augmentation ngẫu nhiên. Không đổi baseline augmentation, MixUp/CutMix hoặc threshold inference.

Reuse legacy `step2_core.py` UnionFind/assign_splits và phash63/HashTree. Không chạy legacy delete/resplit trên V3. Audit đo native EXIF-corrected RGB trước resize: mean Pillow L 0–255; variance Laplacian bốn lân cận, reflect boundary. Candidate flags: min side <96, brightness <35, blur score <50. Các flag có thể chồng nhau; texture/nền/resolution ảnh hưởng blur. Không tự loại ảnh hoặc đưa threshold vào inference.

Provenance phân biệt exact byte/pixel duplicate, probable augmentation (transform filename và named parent cùng label), probable new image (evidence source/license/label/independent capture), unknown, corrupt. pHash ≤4 chỉ tạo candidate review; không chứng minh augmentation hoặc cùng specimen. Tất cả unmatched hiện không tự được admit.

## Thu thập camera và other

`capture_inventory.csv` chỉ có header. Tạo inventory local cho ảnh thật; supported điền fruit/status, other điền category/subcategory. Path tương đối root, source cụ thể, provenance/capture log, license. Camera phải có device, specimen_id, specimen_evidence, session_id, lighting, background, viewpoint, distance, hard_case_tags (phân cách `;`).

Cùng quả dù khác góc/nền/buổi chụp hoặc fresh→rotten phải cùng specimen_id và group_id. Multiple scenes chia sẻ quả phải liên kết group. Không dùng mỗi file một ID để tuyên bố độc lập.

Thu development 20–40 ảnh khó là kế hoạch, chưa phải ảnh đã có. Ưu tiên apple fresh/rotten, orange fresh, tomato rotten; cần đủ tám tổ hợp. Điện thoại/laptop, ánh sáng vàng/yếu, bóng đổ, nền phức tạp, tay cầm, gần/xa, quả nhỏ, hỏng nhẹ. Protocol ban đầu cho development train/val; camera final test độc lập phải thu/khóa riêng trước tuning.

Other: grape/mango/pear/watermelon/dragon fruit/lemon/strawberry và quả khác; phone/laptop/keyboard/cup/book/shoe/bag; multiple fruits, fruit+object, multiple objects. Nhiều nguồn/nền/ánh sáng/góc/khoảng cách. Không đạt diversity chỉ bằng một loại quả. `OTHER DATA INCOMPLETE`.

## Split và bàn giao

Giữ nguyên mọi baseline row, group, SHA, split. Union group+SHA xuyên supported/other. Group mới trùng baseline kế thừa split; component nối frozen splits bị từ chối. Group mới thuần label reuse seed42 60/20/20; mixed-label components train. Ít group có thể không đủ val/test, không tạo thêm ảnh.

TV2: candidate supported manifest, quality, lock và adapter. TV3: other manifest riêng, provenance report và camera capture protocol. Cần thống nhất root/mapping, version, eight-class label index, split `val`, specimen evidence và protocol evaluation. Không merge hoặc sửa branch các thành viên.

`FINAL TEST NOT YET FROZEN`: candidate kế thừa internal test V3; chưa có final camera test mới độc lập. Group khác nhau chưa chứng minh quả vật lý khác nhau. Không tuyên bố accuracy tăng nếu chưa có experiment.

Raw, candidate data, reports local, cache, checkpoint không commit. Snapshot audit JSON trong docs lưu kết quả thực tế để review; CSV chi tiết trong reports có thể tái tạo bằng commands trên.
