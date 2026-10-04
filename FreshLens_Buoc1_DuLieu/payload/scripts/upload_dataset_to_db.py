"""Upload dataset từ thư mục local lên MongoDB + Cloudinary.

Script này:
1. Quét thư mục raw/ (cấu trúc FreshLens tiêu chuẩn)
2. Upload mỗi ảnh lên Cloudinary → lấy URL
3. Lưu metadata + URL vào MongoDB

Cài đặt:
    pip install pymongo[srv] cloudinary python-dotenv

Cách dùng:
    # Upload toàn bộ dataset
    python scripts/upload_dataset_to_db.py --root data/dataset

    # Upload từ 1 thư mục cụ thể
    python scripts/upload_dataset_to_db.py --root E:/FreshLens/dataset

    # Dry-run (không upload, chỉ kiểm tra)
    python scripts/upload_dataset_to_db.py --root data/dataset --dry-run

    # Bỏ qua Cloudinary, chỉ lưu path local vào MongoDB
    python scripts/upload_dataset_to_db.py --root data/dataset --no-cloud
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

# Thêm project root vào sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.db_manager import get_images_collection, test_connection, upsert_image, summary
from src.data_integrity import metadata_errors
from src.dataset import scan_raw_dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FRUIT_CLASSES    = {"apple", "banana", "orange", "tomato"}
STATUS_CLASSES   = {"fresh", "rotten"}
OTHER_CLASS      = "other"


# ─── Cloudinary ──────────────────────────────────────────────────────────────

def _upload_cloudinary(img_path: Path, public_id: str) -> str:
    """Upload ảnh lên Cloudinary, trả về URL."""
    try:
        import cloudinary
        import cloudinary.uploader
        import os
        cloudinary.config(
            cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
            api_key    = os.getenv("CLOUDINARY_API_KEY"),
            api_secret = os.getenv("CLOUDINARY_API_SECRET"),
            secure     = True,
        )
        result = cloudinary.uploader.upload(
            str(img_path),
            public_id = public_id,
            folder    = "freshlens",
            overwrite = False,
        )
        return str(result["secure_url"])
    except ImportError:
        raise ImportError("Chưa cài cloudinary. Chạy: pip install cloudinary")


# ─── Scan thư mục ────────────────────────────────────────────────────────────

def _infer_labels(path: Path, raw_root: Path) -> tuple[str, str]:
    """Suy luận nhãn fruit, status từ cấu trúc thư mục."""
    parts = [p.lower() for p in path.relative_to(raw_root).parts[:-1]]
    fruit  = next((p for p in parts if p in FRUIT_CLASSES or p == OTHER_CLASS), OTHER_CLASS)
    status = next((p for p in parts if p in STATUS_CLASSES), "")
    if fruit == OTHER_CLASS:
        status = ""
    return fruit, status


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scan_images(raw_root: Path, dataset_root: Path) -> list[dict[str, Any]]:
    """Use the same groups.csv, labels and hashes as the local data pipeline."""
    return [dict(row, url="") for row in scan_raw_dataset(raw_root, dataset_root)]


# ─── Main upload ─────────────────────────────────────────────────────────────

def upload_dataset(
    dataset_root: Path,
    use_cloud: bool = True,
    dry_run: bool = False,
    mongodb_uri: str | None = None,
    db_name: str | None = None,
) -> None:
    dataset_root = Path(dataset_root).resolve()
    raw_root     = dataset_root / "raw"

    if not raw_root.exists():
        print(f"❌ Không tìm thấy thư mục raw/: {raw_root}")
        sys.exit(1)

    # Kết nối MongoDB
    from scripts.db_manager import DEFAULT_URI, DEFAULT_DB
    uri  = mongodb_uri or DEFAULT_URI
    db_n = db_name    or DEFAULT_DB

    if not dry_run:
        if not test_connection(uri):
            sys.exit(1)
        col = get_images_collection(uri, db_n, ensure_indexes=True)

    # Quet anh
    print(f"\nQuet anh trong: {raw_root}")
    records = scan_images(raw_root, dataset_root)
    print(f"   Tim thay {len(records)} anh")
    problems = metadata_errors(records, require_split=False)
    if problems:
        raise ValueError(f"Dataset co {len(problems)} loi metadata; loi dau: {problems[0]}")

    # Thong ke nhanh
    from collections import Counter
    label_counts = Counter(f"{r['fruit']}/{r['status'] or 'none'}" for r in records)
    print("   Phan bo nhan:")
    for label, count in sorted(label_counts.items()):
        print(f"     {label:<22} {count:>5} anh")

    if dry_run:
        print("\n[DRY RUN] Khong upload hay ghi DB. Thoat.")
        return

    # Upload tung anh
    print(f"\n[UPLOAD] Dang xu ly {len(records)} anh...")
    ok = err = skip = 0

    for i, rec in enumerate(records, 1):
        img_path = dataset_root / rec["path"]
        try:
            # Tinh sha256
            sha = rec["sha256"]

            # Upload Cloudinary
            if use_cloud:
                public_id = f"{rec['fruit']}_{rec['status']}_{sha[:12]}"
                rec["url"] = _upload_cloudinary(img_path, public_id)

            # Luu vao MongoDB
            upsert_image(col, rec)
            ok += 1

        except Exception as exc:
            print(f"  [LOI] [{i}/{len(records)}] {img_path.name}: {exc}")
            err += 1
            continue

        if i % 500 == 0 or i == len(records):
            print(f"  [OK]  [{i}/{len(records)}] ok={ok} err={err}")

    # Tong ket
    print("\n" + "=" * 50)
    print(f"  [XONG] Hoan tat: {ok} anh OK, {err} loi")
    stats = summary(col)
    print(f"  [DB]  Tong trong DB: {stats['total']} ban ghi")
    if err:
        raise RuntimeError(f"Co {err} anh upload that bai. Can xu ly loi truoc khi tao manifest.")

    print("=" * 50)
    print("\nBuoc tiep theo:")
    print("   python scripts/export_manifest_from_db.py --output data/dataset/manifest.csv")
    print("   python -m src.check_dataset --root data/dataset --manifest data/dataset/manifest.csv")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload dataset từ local lên MongoDB + Cloudinary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scripts/upload_dataset_to_db.py --root data/dataset
  python scripts/upload_dataset_to_db.py --root E:/FreshLens/dataset --no-cloud
  python scripts/upload_dataset_to_db.py --root data/dataset --dry-run
""",
    )
    parser.add_argument("--root",    type=Path, default=Path("data/dataset"), help="Thư mục dataset")
    parser.add_argument("--no-cloud",action="store_true", help="Không upload Cloudinary, chỉ lưu path local")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ quét, không upload hay ghi DB")
    parser.add_argument("--uri",     type=str,  default=None, help="MongoDB URI (ghi đè .env)")
    parser.add_argument("--db",      type=str,  default=None, help="Tên database MongoDB")
    args = parser.parse_args()

    upload_dataset(
        dataset_root = args.root,
        use_cloud    = not args.no_cloud,
        dry_run      = args.dry_run,
        mongodb_uri  = args.uri,
        db_name      = args.db,
    )


if __name__ == "__main__":
    main()
