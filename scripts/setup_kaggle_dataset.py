"""Setup Kaggle dataset cho FreshLens.

Script này tự động:
1. Kiểm tra / cài kaggle CLI
2. Tải dataset "Fruits Fresh and Rotten" từ Kaggle
3. Sắp xếp lại cấu trúc thư mục phù hợp với FreshLens
4. (Tuỳ chọn) Tạo manifest.csv và chạy train

Cách dùng:
    python scripts/setup_kaggle_dataset.py --dest data/dataset

Trước khi chạy, đặt kaggle.json vào:
    Windows: C:\\Users\\<tên>\\.kaggle\\kaggle.json
    Linux:   ~/.kaggle/kaggle.json

Lấy kaggle.json tại: https://www.kaggle.com/settings → API → Create New Token
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

# ─── Cấu hình dataset Kaggle ───────────────────────────────────────────────────
# Dataset "Fruits Fresh and Rotten for Classification"
# https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification
KAGGLE_DATASET   = "sriramr/fruits-fresh-and-rotten-for-classification"
KAGGLE_ZIP_NAME  = "fruits-fresh-and-rotten-for-classification.zip"

# Mapping từ tên thư mục trong Kaggle → tên nhãn FreshLens
# Kaggle dùng: freshapples, rottenapples, freshbanana, rottenbanana, ...
FOLDER_MAP: dict[str, tuple[str, str]] = {
    # Kaggle folder name  →  (fruit, status)
    "freshapples":    ("apple",  "fresh"),
    "rottenapples":   ("apple",  "rotten"),
    "freshbanana":    ("banana", "fresh"),
    "rottenbanana":   ("banana", "rotten"),
    "freshoranges":   ("orange", "fresh"),
    "rottenoranges":  ("orange", "rotten"),
    # Tomato: dataset này không có — thêm thủ công nếu cần
    # "freshtomato":  ("tomato", "fresh"),
    # "rottentomato": ("tomato", "rotten"),
}

# ─── Kiểu ảnh được giữ lại ─────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ───────────────────────────────────────────────────────────────────────────────
# Utilities
# ───────────────────────────────────────────────────────────────────────────────

def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"  ▶ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check, text=True, capture_output=False)


def _ensure_kaggle() -> None:
    """Kiểm tra kaggle CLI đã cài chưa, nếu chưa thì cài."""
    result = subprocess.run(["kaggle", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✔  kaggle CLI: {result.stdout.strip()}")
        return
    print("📦 Cài đặt kaggle CLI...")
    _run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])


def _check_kaggle_token() -> None:
    """Kiểm tra kaggle.json tồn tại."""
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    if token_path.exists():
        print(f"✔  Tìm thấy kaggle.json tại: {token_path}")
        return
    # Thử biến môi trường
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        print("✔  Dùng biến môi trường KAGGLE_USERNAME / KAGGLE_KEY")
        return
    print("\n❌ Không tìm thấy kaggle.json!")
    print("   1. Truy cập https://www.kaggle.com/settings → API → Create New Token")
    print(f"   2. Đặt file kaggle.json vào: {token_path}")
    print("   Hoặc đặt biến môi trường:")
    print("     $env:KAGGLE_USERNAME='<username>'")
    print("     $env:KAGGLE_KEY='<api_key>'")
    sys.exit(1)


def _download_dataset(dest_zip_dir: Path) -> Path:
    """Tải dataset từ Kaggle vào dest_zip_dir."""
    dest_zip_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_zip_dir / KAGGLE_ZIP_NAME
    if zip_path.exists():
        print(f"✔  ZIP đã tồn tại, bỏ qua tải: {zip_path}")
        return zip_path
    print(f"\n⬇️  Đang tải '{KAGGLE_DATASET}' từ Kaggle...")
    _run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(dest_zip_dir)])
    if not zip_path.exists():
        # Tìm file zip bất kỳ nếu tên khác
        zips = list(dest_zip_dir.glob("*.zip"))
        if zips:
            zip_path = zips[0]
        else:
            raise FileNotFoundError(f"Không tìm thấy file ZIP sau khi tải về: {dest_zip_dir}")
    print(f"✔  Đã tải: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB)")
    return zip_path


def _extract_zip(zip_path: Path, extract_to: Path) -> Path:
    """Giải nén ZIP và trả về thư mục gốc bên trong."""
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"✔  Đã giải nén trước đó tại: {extract_to}")
        return extract_to
    print(f"\n📂 Giải nén {zip_path.name} → {extract_to} ...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"✔  Giải nén xong.")
    return extract_to


def _find_source_dirs(extracted_root: Path) -> dict[str, Path]:
    """Tìm các thư mục ảnh trong ZIP đã giải nén."""
    found: dict[str, Path] = {}
    # Duyệt đệ quy, tìm tên thư mục khớp FOLDER_MAP
    for item in sorted(extracted_root.rglob("*")):
        if item.is_dir() and item.name.lower() in FOLDER_MAP:
            found[item.name.lower()] = item
    return found


def _count_images(directory: Path) -> int:
    return sum(
        1 for f in directory.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )


def _reorganize(
    source_dirs: dict[str, Path],
    raw_root: Path,
    dry_run: bool = False,
) -> dict[str, int]:
    """Copy ảnh vào cấu trúc FreshLens: raw/<fruit>/<status>/"""
    stats: dict[str, int] = {}
    for folder_name, src_dir in source_dirs.items():
        fruit, status = FOLDER_MAP[folder_name]
        dest_dir = raw_root / fruit / status
        label = f"{fruit}/{status}"
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for img_path in sorted(src_dir.rglob("*")):
            if not img_path.is_file() or img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            dest_file = dest_dir / img_path.name
            if not dry_run and not dest_file.exists():
                shutil.copy2(img_path, dest_file)
            count += 1
        stats[label] = count
        print(f"  {'[DRY]' if dry_run else '✔ '} {label:<20} {count:>5} ảnh  →  {dest_dir}")
    return stats


def _write_groups_csv(raw_root: Path, dataset_root: Path) -> Path:
    """Tạo groups.csv đơn giản: mỗi ảnh là 1 group độc lập (an toàn nhất khi chưa có thông tin)."""
    import csv
    groups_path = dataset_root / "groups.csv"
    rows: list[dict[str, str]] = []
    for img in sorted(raw_root.rglob("*")):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative = img.relative_to(dataset_root).as_posix()
        # group_id = stem (bỏ __view nếu có)
        group_id = img.stem.split("__")[0]
        rows.append({
            "relative_path": relative,
            "group_id":      group_id,
            "source":        "kaggle",
        })
    with groups_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "group_id", "source"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"✔  Đã tạo groups.csv: {groups_path} ({len(rows)} dòng)")
    return groups_path


# ───────────────────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tải dataset Kaggle và sắp xếp cho FreshLens",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  python scripts/setup_kaggle_dataset.py
  python scripts/setup_kaggle_dataset.py --dest E:/FreshLens/dataset
  python scripts/setup_kaggle_dataset.py --dest data/dataset --train
  python scripts/setup_kaggle_dataset.py --dry-run
""",
    )
    parser.add_argument(
        "--dest", type=Path, default=Path("data/dataset"),
        help="Thư mục dataset đích (mặc định: data/dataset)",
    )
    parser.add_argument(
        "--train", action="store_true",
        help="Tự động chạy train sau khi chuẩn bị dataset",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ kiểm tra, không copy ảnh hay tải file",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Bỏ qua tải Kaggle, dùng ZIP đã có trong --zip-dir",
    )
    parser.add_argument(
        "--zip-dir", type=Path, default=None,
        help="Thư mục chứa ZIP đã tải (mặc định: <dest>/downloads)",
    )
    args = parser.parse_args()

    dest: Path     = args.dest.resolve()
    raw_root: Path = dest / "raw"
    zip_dir: Path  = (args.zip_dir or dest / "downloads").resolve()

    print("=" * 60)
    print("  FreshLens — Kaggle Dataset Setup")
    print("=" * 60)
    print(f"  Thư mục đích : {dest}")
    print(f"  raw/         : {raw_root}")
    print(f"  Chế độ       : {'DRY RUN' if args.dry_run else 'THỰC THI'}")
    print("=" * 60)

    # 1. Kiểm tra kaggle CLI + token
    if not args.skip_download and not args.dry_run:
        _ensure_kaggle()
        _check_kaggle_token()

    # 2. Tải ZIP
    if not args.skip_download and not args.dry_run:
        zip_path = _download_dataset(zip_dir)
    else:
        existing_zips = list(zip_dir.glob("*.zip")) if zip_dir.exists() else []
        if existing_zips:
            zip_path = existing_zips[0]
            print(f"✔  Dùng ZIP sẵn có: {zip_path}")
        else:
            zip_path = zip_dir / KAGGLE_ZIP_NAME
            if not zip_path.exists() and not args.dry_run:
                print(f"❌ Không tìm thấy ZIP tại: {zip_dir}")
                print("   Hãy bỏ --skip-download để tải tự động.")
                sys.exit(1)

    # 3. Giải nén
    extract_dir = zip_dir / "extracted"
    if not args.dry_run:
        _extract_zip(zip_path, extract_dir)

    # 4. Tìm thư mục nguồn
    if not args.dry_run:
        source_dirs = _find_source_dirs(extract_dir)
        if not source_dirs:
            print(f"\n❌ Không tìm thấy thư mục ảnh trong: {extract_dir}")
            print("   Các thư mục được tìm kiếm:", list(FOLDER_MAP.keys()))
            sys.exit(1)
        print(f"\n📁 Tìm thấy {len(source_dirs)} thư mục ảnh:")
        for name, path in source_dirs.items():
            print(f"   {name:<20} ({_count_images(path)} ảnh)  {path}")
    else:
        # dry run: giả sử tất cả đều có
        source_dirs = {k: Path(f"[dry-run]/{k}") for k in FOLDER_MAP}

    # 5. Sắp xếp vào cấu trúc FreshLens
    print(f"\n📂 Sắp xếp ảnh vào cấu trúc FreshLens: {raw_root}")
    stats = _reorganize(source_dirs, raw_root, dry_run=args.dry_run)

    # 6. Tạo groups.csv
    if not args.dry_run:
        _write_groups_csv(raw_root, dest)

    # 7. Tổng kết
    print("\n" + "=" * 60)
    print("  Tổng kết:")
    for label, count in stats.items():
        print(f"    {label:<22}  {count:>5} ảnh")
    total = sum(stats.values())
    print(f"    {'TỔNG':<22}  {total:>5} ảnh")
    print("=" * 60)

    if not args.dry_run:
        print("\n✅ Dataset đã sẵn sàng! Chạy tiếp:")
        print(f"\n  python -m src.dataset --root \"{dest}\"")
        print(f"  python -m src.train --root \"{dest}\" --manifest \"{dest}/manifest.csv\"")
        print(f"  streamlit run src/app.py")

    # 8. Tự động train nếu yêu cầu
    if args.train and not args.dry_run:
        print("\n🚀 Đang chạy pipeline train tự động...")

        # Bước dataset
        print("\n[1/2] Tạo manifest.csv...")
        _run([sys.executable, "-m", "src.dataset", "--root", str(dest)])

        # Bước train
        manifest_path = dest / "manifest.csv"
        print("\n[2/2] Train mô hình...")
        _run([
            sys.executable, "-m", "src.train",
            "--root", str(dest),
            "--manifest", str(manifest_path),
        ])
        print("\n✅ Train hoàn tất! Mô hình đã lưu tại artifacts/freshlens_model.joblib")
        print("   Khởi động app: streamlit run src/app.py")


if __name__ == "__main__":
    main()
