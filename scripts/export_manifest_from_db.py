"""Export a validated local dataset snapshot from MongoDB.

--stats is read-only. Export preserves existing group splits and the previous
manifest. Database splits are only saved when --write-splits is requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from scripts.db_manager import DEFAULT_DB, DEFAULT_URI, get_all_records, get_images_collection, test_connection, update_splits
from src.config import RANDOM_SEED
from src.data_integrity import SPLITS, assign_missing_splits, metadata_errors, metadata_summary
from src.dataset import load_manifest, save_manifest, sha256_file


def _valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
        return True
    except (OSError, ValueError, Image.DecompressionBombError):
        return False


def _download_image(url: str, dest_path: Path) -> bool:
    """Download with timeout, decode validation and an atomic rename."""
    if _valid_image(dest_path):
        return True
    temporary: Path | None = None
    try:
        if urlsplit(url).scheme not in ("http", "https"):
            raise ValueError("URL anh phai dung http/https")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=dest_path.parent, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            with urllib.request.urlopen(url, timeout=30) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
        if not _valid_image(temporary):
            raise ValueError("Noi dung tai ve khong phai anh doc duoc")
        temporary.replace(dest_path)
        return True
    except Exception as exc:
        print(f"[LOI] Tai anh {dest_path.name}: {type(exc).__name__}")
        return False
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_images(records: list[dict[str, Any]], dest_root: Path) -> int:
    dest_root = Path(dest_root).resolve()
    ok = 0
    for index, record in enumerate(records, 1):
        current = str(record.get("path") or "")
        current_path = dest_root / current
        expected_hash = str(record.get("sha256") or "").lower()
        if current and _valid_image(current_path):
            if not expected_hash or sha256_file(current_path) == expected_hash:
                ok += 1
                continue
        url = str(record.get("url") or "")
        if not url:
            raise ValueError(f"Anh thu {index} khong co file local hop le va cung khong co URL cloud.")
        extension = Path(urlsplit(url).path).suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}:
            extension = ".jpg"
        # Hash the entire URL so identical basenames cannot collide.
        filename = hashlib.sha256(url.encode("utf-8")).hexdigest() + extension
        destination = dest_root / "raw" / str(record["fruit"]) / str(record.get("status") or "") / filename
        if not _download_image(url, destination):
            raise RuntimeError(f"Dung export: tai anh thu {index} that bai. Manifest cu duoc giu nguyen.")
        record["path"] = destination.relative_to(dest_root).as_posix()
        # Cloudinary may re-encode images; the manifest hashes LOCAL bytes.
        record["sha256"] = sha256_file(destination)
        ok += 1
        if index % 100 == 0 or index == len(records):
            print(f"[CACHE] {index}/{len(records)} anh san sang")
    return ok


def _inherit_previous_splits(records: list[dict[str, Any]], manifest: Path) -> None:
    if not manifest.is_file():
        return
    previous: dict[str, str] = {}
    for row in load_manifest(manifest):
        group = str(row.get("group_id") or "")
        split = row.get("split") or ""
        if not group or split not in SPLITS:
            raise ValueError("Manifest cu co group/split khong hop le. Can ra soat truoc khi ghi de.")
        if group in previous and previous[group] != split:
            raise ValueError("Manifest cu bi trung group giua cac split. Can ra soat du lieu.")
        previous[group] = split
    for row in records:
        old_split = previous.get(str(row.get("group_id") or ""))
        if old_split:
            if row.get("split") and row["split"] != old_split:
                raise ValueError("Split trong MongoDB khac manifest da khoa. Can chon dung phien ban dataset.")
            row["split"] = old_split


def _verify_local_records(records: list[dict[str, Any]], root: Path) -> None:
    for index, record in enumerate(records, 1):
        path = str(record.get("path") or "")
        local = root / path
        if not path or not _valid_image(local):
            raise ValueError(f"Anh thu {index} khong doc duoc tai {local}. Kiem tra --dest hoac dung --download.")
        actual = sha256_file(local)
        expected = str(record.get("sha256") or "").lower()
        if expected and expected != actual:
            raise ValueError(f"SHA256 khong khop o anh thu {index}. File local khac metadata; can ra soat.")
        record["sha256"] = actual
    problems = metadata_errors(records)
    if problems:
        raise ValueError(f"Dung export: {len(problems)} loi du lieu; loi dau: {problems[0]}")


def export_manifest(
    output_path: Path,
    download: bool = False,
    dest_root: Path | None = None,
    mongodb_uri: str = DEFAULT_URI,
    db_name: str = DEFAULT_DB,
    seed: int = RANDOM_SEED,
    force_resplit: bool = False,
    write_splits: bool = False,
) -> Path:
    output_path = Path(output_path)
    root = Path(dest_root or output_path.parent).resolve()
    if not test_connection(mongodb_uri):
        raise RuntimeError("Khong ket noi duoc MongoDB.")
    collection = get_images_collection(mongodb_uri, db_name, ensure_indexes=False)
    records = get_all_records(collection)
    if not force_resplit:
        _inherit_previous_splits(records, output_path)
    records = assign_missing_splits(records, seed, force_resplit=force_resplit)
    if download:
        download_images(records, root)
    _verify_local_records(records, root)

    # Failed downloads/checks cannot overwrite a previously valid manifest.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output_path.parent, suffix=".csv", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        save_manifest(records, temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    if write_splits:
        updated = update_splits(collection, {str(row["group_id"]): row["split"] for row in records})
        print(f"[DB] Da luu split cho {updated} ban ghi.")
    else:
        print("[DB] Che do doc; split chi duoc luu trong manifest local.")
    print(json.dumps(metadata_summary(records), ensure_ascii=False, indent=2))
    print(f"[OK] Manifest: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Thong ke MongoDB / xuat manifest da kiem tra")
    parser.add_argument("--output", type=Path, default=Path("data/dataset/manifest.csv"))
    parser.add_argument("--dest", type=Path, default=None, help="Thu muc goc dataset local")
    parser.add_argument("--download", action="store_true", help="Tai anh cloud neu file local chua san sang")
    parser.add_argument("--force-resplit", action="store_true", help="Tao phien ban split moi; khong dung de so voi ket qua cu")
    parser.add_argument("--write-splits", action="store_true", help="Ghi split da kiem tra vao MongoDB")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--stats", action="store_true", help="Chi doc metadata; khong tai anh, chia tap hay ghi DB")
    parser.add_argument("--report", type=Path, default=None, help="Luu thong ke --stats thanh JSON")
    args = parser.parse_args()
    if args.report and not args.stats:
        parser.error("--report duoc dung kem --stats")
    if args.stats:
        if not test_connection(args.uri):
            raise SystemExit(1)
        collection = get_images_collection(args.uri, args.db, ensure_indexes=False)
        report = metadata_summary(get_all_records(collection))
        text = json.dumps(report, ensure_ascii=False, indent=2)
        print(text)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(text + "\n", encoding="utf-8")
            print(f"[OK] Da luu bao cao: {args.report}")
        return
    export_manifest(args.output, args.download, args.dest, args.uri, args.db, args.seed, args.force_resplit, args.write_splits)


if __name__ == "__main__":
    main()
