"""MongoDB manager cho FreshLens dataset.

Kết nối, quản lý collection và các thao tác CRUD cơ bản.

Cài đặt:
    pip install pymongo[srv] python-dotenv

Cấu hình (tạo file .env tại thư mục FreshLens/):
    MONGODB_URI=mongodb+srv://<user>:<pass>@cluster0.xxx.mongodb.net/
    MONGODB_DB=freshlens
    CLOUDINARY_CLOUD_NAME=xxx
    CLOUDINARY_API_KEY=xxx
    CLOUDINARY_API_SECRET=xxx

Lấy MongoDB URI miễn phí tại: https://www.mongodb.com/atlas
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# Load .env nếu có
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass  # python-dotenv chưa cài, dùng biến môi trường thực

try:
    from pymongo import MongoClient, ASCENDING
    from pymongo.collection import Collection
    from pymongo.database import Database
    _PYMONGO_OK = True
except ImportError:
    _PYMONGO_OK = False


# ─── Cấu hình mặc định ───────────────────────────────────────────────────────

DEFAULT_URI    = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
DEFAULT_DB     = os.getenv("MONGODB_DB",  "freshlens")
COLLECTION_IMAGES = "images"


# ─── Kết nối ─────────────────────────────────────────────────────────────────

def get_client(uri: str = DEFAULT_URI) -> "MongoClient":
    if not _PYMONGO_OK:
        raise ImportError("Chưa cài pymongo. Chạy: pip install 'pymongo[srv]'")
    return MongoClient(uri, serverSelectionTimeoutMS=5000)


def get_db(uri: str = DEFAULT_URI, db_name: str = DEFAULT_DB) -> "Database":
    return get_client(uri)[db_name]


def get_images_collection(
    uri: str = DEFAULT_URI,
    db_name: str = DEFAULT_DB,
    *,
    ensure_indexes: bool = False,
) -> "Collection":
    db = get_db(uri, db_name)
    col = db[COLLECTION_IMAGES]
    # Read-only commands must not create database indexes.
    if ensure_indexes:
        col.create_index([("sha256", ASCENDING)], unique=True, sparse=True)
        col.create_index([("fruit", ASCENDING), ("status", ASCENDING)])
        col.create_index([("split", ASCENDING)])
        col.create_index([("group_id", ASCENDING)])
    return col


def test_connection(uri: str = DEFAULT_URI) -> bool:
    """Kiem tra ket noi. Tra ve True neu thanh cong."""
    client = None
    try:
        client = get_client(uri)
        client.admin.command("ping")
        print("[OK] Ket noi MongoDB thanh cong.")
        return True
    except Exception as exc:
        print(f"[LOI] Ket noi MongoDB: {type(exc).__name__}. Kiem tra .env, quyen truy cap va mang.")
        return False
    finally:
        if client is not None:
            client.close()


# ─── CRUD ─────────────────────────────────────────────────────────────────────

def upsert_image(col: "Collection", record: dict[str, Any]) -> str:
    """Thêm hoặc cập nhật một bản ghi ảnh. Dùng sha256 làm key."""
    key = record.get("sha256") or record.get("url") or record.get("path")
    if not key:
        raise ValueError("Bản ghi phải có sha256, url hoặc path")
    filter_doc = (
        {"sha256": record["sha256"]} if record.get("sha256")
        else {"url": record["url"]} if record.get("url")
        else {"path": record["path"]}
    )
    existing = col.find_one(filter_doc, {"_id": 0}) or {}
    # Upload is not a split-editing operation. Never reset existing splits,
    # curated groups/sources or a cloud URL just because a field is empty.
    fields = {name: record[name] for name in ("path", "url", "sha256") if record.get(name)}
    for name in ("fruit", "status"):
        incoming = record.get(name) or ""
        if name in existing and existing.get(name) != incoming:
            raise ValueError(f"Cung noi dung anh nhung nhan {name} khac nhau; can ra soat nhan truoc khi upload.")
        fields[name] = incoming
    for name in ("group_id", "source"):
        if not existing.get(name) and record.get(name):
            fields[name] = record[name]
    result = col.update_one(
        filter_doc,
        {"$set": fields, "$setOnInsert": {"split": record.get("split") or ""}},
        upsert=True,
    )
    return str(result.upserted_id or filter_doc)


def get_all_records(
    col: "Collection",
    split: str | None = None,
    fruit: str | None = None,
) -> list[dict[str, Any]]:
    """Lấy tất cả bản ghi, tuỳ chọn lọc theo split hoặc fruit."""
    query: dict[str, Any] = {}
    if split:
        query["split"] = split
    if fruit:
        query["fruit"] = fruit
    return [doc for doc in col.find(query, {"_id": 0})]


def update_splits(col: "Collection", split_map: dict[str, str]) -> int:
    """Cập nhật trường split cho nhiều group_id. split_map = {group_id: split}."""
    updated = 0
    for group_id, split_value in split_map.items():
        result = col.update_many({"group_id": group_id}, {"$set": {"split": split_value}})
        updated += result.modified_count
    return updated


def summary(col: "Collection") -> dict[str, Any]:
    """Thống kê nhanh collection."""
    pipeline = [
        {"$group": {
            "_id": {"fruit": "$fruit", "status": "$status", "split": "$split"},
            "count": {"$sum": 1},
        }},
        {"$sort": {"_id.fruit": 1, "_id.status": 1}},
    ]
    rows = list(col.aggregate(pipeline))
    total = col.count_documents({})
    return {"total": total, "breakdown": rows}


def export_to_csv(col: "Collection", output_path: Path) -> Path:
    """Xuất toàn bộ collection ra CSV (định dạng manifest FreshLens)."""
    import csv
    records = get_all_records(col)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["path", "url", "fruit", "status", "source", "group_id", "sha256", "split"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow({col: rec.get(col, "") for col in columns})
    print(f"✔  Xuất {len(records)} bản ghi → {output_path}")
    return output_path
