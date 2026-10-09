"""Stage tomato original images on Cloudinary with URL metadata in MongoDB Atlas.

Default is dry-run. Uploads always remain PENDING and must never be used for
training or final testing until labels and specimen provenance are reviewed.
"""
from __future__ import annotations
import argparse
import hashlib
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def scan(root, limit):
    root = root.resolve(strict=True)
    rows = []
    seen = set()
    for folder, label in [('Fresh', 'fresh'), ('Rotten', 'rotten')]:
        path = root / folder
        if not path.is_dir():
            raise ValueError(f'Missing directory: {path}')
        files = sorted((p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in EXTS), key=str)
        for p in files[:limit or None]:
            real = p.resolve(strict=True)
            if not real.is_relative_to(root):
                raise ValueError(f'Path escapes source: {p}')
            sha = digest(real)
            if sha in seen:
                raise ValueError(f'Duplicate SHA inside source: {sha}')
            seen.add(sha)
            rows.append((real, real.relative_to(root).as_posix(), label, sha))
        print(f'[DATA] {folder}={len(files[:limit or None])}')
    return rows


def publish(rows, version):
    from dotenv import load_dotenv
    from pymongo import ASCENDING
    import cloudinary
    import cloudinary.api
    import cloudinary.uploader
    from freshlens_ai.storage import get_db
    load_dotenv(PROJECT / '.env')
    required = ('CLOUDINARY_CLOUD_NAME', 'CLOUDINARY_API_KEY', 'CLOUDINARY_API_SECRET', 'MONGODB_URI')
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        raise RuntimeError('Missing .env keys: ' + ', '.join(missing))
    cloudinary.config(cloud_name=os.environ['CLOUDINARY_CLOUD_NAME'], api_key=os.environ['CLOUDINARY_API_KEY'], api_secret=os.environ['CLOUDINARY_API_SECRET'], secure=True)
    db = get_db()
    db.client.admin.command('ping')
    collection = db['dataset_candidates']
    collection.create_index([('dataset_version', ASCENDING), ('sha256', ASCENDING)], unique=True)
    prefix = f'freshlens_candidates/{version}/'
    known = {}
    cursor = None
    while True:
        page = cloudinary.api.resources(resource_type='raw', type='upload', prefix=prefix, max_results=500, next_cursor=cursor)
        for item in page.get('resources', []):
            pub = str(item.get('public_id', ''))
            sha = pub.removeprefix(prefix).split('.', 1)[0]
            if pub.startswith(prefix) and re.fullmatch('[0-9a-f]{64}', sha):
                if sha in known:
                    raise RuntimeError(f'Cloudinary duplicate SHA: {sha}')
                known[sha] = item
        cursor = page.get('next_cursor')
        if not cursor:
            break
    uploaded = reused = 0
    for i, (path, relative, label, sha) in enumerate(rows, 1):
        query = {'dataset_version': version, 'sha256': sha}
        previous = collection.find_one(query)
        if previous and (previous.get('path') != relative or previous.get('source_label') != label):
            raise RuntimeError(f'Atlas SHA metadata conflict: {relative}')
        item = known.get(sha)
        if previous and previous.get('cloud_url') and not item:
            raise RuntimeError(f'Cloudinary missing asset referenced in Atlas: {relative}')
        if item:
            reused += 1
        else:
            if digest(path) != sha:
                raise RuntimeError(f'Source changed: {relative}')
            item = cloudinary.uploader.upload(str(path), resource_type='raw', folder=f'freshlens_candidates/{version}', public_id=sha + path.suffix.lower(), overwrite=False, unique_filename=False, use_filename=False)
            if not str(item.get('public_id', '')).startswith(prefix + sha + '.'):
                raise RuntimeError(f'Unexpected Cloudinary public_id: {relative}')
            known[sha] = item
            uploaded += 1
        url = str(item.get('secure_url', ''))
        if not url.startswith('https://'):
            raise RuntimeError(f'Missing secure URL: {relative}')
        collection.update_one(query, {'$setOnInsert': {'path': relative, 'source_dataset': 'mendeley_tomato_quality_grading', 'source_label': label, 'review_status': 'PENDING', 'label_verified': False, 'source_verified': False, 'train_approved': False, 'test_approved': False, 'created_at': datetime.now(timezone.utc)}, '$set': {'cloud_url': url, 'cloud_public_id': item['public_id'], 'cloud_resource_type': 'raw', 'synced_at': datetime.now(timezone.utc)}}, upsert=True)
        if i % 100 == 0 or i == len(rows):
            print(f'[SYNC] {i}/{len(rows)} uploaded={uploaded} reused={reused}', flush=True)
    print('[OK] Candidate image URLs stored in MongoDB Atlas dataset_candidates. Nothing was approved for training.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--version', default='tomato_quality_v4_candidates')
    p.add_argument('--limit-per-class', type=int, default=0)
    p.add_argument('--execute', action='store_true', help='Actually upload photos and write to Atlas')
    args = p.parse_args()
    if args.limit_per_class < 0 or not re.fullmatch(r'[a-z0-9_]{3,64}', args.version):
        p.error('Invalid limit/version')
    try:
        rows = scan(args.root, args.limit_per_class)
        print(f'[DATA] selected={len(rows)}')
        if args.execute:
            publish(rows, args.version)
        else:
            print('[DRY RUN] No uploads or Atlas writes; add --execute only after reviewing.')
        return 0
    except Exception as exc:
        print(f'[ERROR] {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
