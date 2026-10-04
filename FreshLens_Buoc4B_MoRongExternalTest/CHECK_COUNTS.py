from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
KNOWN_FOLDERS = [
    'apple_fresh', 'apple_rotten',
    'banana_fresh', 'banana_rotten',
    'orange_fresh', 'orange_rotten',
    'tomato_fresh', 'tomato_rotten',
]


def images(folder: Path):
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description='Kiem tra so luong anh External Test V2')
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--min-known', type=int, default=10)
    ap.add_argument('--min-unknown', type=int, default=20)
    args = ap.parse_args()

    root = args.root
    if not root.is_dir():
        raise SystemExit(f'[ERROR] Khong tim thay: {root}')

    ok = True
    all_paths = []
    print('[COUNT]')
    for name in KNOWN_FOLDERS:
        paths = images(root / name)
        all_paths.extend(paths)
        mark = 'OK' if len(paths) >= args.min_known else 'THIEU'
        print(f'  {name:16s}: {len(paths):3d}  [{mark}]')
        if len(paths) < args.min_known:
            ok = False

    unknown = images(root / 'unknown')
    all_paths.extend(unknown)
    mark = 'OK' if len(unknown) >= args.min_unknown else 'THIEU'
    print(f'  {"unknown":16s}: {len(unknown):3d}  [{mark}]')
    if len(unknown) < args.min_unknown:
        ok = False

    ambiguous = images(root / 'ambiguous_review')
    print(f'  {"ambiguous_review":16s}: {len(ambiguous):3d}  [KHONG TINH ACCURACY]')

    print('\n[CHECK] Trung byte ben trong External V2...')
    by_hash = {}
    for i, path in enumerate(all_paths, 1):
        digest = sha256(path)
        by_hash.setdefault(digest, []).append(path)
        if i % 25 == 0 or i == len(all_paths):
            print(f'  {i}/{len(all_paths)}')

    duplicate_groups = [v for v in by_hash.values() if len(v) > 1]
    if duplicate_groups:
        ok = False
        print(f'[WARN] Co {len(duplicate_groups)} nhom anh trung byte trong External V2:')
        for group in duplicate_groups[:10]:
            print('  ---')
            for p in group:
                print('   ', p)
        if len(duplicate_groups) > 10:
            print(f'  ... con {len(duplicate_groups)-10} nhom')
    else:
        print('[OK] Khong co anh trung byte trong External V2.')

    print('\n[RESULT]')
    if ok:
        print('[OK] Du dieu kien chay External Test V2.')
        return 0
    print('[ERROR] Chua du dieu kien. Bo sung/loai anh theo thong bao tren.')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
