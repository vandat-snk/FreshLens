"""Train nhanh FreshLens với subset dataset.

Thay vì chạy 2 experiments trên toàn bộ 15k+ anh,
script nay:
  - Chi chay 1 experiment: candidate_full_rbf (tot nhat)
  - Lay mau can bang (balanced sample) de giu phan bo nhan deu
  - Hoan thanh trong ~15-30 phut thay vi 5+ gio

Cach dung:
  python scripts/fast_train.py
  python scripts/fast_train.py --max-per-class 800
  python scripts/fast_train.py --manifest path/to/manifest.csv --output artifacts/model.joblib
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    ARTIFACT_ROOT,
    DATASET_ROOT,
    DEFAULT_DECISION,
    DEFAULT_FEATURES,
    FeatureConfig,
    ensure_runtime_dirs,
)

MANIFEST_DEFAULT = Path("E:/VanDat_/XuLyAnh/FreshLens_Pro/FreshLens/dataset/manifest.csv")
OUTPUT_DEFAULT   = ARTIFACT_ROOT / "freshlens_model.joblib"
MAX_PER_CLASS    = 1000   # toi da anh moi nhan (fruit/status)
RANDOM_SEED      = 42


def load_and_sample(manifest_path: Path, max_per_class: int, seed: int) -> Path:
    """Doc manifest, lay mau can bang, ghi ra manifest_sampled.csv."""
    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(seed)

    # Nhom theo (split, fruit, status)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        key = f"{row.get('split','train')}::{row.get('fruit','other')}::{row.get('status') or 'none'}"
        buckets[key].append(row)

    sampled = []
    for key, bucket_rows in sorted(buckets.items()):
        rng.shuffle(bucket_rows)
        taken = bucket_rows[:max_per_class]
        sampled.extend(taken)
        split, fruit, status = key.split("::")
        print(f"  [{split}] {fruit}/{status}: {len(taken)}/{len(bucket_rows)} anh")

    # Ghi ra file tam
    out_path = manifest_path.parent / "manifest_sampled.csv"
    columns  = ["path", "fruit", "status", "source", "group_id", "sha256", "split"]
    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sampled)

    train_n = sum(1 for r in sampled if r.get("split") == "train")
    val_n   = sum(1 for r in sampled if r.get("split") == "val")
    test_n  = sum(1 for r in sampled if r.get("split") == "test")
    print(f"\nTong mau: {len(sampled)} anh  (train={train_n}, val={val_n}, test={test_n})")
    print(f"Manifest mau: {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train nhanh FreshLens voi subset dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--manifest",      type=Path, default=MANIFEST_DEFAULT,
                        help="Duong dan manifest.csv goc")
    parser.add_argument("--root",          type=Path,
                        default=Path("E:/VanDat_/XuLyAnh/FreshLens_Pro/FreshLens/dataset"),
                        help="Thu muc dataset goc")
    parser.add_argument("--output",        type=Path, default=OUTPUT_DEFAULT,
                        help="Duong dan luu model (.joblib)")
    parser.add_argument("--max-per-class", type=int,  default=MAX_PER_CLASS,
                        help=f"So anh toi da moi nhan (mac dinh: {MAX_PER_CLASS})")
    parser.add_argument("--seed",          type=int,  default=RANDOM_SEED)
    parser.add_argument("--full",          action="store_true",
                        help="Dung toan bo dataset (khong lay mau, rat cham)")
    args = parser.parse_args()

    ensure_runtime_dirs()

    print("=" * 55)
    print("  FreshLens Fast Train")
    print("=" * 55)
    print(f"  Manifest : {args.manifest}")
    print(f"  Root     : {args.root}")
    print(f"  Output   : {args.output}")
    if not args.full:
        print(f"  Max/nhan : {args.max_per_class} anh")
    print("=" * 55)

    # Lay mau hoac dung toan bo
    if args.full:
        manifest_to_use = args.manifest
        print("[INFO] Che do FULL - dung toan bo dataset")
    else:
        print(f"\n[1/3] Lay mau can bang ({args.max_per_class} anh/nhan)...")
        manifest_to_use = load_and_sample(args.manifest, args.max_per_class, args.seed)

    # Import va chay train truc tiep voi 1 experiment
    print("\n[2/3] Bat dau train (chi experiment: candidate_full_rbf)...")

    # Ghi de EXPERIMENTS de chi chay 1 experiment
    import src.train as train_module
    train_module.EXPERIMENTS = {
        "candidate_full_rbf": (DEFAULT_FEATURES, "rbf"),
    }

    from src.train import run_training
    artifact = run_training(
        dataset_root  = args.root,
        manifest_path = manifest_to_use,
        output_path   = args.output,
        ablation      = True,   # dung list(EXPERIMENTS) thay vi hardcode 2 experiments
    )

    print("\n[3/3] Ket qua:")
    print(f"  Model luu tai: {args.output}")
    best = artifact.get("selection", {}).get("best_experiment", "candidate_full_rbf")
    print(f"  Best experiment: {best}")
    ev = artifact.get("evaluation", {})
    val = ev.get("validation", {})
    test = ev.get("test", {})
    if val:
        print(f"  Validation accuracy : {val.get('accuracy_on_accepted_in_scope', 'N/A'):.4f}")
        print(f"  Validation coverage : {val.get('coverage_in_scope', 'N/A'):.4f}")
    if test:
        print(f"  Test accuracy       : {test.get('accuracy_on_accepted_in_scope', 'N/A'):.4f}")
        print(f"  Test coverage       : {test.get('coverage_in_scope', 'N/A'):.4f}")
    print("\n[OK] Chay app: streamlit run src/app.py")


if __name__ == "__main__":
    main()
