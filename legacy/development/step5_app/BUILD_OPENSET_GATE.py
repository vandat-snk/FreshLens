"""Build FreshLens supported/unsupported gate without retraining CNN weights."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from cnn_data import (
    FRUITS, PROJECT_DIR, FruitDataset, load_locked_dataset, rgb_from_bytes,
    image_transform, sha, DataError,
)
from cnn_model import choose_device, load_model
from open_set import embedding_and_probabilities, gate_features, file_sha256

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_unknown(folder: Path, internal_hashes):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files, seen = [], set()
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        data = path.read_bytes()
        digest = sha(data)
        if digest in internal_hashes or digest in seen:
            continue
        try:
            rgb_from_bytes(data)
        except Exception:
            continue
        seen.add(digest)
        files.append(path)
    return files


@torch.inference_mode()
def infer_dataset(model, device, root, rows, batch_size):
    loader = DataLoader(FruitDataset(root, rows, training=False), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    embeddings, probabilities, targets = [], [], []
    done = 0
    for images, target, _ in loader:
        images = images.to(device, non_blocking=True)
        emb, probs = embedding_and_probabilities(model, images)
        embeddings.append(emb.cpu().numpy())
        probabilities.append(probs.cpu().numpy())
        targets.append(target.numpy())
        done += len(images)
        if done % 250 < len(images) or done == len(rows):
            print(f"[EMBED] {done}/{len(rows)}", flush=True)
    return np.concatenate(embeddings), np.concatenate(probabilities), np.concatenate(targets)


@torch.inference_mode()
def infer_files(model, device, paths, batch_size):
    transform = image_transform(training=False)
    embeddings, probabilities = [], []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        tensors = [transform(rgb_from_bytes(path.read_bytes())) for path in chunk]
        images = torch.stack(tensors).to(device)
        emb, probs = embedding_and_probabilities(model, images)
        embeddings.append(emb.cpu().numpy())
        probabilities.append(probs.cpu().numpy())
        print(f"[UNKNOWN] {min(start + len(chunk), len(paths))}/{len(paths)}", flush=True)
    if not paths:
        return np.empty((0, 1280), np.float32), np.empty((0, 8), np.float32)
    return np.concatenate(embeddings), np.concatenate(probabilities)


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def build_prototypes(train_embeddings, train_targets, prototypes_per_fruit=12):
    train_embeddings = normalize(train_embeddings)
    fruit_targets = train_targets // 2
    result = []
    for fruit_idx, fruit in enumerate(FRUITS):
        selected = train_embeddings[fruit_targets == fruit_idx]
        clusters = min(prototypes_per_fruit, len(selected))
        if clusters < 2:
            raise DataError(f"Không đủ embedding để tạo prototype cho {fruit}.")
        km = MiniBatchKMeans(n_clusters=clusters, random_state=42, batch_size=256, n_init=5)
        km.fit(selected)
        centers = normalize(km.cluster_centers_)
        if clusters < prototypes_per_fruit:
            centers = np.concatenate([centers, np.repeat(centers[-1:], prototypes_per_fruit - clusters, axis=0)], axis=0)
        result.append(centers[:prototypes_per_fruit])
        print(f"[PROTO] {fruit}: {len(selected)} images -> {prototypes_per_fruit} prototypes")
    return np.stack(result).astype(np.float32)


def choose_threshold(known_probability, unknown_probability, min_known_accept=0.97):
    # Search all observed cut points. First protect known-fruit usability, then reject as many unknowns as possible.
    candidates = np.unique(np.concatenate([known_probability, unknown_probability, [0.0, 1.0]]))
    best = None
    for threshold in candidates:
        known_accept = float(np.mean(known_probability >= threshold))
        if known_accept + 1e-12 < min_known_accept:
            continue
        unknown_reject = float(np.mean(unknown_probability < threshold)) if len(unknown_probability) else 0.0
        key = (unknown_reject, known_accept, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold), known_accept, unknown_reject)
    if best is None:
        threshold = float(np.quantile(known_probability, 1.0 - min_known_accept))
        return threshold, float(np.mean(known_probability >= threshold)), float(np.mean(unknown_probability < threshold))
    return best[1], best[2], best[3]


def main():
    parser = argparse.ArgumentParser(description="Build open-set gate for FreshLens")
    parser.add_argument("--root", type=Path, default=Path(r"E:\VanDat_\XuLyAnh\FreshLens_Pro\FreshLens\dataset"))
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data" / "cnn_dataset_v3")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "best.pt")
    parser.add_argument("--unknown", type=Path, default=Path(r"E:\VanDat_\XuLyAnh\FreshLens_external_test_v2\unknown"))
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "open_set_gate.npz")
    parser.add_argument("--meta", type=Path, default=PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "open_set_gate.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--min-known-accept", type=float, default=0.97)
    args = parser.parse_args()

    try:
        if not 0.90 <= args.min_known_accept < 1.0:
            raise DataError("--min-known-accept phải từ 0.90 đến dưới 1.0")
        rows, identity = load_locked_dataset(args.data)
        train_rows = [row for row in rows if row["split"] == "train"]
        val_rows = [row for row in rows if row["split"] == "val"]
        internal_hashes = {row["sha256"] for row in rows}
        unknown_paths = collect_unknown(args.unknown, internal_hashes)
        if len(unknown_paths) < 10:
            raise DataError(f"Cần ít nhất 10 ảnh unknown để hiệu chỉnh gate; hiện có {len(unknown_paths)} tại {args.unknown}")

        device = choose_device(args.device)
        model, checkpoint = load_model(args.checkpoint, device)
        print(f"[OK] Device: {device}; checkpoint epoch={checkpoint['epoch']}")
        print(f"[DATA] train={len(train_rows)} val={len(val_rows)} unknown_calibration={len(unknown_paths)}")

        train_emb, _, train_target = infer_dataset(model, device, args.root, train_rows, args.batch)
        prototypes = build_prototypes(train_emb, train_target)

        val_emb, val_probs, _ = infer_dataset(model, device, args.root, val_rows, args.batch)
        unknown_emb, unknown_probs = infer_files(model, device, unknown_paths, args.batch)
        known_x, _ = gate_features(val_emb, val_probs, prototypes)
        unknown_x, _ = gate_features(unknown_emb, unknown_probs, prototypes)

        x = np.concatenate([known_x, unknown_x], axis=0)
        y = np.concatenate([np.ones(len(known_x)), np.zeros(len(unknown_x))])
        scaler = StandardScaler().fit(x)
        xs = scaler.transform(x)
        classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        classifier.fit(xs, y)
        known_p = classifier.predict_proba(scaler.transform(known_x))[:, 1]
        unknown_p = classifier.predict_proba(scaler.transform(unknown_x))[:, 1]
        threshold, known_accept, unknown_reject = choose_threshold(known_p, unknown_p, args.min_known_accept)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            prototypes=prototypes,
            scaler_mean=scaler.mean_.astype(np.float64),
            scaler_scale=scaler.scale_.astype(np.float64),
            coef=classifier.coef_[0].astype(np.float64),
            intercept=np.asarray([classifier.intercept_[0]], dtype=np.float64),
            decision_threshold=np.asarray([threshold], dtype=np.float64),
        )
        meta = {
            "version": "1.0",
            "purpose": "supported-vs-unsupported gate; CNN weights unchanged",
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "supported_fruits": list(FRUITS),
            "dataset_identity": identity,
            "train_embeddings": len(train_rows),
            "known_validation_images": len(val_rows),
            "unknown_calibration_images": len(unknown_paths),
            "unknown_calibration_folder": str(args.unknown),
            "min_known_accept_target": args.min_known_accept,
            "chosen_threshold": threshold,
            "known_validation_accept_rate": known_accept,
            "unknown_calibration_reject_rate": unknown_reject,
            "feature_names": ["fruit_score", "fruit_margin", "joint_max", "prototype_similarity", "prototype_gap", "fruit_certainty"],
            "important_note": "This is an open-set heuristic calibrated on known validation images and the supplied unknown folder; it is not a mathematical guarantee for every possible outside image.",
        }
        args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\n[OK] Open-set gate created")
        print(f"[OK] Known validation accepted: {known_accept:.2%}")
        print(f"[OK] Unknown calibration rejected: {unknown_reject:.2%}")
        print(f"[OK] Gate: {args.output}")
        print(f"[OK] Meta: {args.meta}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
