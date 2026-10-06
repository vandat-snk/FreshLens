"""Build FreshLens supported/unsupported gate from the already-finished external V2 set.

This patch does NOT retrain EfficientNet-B0 and does not need the original training image folder.
It only creates open_set_gate.npz/json for the final upload/camera app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from cnn_data import FRUITS, image_transform, rgb_from_bytes, DataError
from cnn_model import choose_device, load_model
from open_set import embedding_and_probabilities, gate_features, file_sha256

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
KNOWN_FOLDERS = {
    "apple_fresh": 0, "apple_rotten": 0,
    "banana_fresh": 1, "banana_rotten": 1,
    "orange_fresh": 2, "orange_rotten": 2,
    "tomato_fresh": 3, "tomato_rotten": 3,
}


def collect_images(folder: Path):
    if not folder.is_dir():
        return []
    result, seen = [], set()
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            continue
        try:
            rgb_from_bytes(data)
        except Exception:
            continue
        seen.add(digest)
        result.append(path)
    return result


def collect_known(root: Path):
    rows = []
    for folder_name, fruit_idx in KNOWN_FOLDERS.items():
        files = collect_images(root / folder_name)
        if len(files) < 5:
            raise DataError(f"Can it nhat 5 anh trong {root / folder_name}; hien co {len(files)}")
        for path in files:
            rows.append((path, fruit_idx, folder_name))
    return rows


def deterministic_split(rows):
    """Use ~75% known images for fruit prototypes and ~25% for gate calibration."""
    proto, calib = [], []
    by_fruit = {i: [] for i in range(len(FRUITS))}
    for row in rows:
        by_fruit[row[1]].append(row)
    for fruit_idx, items in by_fruit.items():
        items = sorted(items, key=lambda x: hashlib.sha256(str(x[0]).encode("utf-8")).hexdigest())
        for i, row in enumerate(items):
            (calib if i % 4 == 0 else proto).append(row)
        # Safety for unexpectedly tiny collections.
        if sum(r[1] == fruit_idx for r in proto) < 3 or sum(r[1] == fruit_idx for r in calib) < 1:
            raise DataError(f"Khong du anh de tach prototype/calibration cho {FRUITS[fruit_idx]}")
    return proto, calib


@torch.inference_mode()
def infer_paths(model, device, paths, batch_size):
    transform = image_transform(training=False)
    embeddings, probabilities = [], []
    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        tensors = [transform(rgb_from_bytes(path.read_bytes())) for path in chunk]
        images = torch.stack(tensors).to(device)
        emb, probs = embedding_and_probabilities(model, images)
        embeddings.append(emb.cpu().numpy())
        probabilities.append(probs.cpu().numpy())
        print(f"[EMBED] {min(start + len(chunk), len(paths))}/{len(paths)}", flush=True)
    if not paths:
        return np.empty((0, 1280), np.float32), np.empty((0, 8), np.float32)
    return np.concatenate(embeddings), np.concatenate(probabilities)


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def build_prototypes(embeddings, fruit_targets, prototypes_per_fruit=4):
    embeddings = normalize(embeddings)
    fruit_targets = np.asarray(fruit_targets, dtype=np.int64)
    result = []
    for fruit_idx, fruit in enumerate(FRUITS):
        selected = embeddings[fruit_targets == fruit_idx]
        clusters = min(prototypes_per_fruit, len(selected))
        if clusters < 1:
            raise DataError(f"Khong co embedding prototype cho {fruit}")
        if clusters == 1:
            centers = normalize(selected.mean(axis=0, keepdims=True))
        else:
            km = MiniBatchKMeans(n_clusters=clusters, random_state=42, batch_size=64, n_init=10)
            km.fit(selected)
            centers = normalize(km.cluster_centers_)
        if clusters < prototypes_per_fruit:
            centers = np.concatenate([centers, np.repeat(centers[-1:], prototypes_per_fruit - clusters, axis=0)], axis=0)
        result.append(centers[:prototypes_per_fruit])
        print(f"[PROTO] {fruit}: {len(selected)} images -> {prototypes_per_fruit} prototypes")
    return np.stack(result).astype(np.float32)


def choose_threshold(known_probability, unknown_probability, min_known_accept=0.90):
    candidates = np.unique(np.concatenate([known_probability, unknown_probability, [0.0, 1.0]]))
    best = None
    for threshold in candidates:
        known_accept = float(np.mean(known_probability >= threshold))
        if known_accept + 1e-12 < min_known_accept:
            continue
        unknown_reject = float(np.mean(unknown_probability < threshold))
        # Primary: reject unknown. Secondary: keep known. Third: higher threshold.
        key = (unknown_reject, known_accept, threshold)
        if best is None or key > best[0]:
            best = (key, float(threshold), known_accept, unknown_reject)
    if best is None:
        threshold = float(np.quantile(known_probability, 1.0 - min_known_accept))
        return threshold, float(np.mean(known_probability >= threshold)), float(np.mean(unknown_probability < threshold))
    return best[1], best[2], best[3]


def main():
    parser = argparse.ArgumentParser(description="Build FreshLens open-set gate without original training images")
    parser.add_argument("--external", type=Path, default=Path(r"E:\VanDat_\XuLyAnh\FreshLens_external_test_v2"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/cnn_efficientnet_b0/best.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/cnn_efficientnet_b0/open_set_gate.npz"))
    parser.add_argument("--meta", type=Path, default=Path("models/cnn_efficientnet_b0/open_set_gate.json"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--min-known-accept", type=float, default=0.90)
    args = parser.parse_args()

    try:
        known_rows = collect_known(args.external)
        unknown_paths = collect_images(args.external / "unknown")
        if len(unknown_paths) < 10:
            raise DataError(f"Can it nhat 10 anh unknown; hien co {len(unknown_paths)} tai {args.external / 'unknown'}")
        proto_rows, calib_rows = deterministic_split(known_rows)

        device = choose_device(args.device)
        model, checkpoint = load_model(args.checkpoint, device)
        print(f"[OK] Device: {device}; checkpoint epoch={checkpoint['epoch']}")
        print(f"[DATA] known_total={len(known_rows)} prototype={len(proto_rows)} known_calibration={len(calib_rows)} unknown_calibration={len(unknown_paths)}")

        proto_paths = [row[0] for row in proto_rows]
        proto_targets = [row[1] for row in proto_rows]
        proto_emb, _ = infer_paths(model, device, proto_paths, args.batch)
        prototypes = build_prototypes(proto_emb, proto_targets)

        known_emb, known_probs = infer_paths(model, device, [row[0] for row in calib_rows], args.batch)
        unknown_emb, unknown_probs = infer_paths(model, device, unknown_paths, args.batch)
        known_x, _ = gate_features(known_emb, known_probs, prototypes)
        unknown_x, _ = gate_features(unknown_emb, unknown_probs, prototypes)

        x = np.concatenate([known_x, unknown_x], axis=0)
        y = np.concatenate([np.ones(len(known_x)), np.zeros(len(unknown_x))])
        scaler = StandardScaler().fit(x)
        classifier = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
        classifier.fit(scaler.transform(x), y)

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
            "version": "1.1-external-calibration",
            "purpose": "supported-vs-unsupported gate; CNN weights unchanged",
            "checkpoint_sha256": file_sha256(args.checkpoint),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "supported_fruits": list(FRUITS),
            "calibration_source": str(args.external),
            "known_total_images": len(known_rows),
            "prototype_images": len(proto_rows),
            "known_calibration_images": len(calib_rows),
            "unknown_calibration_images": len(unknown_paths),
            "chosen_threshold": threshold,
            "known_calibration_accept_rate": known_accept,
            "unknown_calibration_reject_rate": unknown_reject,
            # Keep old key so current app sidebar can show it without modification.
            "known_validation_accept_rate": known_accept,
            "important_note": "The external V2 set had already been evaluated before this calibration. From this point it is calibration data, not a future untouched test set.",
        }
        args.meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("\n[OK] Open-set gate created")
        print(f"[OK] Known accept: {known_accept:.1%}")
        print(f"[OK] Unknown reject: {unknown_reject:.1%}")
        print(f"[OK] Threshold: {threshold:.4f}")
        print(f"[OK] Saved: {args.output}")
        print(f"[OK] Saved: {args.meta}")
    except (DataError, OSError, ValueError) as exc:
        parser.exit(1, f"[ERROR] {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
