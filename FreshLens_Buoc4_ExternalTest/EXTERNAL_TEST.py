"""FreshLens Step 4: evaluate a locked CNN checkpoint on NEW real-world images.

Expected external folder names:
  apple_fresh, apple_rotten, banana_fresh, banana_rotten,
  orange_fresh, orange_rotten, tomato_fresh, tomato_rotten, unknown

The unknown folder is diagnostic only: the current model has no trained "other" class.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


STEP4_DIR = Path(__file__).resolve().parent
PROJECT_DIR = STEP4_DIR.parent
CNN_DIR = PROJECT_DIR / "FreshLens_Buoc3_CNN"
if not CNN_DIR.is_dir():
    raise SystemExit(f"[ERROR] Missing CNN code folder: {CNN_DIR}")
sys.path.insert(0, str(CNN_DIR))

from cnn_data import CLASSES, FRUITS, STATUSES, DataError, image_transform, rgb_from_bytes  # noqa: E402
from cnn_model import choose_device, decode_probabilities, load_model  # noqa: E402


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FOLDER_TO_CLASS = {
    "apple_fresh": "apple::fresh",
    "apple_rotten": "apple::rotten",
    "banana_fresh": "banana::fresh",
    "banana_rotten": "banana::rotten",
    "orange_fresh": "orange::fresh",
    "orange_rotten": "orange::rotten",
    "tomato_fresh": "tomato::fresh",
    "tomato_rotten": "tomato::rotten",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_training_hashes(data_dir: Path) -> set[str]:
    manifest = data_dir / "manifest.csv"
    if not manifest.is_file():
        raise DataError(f"Training manifest not found: {manifest}")
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "sha256" not in (reader.fieldnames or []):
            raise DataError("Training manifest has no sha256 column.")
        hashes = {(row.get("sha256") or "").strip().lower() for row in reader}
    return {value for value in hashes if len(value) == 64}


def discover_images(root: Path) -> list[tuple[Path, str | None, str]]:
    if not root.is_dir():
        raise DataError(f"External test folder not found: {root}")
    rows: list[tuple[Path, str | None, str]] = []
    for folder_name, class_name in FOLDER_TO_CLASS.items():
        folder = root / folder_name
        if not folder.is_dir():
            raise DataError(f"Missing required folder: {folder}")
        files = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise DataError(f"No supported images in: {folder}")
        rows.extend((path, class_name, folder_name) for path in files)
    unknown = root / "unknown"
    if unknown.is_dir():
        files = sorted(p for p in unknown.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        rows.extend((path, None, "unknown") for path in files)
    return rows


@torch.inference_mode()
def predict_one(model, path: Path, device: torch.device) -> dict:
    data = path.read_bytes()
    image = rgb_from_bytes(data)
    tensor = image_transform(training=False)(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probs = logits.float().softmax(dim=1).cpu().numpy()
    decoded = decode_probabilities(probs)
    fruit_idx = int(decoded["fruit"][0])
    condition_idx = int(decoded["condition"][0])
    joint_idx = fruit_idx * 2 + condition_idx
    raw_idx = int(decoded["raw_joint_argmax"][0])
    return {
        "sha256": sha256_bytes(data),
        "predicted_class": CLASSES[joint_idx],
        "predicted_fruit": FRUITS[fruit_idx],
        "predicted_condition": STATUSES[condition_idx],
        "fruit_score": float(decoded["fruit_probabilities"][0, fruit_idx]),
        "condition_score_given_fruit": float(decoded["condition_probabilities"][0, condition_idx]),
        "deployed_joint_score": float(probs[0, joint_idx]),
        "raw_argmax_class": CLASSES[raw_idx],
        "raw_argmax_score": float(probs[0, raw_idx]),
    }


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def metrics_for_known(rows: list[dict]) -> dict:
    true = [row["true_class"] for row in rows]
    pred = [row["predicted_class"] for row in rows]
    matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for a, b in zip(true, pred):
        matrix[CLASSES.index(a), CLASSES.index(b)] += 1

    per_class = {}
    f1s = []
    recalls = []
    for i, name in enumerate(CLASSES):
        tp = int(matrix[i, i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i, :].sum() - tp)
        support = int(matrix[i, :].sum())
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1s.append(f1)
        recalls.append(recall)

    fruit_correct = sum(r["true_class"].split("::")[0] == r["predicted_fruit"] for r in rows)
    cond_correct = sum(r["true_class"].split("::")[1] == r["predicted_condition"] for r in rows)
    joint_correct = sum(r["true_class"] == r["predicted_class"] for r in rows)
    return {
        "images": len(rows),
        "fruit_accuracy": safe_div(fruit_correct, len(rows)),
        "condition_accuracy": safe_div(cond_correct, len(rows)),
        "joint_accuracy": safe_div(joint_correct, len(rows)),
        "joint_macro_f1": float(np.mean(f1s)),
        "joint_balanced_accuracy": float(np.mean(recalls)),
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def unknown_summary(rows: list[dict], threshold: float) -> dict:
    if not rows:
        return {"images": 0, "note": "No unknown images supplied."}
    fruit_scores = np.array([r["fruit_score"] for r in rows], dtype=float)
    condition_scores = np.array([r["condition_score_given_fruit"] for r in rows], dtype=float)
    joint_scores = np.array([r["deployed_joint_score"] for r in rows], dtype=float)
    review = np.array([r["review_heuristic"] for r in rows], dtype=bool)
    predicted = Counter(r["predicted_class"] for r in rows)
    return {
        "images": len(rows),
        "review_threshold": threshold,
        "review_flagged": int(review.sum()),
        "review_flagged_rate": float(review.mean()),
        "mean_fruit_score": float(fruit_scores.mean()),
        "mean_condition_score_given_fruit": float(condition_scores.mean()),
        "mean_deployed_joint_score": float(joint_scores.mean()),
        "max_deployed_joint_score": float(joint_scores.max()),
        "predicted_class_counts": dict(sorted(predicted.items())),
        "warning": "This model has no trained other/unknown class. The review threshold is only a heuristic, not an OOD detector.",
    }


def plot_confusion(path: Path, matrix: list[list[int]]) -> None:
    array = np.asarray(matrix, dtype=int)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(array)
    fig.colorbar(image, ax=ax, label="Images")
    ax.set_xticks(range(len(CLASSES)), labels=CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), labels=CLASSES)
    ax.set_xlabel("Predicted fruit + condition")
    ax.set_ylabel("True fruit + condition")
    ax.set_title("External test: fruit + condition")
    maximum = max(int(array.max()), 1)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            value = int(array[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > maximum / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate FreshLens CNN on NEW external photos")
    parser.add_argument("--root", type=Path, required=True, help="Folder containing 8 labeled folders and optional unknown/")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "best.pt")
    parser.add_argument("--data", type=Path, default=PROJECT_DIR / "data" / "cnn_dataset_v3",
                        help="Locked internal dataset directory; used only to detect exact external duplicates")
    parser.add_argument("--output", type=Path, default=PROJECT_DIR / "reports" / "external_test_v1")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--review-threshold", type=float, default=0.70,
                        help="Heuristic review threshold for fruit/condition scores; not an unknown detector")
    args = parser.parse_args()

    try:
        if not 0 < args.review_threshold < 1:
            raise DataError("review-threshold must be between 0 and 1.")
        if args.output.exists():
            raise DataError(f"Output already exists: {args.output}. Preserve it and choose another --output.")
        images = discover_images(args.root)
        training_hashes = load_training_hashes(args.data)
        device = choose_device(args.device)
        model, checkpoint = load_model(args.checkpoint, device)

        args.output.mkdir(parents=True, exist_ok=False)
        results: list[dict] = []
        for index, (path, true_class, folder) in enumerate(images, 1):
            pred = predict_one(model, path, device)
            true_fruit, true_condition = (true_class.split("::") if true_class else ("", ""))
            review = pred["fruit_score"] < args.review_threshold or pred["condition_score_given_fruit"] < args.review_threshold
            row = {
                "path": str(path.resolve()),
                "folder": folder,
                "true_class": true_class or "unknown",
                "true_fruit": true_fruit,
                "true_condition": true_condition,
                **pred,
                "joint_correct": bool(true_class and pred["predicted_class"] == true_class),
                "fruit_correct": bool(true_class and pred["predicted_fruit"] == true_fruit),
                "condition_correct": bool(true_class and pred["predicted_condition"] == true_condition),
                "review_heuristic": bool(review),
                "exact_duplicate_of_internal_dataset": pred["sha256"].lower() in training_hashes,
            }
            results.append(row)
            if index % 10 == 0 or index == len(images):
                print(f"[PREDICT] {index}/{len(images)} images", flush=True)

        known = [r for r in results if r["true_class"] != "unknown"]
        unknown = [r for r in results if r["true_class"] == "unknown"]
        duplicate_count = sum(r["exact_duplicate_of_internal_dataset"] for r in results)
        metrics = metrics_for_known(known)
        report = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "checkpoint_architecture": checkpoint["architecture"],
            "external_root": str(args.root.resolve()),
            "known_metrics": metrics,
            "unknown_diagnostics": unknown_summary(unknown, args.review_threshold),
            "exact_internal_duplicate_images": int(duplicate_count),
            "external_test_is_clean_of_exact_internal_duplicates": duplicate_count == 0,
            "review_threshold": args.review_threshold,
            "scope": "One main fruit; apple/banana/orange/tomato only for accuracy metrics.",
            "important_limitations": [
                "External accuracy is meaningful only if images are genuinely new and labels are correct.",
                "Exact SHA-256 comparison detects byte-identical overlap only; it does not prove physical-specimen independence.",
                "Unknown images are diagnostic only because the model has no trained other/unknown class.",
                "Model scores are uncalibrated outputs and are not probabilities of correctness.",
            ],
        }
        json_dump(args.output / "metrics.json", report)
        write_csv(args.output / "predictions.csv", results)
        write_csv(args.output / "errors.csv", [r for r in known if not r["joint_correct"]])
        write_csv(args.output / "unknown_predictions.csv", unknown)
        plot_confusion(args.output / "confusion_matrix.png", metrics["confusion_matrix"])

        print("[OK] External known images:", metrics["images"])
        print(f"[OK] fruit_acc={metrics['fruit_accuracy']:.2%} | condition_acc={metrics['condition_accuracy']:.2%} | joint_acc={metrics['joint_accuracy']:.2%} | macro_F1={metrics['joint_macro_f1']:.4f}")
        print(f"[CHECK] exact duplicates vs internal dataset: {duplicate_count}")
        if unknown:
            u = report["unknown_diagnostics"]
            print(f"[UNKNOWN] {u['images']} images | heuristic review flagged {u['review_flagged']}/{u['images']} ({u['review_flagged_rate']:.1%})")
            print("[UNKNOWN] Diagnostic only: no trained unknown/other class.")
        print(f"[OK] Saved: {args.output}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
