"""Independent evaluation for fruit, open-set and freshness decisions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .config import DATASET_ROOT, FRUIT_CLASSES, OTHER_CLASS, REPORT_ROOT, STATUS_CLASSES
from .dataset import load_manifest, resolve_record_path
from .model import json_safe, write_json
from .predict import FreshLensPredictor


def evaluate_records(
    predictor: FreshLensPredictor,
    records: list[dict[str, Any]],
    dataset_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        path = resolve_record_path(record, dataset_root)
        try:
            result = predictor.predict(path)
            top = result.top_fruit_probabilities[0] if result.top_fruit_probabilities else {}
            raw_fruit = top.get("class")
            raw_confidence = top.get("probability")
            rows.append({
                "path": record.get("path", ""),
                "source": record.get("source", ""),
                "group_id": record.get("group_id", ""),
                "true_fruit": record.get("fruit", ""),
                "true_status": record.get("status", ""),
                "state": result.state,
                "predicted_fruit_raw": raw_fruit or "",
                "predicted_fruit": result.fruit or "",
                "fruit_confidence": raw_confidence if raw_confidence is not None else "",
                "accepted": result.accepted,
                "predicted_status": result.status or "",
                "status_confidence": result.status_confidence if result.status_confidence is not None else "",
                "reason": result.reason,
                "run_id": result.model_run_id or "",
                "image_sha256": result.image_sha256 or "",
            })
        except Exception as exc:  # noqa: BLE001
            rows.append({
                "path": record.get("path", ""),
                "source": record.get("source", ""),
                "group_id": record.get("group_id", ""),
                "true_fruit": record.get("fruit", ""),
                "true_status": record.get("status", ""),
                "state": "INPUT_ERROR",
                "predicted_fruit_raw": "",
                "predicted_fruit": "",
                "fruit_confidence": "",
                "accepted": False,
                "predicted_status": "",
                "status_confidence": "",
                "reason": f"{type(exc).__name__}: {exc}",
                "run_id": predictor.model_run_id,
                "image_sha256": "",
            })

    true_fruit = np.asarray([str(row["true_fruit"]) for row in rows])
    raw_pred = np.asarray([str(row["predicted_fruit_raw"]) for row in rows])
    accepted = np.asarray([bool(row["accepted"]) for row in rows])
    in_scope = np.isin(true_fruit, list(FRUIT_CLASSES))
    other = true_fruit == OTHER_CLASS
    report_labels = list(FRUIT_CLASSES) + [OTHER_CLASS]
    fruit_report = classification_report(true_fruit, raw_pred, labels=report_labels, output_dict=True, zero_division=0) if rows else {}
    fruit_metrics = {
        "count": len(rows),
        "in_scope_count": int(in_scope.sum()),
        "other_count": int(other.sum()),
        "accepted_count": int(accepted.sum()),
        "coverage_in_scope": float((accepted & in_scope).sum() / max(int(in_scope.sum()), 1)),
        "accuracy_on_all_in_scope": float((raw_pred[in_scope] == true_fruit[in_scope]).mean()) if np.any(in_scope) else 0.0,
        "accuracy_on_accepted_in_scope": float((raw_pred[accepted & in_scope] == true_fruit[accepted & in_scope]).mean()) if np.any(accepted & in_scope) else None,
        "false_acceptance_rate_other": float((accepted & other).sum() / max(int(other.sum()), 1)) if np.any(other) else None,
        "confusion_matrix_labels": report_labels,
        "confusion_matrix": confusion_matrix(true_fruit, raw_pred, labels=report_labels).tolist() if rows else [],
        "classification_report": fruit_report,
        "state_counts": dict(Counter(str(row["state"]) for row in rows)),
    }

    true_status = np.asarray([str(row["true_status"]) for row in rows])
    pred_status = np.asarray([str(row["predicted_status"]) for row in rows])
    status_scope = in_scope & np.isin(true_status, list(STATUS_CLASSES))
    status_predicted = status_scope & (pred_status != "")
    status_metrics = {
        "count_in_scope": int(status_scope.sum()),
        "count_with_status_prediction": int(status_predicted.sum()),
        "coverage_after_fruit_gate": float(status_predicted.sum() / max(int(status_scope.sum()), 1)),
        "accuracy_on_predicted_status": float((pred_status[status_predicted] == true_status[status_predicted]).mean()) if np.any(status_predicted) else None,
        "macro_f1_on_predicted_status": float(f1_score(true_status[status_predicted], pred_status[status_predicted], labels=list(STATUS_CLASSES), average="macro", zero_division=0)) if np.any(status_predicted) else None,
        "confusion_matrix_labels": list(STATUS_CLASSES),
        "confusion_matrix": confusion_matrix(true_status[status_scope], pred_status[status_scope], labels=list(STATUS_CLASSES)).tolist() if np.any(status_scope) else [],
        "classification_report": classification_report(true_status[status_predicted], pred_status[status_predicted], labels=list(STATUS_CLASSES), output_dict=True, zero_division=0) if np.any(status_predicted) else {},
    }

    joint_correct = (
        in_scope
        & accepted
        & (raw_pred == true_fruit)
        & (pred_status == true_status)
    )
    by_source: defaultdict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "accepted": 0})
    for row in rows:
        source = str(row.get("source") or "unspecified")
        by_source[source]["count"] += 1
        by_source[source]["accepted"] += int(bool(row["accepted"]))

    report = {
        "run_id": predictor.model_run_id,
        "model_version": predictor.model_version,
        "fruit": fruit_metrics,
        "status": status_metrics,
        "end_to_end": {
            "correct_fruit_and_status": int(joint_correct.sum()),
            "accuracy_on_all_in_scope": float(joint_correct[in_scope].mean()) if np.any(in_scope) else 0.0,
            "accuracy_on_accepted_in_scope": float(joint_correct[accepted & in_scope].mean()) if np.any(accepted & in_scope) else None,
        },
        "outside_scope_by_source": dict(by_source),
        "limitations": [
            "Tỷ lệ coverage và false acceptance phải đọc cùng accuracy; không chỉ công bố accuracy trên ảnh được nhận.",
            "Lớp other hữu hạn, không thể đại diện mọi ảnh ngoài phạm vi.",
            "fresh/rotten là nhãn dấu hiệu bề ngoài của dataset, không kết luận an toàn thực phẩm.",
        ],
    }
    return json_safe(report), rows


def save_predictions(rows: list[dict[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "path", "source", "group_id", "true_fruit", "true_status", "state",
        "predicted_fruit_raw", "predicted_fruit", "fruit_confidence", "accepted",
        "predicted_status", "status_confidence", "reason", "run_id", "image_sha256",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Đánh giá độc lập FreshLens trên val/test của manifest")
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    manifest_path = args.manifest or (args.root / "manifest.csv")
    records = [row for row in load_manifest(manifest_path) if row.get("split") == args.split]
    if not records:
        raise SystemExit(f"Manifest không có split={args.split}")
    model_path = args.model or (Path(__file__).resolve().parents[1] / "artifacts" / "freshlens_model.joblib")
    predictor = FreshLensPredictor(model_path)
    report, rows = evaluate_records(predictor, records, args.root)
    output = args.output or REPORT_ROOT / f"evaluation_{predictor.model_run_id}_{args.split}.json"
    write_json(report, output)
    csv_path = output.with_suffix(".csv")
    save_predictions(rows, csv_path)
    print(json.dumps({"report": str(output), "predictions": str(csv_path), "metrics": report}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

