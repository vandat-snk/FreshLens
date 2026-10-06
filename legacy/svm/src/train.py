"""Train FreshLens models and save a reproducible, self-describing artifact.

Default run compares a color-only linear SVM baseline with a full-feature RBF
candidate. ``--ablation`` adds color+texture and color+texture+shape so the
team can report the effect of each group on the same group-aware split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from .config import (
    ARTIFACT_ROOT,
    DATASET_ROOT,
    DEFAULT_DECISION,
    DEFAULT_FEATURES,
    DEFAULT_PROCESSING,
    FRUIT_CLASSES,
    OTHER_CLASS,
    REPORT_ROOT,
    STATUS_CLASSES,
    FeatureConfig,
    ensure_runtime_dirs,
)
from .dataset import load_manifest, resolve_record_path
from .check_dataset import check_dataset
from .features import extract_features, feature_dimension
from .image_processing import preprocess_image
from .model import (
    artifact_configs,
    estimator_classes,
    fit_classifier,
    json_safe,
    library_versions,
    new_run_id,
    predict_proba_safe,
    save_artifact,
    select_open_set_thresholds,
    select_status_threshold,
    write_json,
)


EXPERIMENTS: dict[str, tuple[FeatureConfig, str]] = {
    "baseline_color_linear": (FeatureConfig(color=True, texture=False, shape=False, hog=False), "linear"),
    "color_texture_rbf": (FeatureConfig(color=True, texture=True, shape=False, hog=False), "rbf"),
    "color_texture_shape_rbf": (FeatureConfig(color=True, texture=True, shape=True, hog=False), "rbf"),
    "candidate_full_rbf": (DEFAULT_FEATURES, "rbf"),
}


def _records_for_split(records: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [record for record in records if str(record.get("split", "")).lower() == split]


def extract_matrix(
    records: list[dict[str, Any]],
    dataset_root: Path,
    feature_config: FeatureConfig,
    processing_config=DEFAULT_PROCESSING,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract X and all labels while preserving manifest order."""

    if not records:
        raise ValueError("Không có ảnh để trích xuất đặc trưng")
    vectors: list[np.ndarray] = []
    fruits: list[str] = []
    statuses: list[str] = []
    groups: list[str] = []
    paths: list[str] = []
    for index, record in enumerate(records, start=1):
        path = resolve_record_path(record, dataset_root)
        try:
            processed = preprocess_image(path, processing_config)
            vector = extract_features(processed, feature_config)
        except Exception as exc:  # noqa: BLE001 - add the exact file to the error
            raise RuntimeError(f"Không xử lý được ảnh {path}: {type(exc).__name__}: {exc}") from exc
        vectors.append(vector)
        fruits.append(str(record.get("fruit", OTHER_CLASS)))
        statuses.append(str(record.get("status", "")))
        groups.append(str(record.get("group_id", path.stem)))
        paths.append(str(record["path"]))
        if index % 50 == 0 or index == len(records):
            print(f"  Đã trích xuất {index}/{len(records)} ảnh ({feature_config})")
    X = np.vstack(vectors).astype(np.float32)
    if X.shape[1] != feature_dimension(feature_config):
        raise AssertionError(f"Sai số chiều: {X.shape[1]} != {feature_dimension(feature_config)}")
    return X, np.asarray(fruits), np.asarray(statuses), np.asarray(groups), np.asarray(paths)


def _decision_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    decision_config,
) -> dict[str, Any]:
    classes = np.asarray(classes).astype(str)
    labels = np.asarray(labels).astype(str)
    top_names = classes[np.argmax(probabilities, axis=1)]
    top_probabilities = np.max(probabilities, axis=1)
    order = np.argsort(probabilities, axis=1)
    second_probabilities = probabilities[np.arange(len(probabilities)), order[:, -2]] if probabilities.shape[1] > 1 else np.zeros(len(probabilities))
    accepted = (
        (top_names != OTHER_CLASS)
        & (top_probabilities >= decision_config.min_fruit_probability)
        & ((top_probabilities - second_probabilities) >= decision_config.min_probability_margin)
    )
    in_scope = np.isin(labels, list(FRUIT_CLASSES))
    other = labels == OTHER_CLASS
    report_labels = list(FRUIT_CLASSES) + ([OTHER_CLASS] if OTHER_CLASS in classes else [])
    report = classification_report(labels, top_names, labels=report_labels, output_dict=True, zero_division=0)
    accepted_in = accepted & in_scope
    correct_in = accepted_in & (top_names == labels)
    false_acceptance = float((accepted & other).sum() / max(int(other.sum()), 1)) if np.any(other) else 0.0
    status = {
        "accepted_count": int(accepted.sum()),
        "coverage_in_scope": float(accepted_in.sum() / max(int(in_scope.sum()), 1)),
        "accuracy_on_accepted_in_scope": float(correct_in.sum() / max(int(accepted_in.sum()), 1)),
        "accuracy_on_all_in_scope": float((top_names[in_scope] == labels[in_scope]).mean()) if np.any(in_scope) else 0.0,
        "false_acceptance_rate_other": false_acceptance,
        "confusion_matrix_labels": report_labels,
        "confusion_matrix": confusion_matrix(labels, top_names, labels=report_labels).tolist(),
        "classification_report": report,
        "accepted_mask": accepted.tolist(),
    }
    return status


def _status_metrics(
    estimator: Any,
    X: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    labels = np.asarray(labels).astype(str)
    if estimator is None or len(labels) == 0:
        return {"available": False, "count": int(len(labels))}
    probabilities = predict_proba_safe(estimator, X)
    classes = estimator_classes(estimator).astype(str)
    top_index = np.argmax(probabilities, axis=1)
    predicted = classes[top_index]
    confidence = probabilities[np.arange(len(probabilities)), top_index]
    accepted = confidence >= threshold
    return {
        "available": True,
        "count": int(len(labels)),
        "threshold": float(threshold),
        "coverage": float(accepted.mean()) if len(accepted) else 0.0,
        "accuracy_all": float((predicted == labels).mean()) if len(labels) else 0.0,
        "macro_f1_all": float(f1_score(labels, predicted, labels=list(STATUS_CLASSES), average="macro", zero_division=0)) if len(labels) else 0.0,
        "accuracy_on_accepted": float((predicted[accepted] == labels[accepted]).mean()) if np.any(accepted) else None,
        "classification_report": classification_report(labels, predicted, labels=list(STATUS_CLASSES), output_dict=True, zero_division=0),
        "confusion_matrix_labels": list(STATUS_CLASSES),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=list(STATUS_CLASSES)).tolist(),
    }


def _fit_status_models(
    X: np.ndarray,
    fruits: np.ndarray,
    statuses: np.ndarray,
    groups: np.ndarray,
    kernel: str,
    C: float,
    gamma: str | float,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    status_mask = np.isin(fruits, list(FRUIT_CLASSES)) & np.isin(statuses, list(STATUS_CLASSES))
    if status_mask.sum() < 4 or len(np.unique(statuses[status_mask])) < 2:
        return None, {}, {}
    global_model, global_info = fit_classifier(
        X[status_mask], statuses[status_mask], groups=groups[status_mask], kernel=kernel, C=C, gamma=gamma, tune=False
    )
    per_fruit: dict[str, Any] = {}
    per_fruit_info: dict[str, Any] = {}
    for fruit in FRUIT_CLASSES:
        fruit_mask = status_mask & (fruits == fruit)
        if fruit_mask.sum() < 4 or len(np.unique(statuses[fruit_mask])) < 2:
            continue
        try:
            model, info = fit_classifier(
                X[fruit_mask], statuses[fruit_mask], groups=groups[fruit_mask], kernel=kernel, C=C, gamma=gamma, tune=False
            )
            per_fruit[fruit] = model
            per_fruit_info[fruit] = info
        except Exception as exc:  # noqa: BLE001
            per_fruit_info[fruit] = {"error": f"{type(exc).__name__}: {exc}"}
    return global_model, {"global": global_info, "per_fruit": per_fruit_info}, per_fruit


def run_training(
    dataset_root: Path = DATASET_ROOT,
    manifest_path: Path | None = None,
    output_path: Path = ARTIFACT_ROOT / "freshlens_model.joblib",
    ablation: bool = False,
) -> dict[str, Any]:
    ensure_runtime_dirs()
    dataset_root = Path(dataset_root)
    manifest_path = Path(manifest_path or dataset_root / "manifest.csv")
    dataset_check = check_dataset(dataset_root, manifest_path)
    write_json(dataset_check, REPORT_ROOT / "dataset_check_before_training.json")
    if not dataset_check["is_valid"]:
        raise ValueError(
            f"Dung train: dataset co {len(dataset_check['errors'])} loi. "
            f"Xem {REPORT_ROOT / 'dataset_check_before_training.json'}"
        )
    records = load_manifest(manifest_path)
    train_records = _records_for_split(records, "train")
    val_records = _records_for_split(records, "val")
    test_records = _records_for_split(records, "test")
    if not train_records:
        raise ValueError("Manifest chưa có split=train")
    if not val_records:
        print("CẢNH BÁO: manifest chưa có validation; ngưỡng mặc định sẽ được dùng.")
    if not test_records:
        print("CẢNH BÁO: manifest chưa có test; chưa thể công bố kết quả test độc lập.")

    experiment_names = list(EXPERIMENTS) if ablation else ["baseline_color_linear", "candidate_full_rbf"]
    all_records = train_records + val_records + test_records
    matrices: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    results: list[dict[str, Any]] = []
    experiment_models: dict[str, dict[str, Any]] = {}
    print(f"Dataset: {dataset_root} | train={len(train_records)}, val={len(val_records)}, test={len(test_records)}")

    for experiment_name in experiment_names:
        feature_config, kernel = EXPERIMENTS[experiment_name]
        print(f"\n[{experiment_name}] features={feature_config} kernel={kernel}")
        try:
            matrices[experiment_name] = extract_matrix(all_records, dataset_root, feature_config)
            X, fruits, statuses, groups, paths = matrices[experiment_name]
            train_count = len(train_records)
            val_count = len(val_records)
            X_train, y_train, g_train = X[:train_count], fruits[:train_count], groups[:train_count]
            X_val, y_val = X[train_count:train_count + val_count], fruits[train_count:train_count + val_count]
            model, fit_info = fit_classifier(X_train, y_train, groups=g_train, kernel=kernel, tune=True)
            classes = estimator_classes(model).astype(str)
            if len(X_val):
                val_probabilities = predict_proba_safe(model, X_val)
                selected_decision, threshold_info = select_open_set_thresholds(val_probabilities, y_val, classes, DEFAULT_DECISION)
                val_metrics = _decision_metrics(val_probabilities, y_val, classes, selected_decision)
            else:
                selected_decision, threshold_info = DEFAULT_DECISION, {"selection": "default_no_validation"}
                val_metrics = {}
            score = float(threshold_info.get("validation_score", val_metrics.get("accuracy_on_accepted_in_scope", 0.0)))
            spec = {
                "name": experiment_name,
                "kernel": kernel,
                "feature_config": asdict(feature_config),
                "feature_dim": int(X.shape[1]),
                "fit": fit_info,
                "validation": val_metrics,
                "threshold_selection": threshold_info,
                "selection_score": score,
            }
            results.append(spec)
            experiment_models[experiment_name] = {
                "model": model,
                "decision": selected_decision,
                "fit_info": fit_info,
                "classes": classes,
            }
            print(f"  validation selection score={score:.4f}, dim={X.shape[1]}")
        except Exception as exc:  # noqa: BLE001 - continue so a failed ablation is documented
            failure = {"name": experiment_name, "error": f"{type(exc).__name__}: {exc}"}
            results.append(failure)
            print(f"  BỎ QUA thí nghiệm: {failure['error']}")

    successful = [item for item in results if "error" not in item]
    if not successful:
        raise RuntimeError("Không có thí nghiệm nào huấn luyện thành công")
    best_spec = max(successful, key=lambda item: float(item.get("selection_score", 0.0)))
    best_name = str(best_spec["name"])
    best_exp = experiment_models[best_name]
    best_feature_config = EXPERIMENTS[best_name][0]
    best_kernel = str(best_spec["kernel"])
    best_fit = best_spec.get("fit", {})
    final_C = float(best_fit.get("C", 10.0))
    final_gamma = best_fit.get("gamma", "scale")

    best_X, best_fruits, best_statuses, best_groups, _ = matrices[best_name]
    train_val_count = len(train_records) + len(val_records)
    X_train_val = best_X[:train_val_count]
    fruits_train_val = best_fruits[:train_val_count]
    statuses_train_val = best_statuses[:train_val_count]
    groups_train_val = best_groups[:train_val_count]
    final_fruit_model, final_fruit_info = fit_classifier(
        X_train_val,
        fruits_train_val,
        groups=groups_train_val,
        kernel=best_kernel,
        C=final_C,
        gamma=final_gamma,
        tune=False,
    )

    global_status_model, status_fit_info, per_fruit_models = _fit_status_models(
        X_train_val, fruits_train_val, statuses_train_val, groups_train_val, best_kernel, final_C, final_gamma
    )

    decision_config = best_exp["decision"]
    status_threshold = DEFAULT_DECISION.min_status_probability
    status_threshold_info: dict[str, Any] = {"selection": "default_no_validation"}
    best_X_val = best_X[len(train_records):train_val_count]
    best_fruits_val = best_fruits[len(train_records):train_val_count]
    best_statuses_val = best_statuses[len(train_records):train_val_count]
    status_val_mask = np.isin(best_fruits_val, list(FRUIT_CLASSES)) & np.isin(best_statuses_val, list(STATUS_CLASSES))
    if global_status_model is not None and np.any(status_val_mask):
        status_probabilities = predict_proba_safe(global_status_model, best_X_val[status_val_mask])
        status_classes = estimator_classes(global_status_model).astype(str)
        status_threshold, status_threshold_info = select_status_threshold(
            status_probabilities, best_statuses_val[status_val_mask], status_classes, DEFAULT_DECISION.min_status_probability
        )
    decision_dict = asdict(decision_config)
    decision_dict["min_status_probability"] = float(status_threshold)

    X_val = best_X[len(train_records):train_val_count]
    y_val = best_fruits[len(train_records):train_val_count]
    X_test = best_X[train_val_count:]
    y_test = best_fruits[train_val_count:]
    validation_metrics = {}
    test_metrics = {}
    final_classes = estimator_classes(final_fruit_model).astype(str)
    if len(X_val):
        validation_metrics = _decision_metrics(predict_proba_safe(final_fruit_model, X_val), y_val, final_classes, decision_config)
    if len(X_test):
        test_metrics = _decision_metrics(predict_proba_safe(final_fruit_model, X_test), y_test, final_classes, decision_config)
    validation_status_mask = np.isin(y_val, list(FRUIT_CLASSES)) & np.isin(best_statuses_val, list(STATUS_CLASSES))
    test_status_mask = np.isin(y_test, list(FRUIT_CLASSES)) & np.isin(best_statuses[train_val_count:], list(STATUS_CLASSES))
    if global_status_model is not None and np.any(validation_status_mask):
        validation_metrics["status"] = _status_metrics(global_status_model, X_val[validation_status_mask], best_statuses_val[validation_status_mask], status_threshold)
    if global_status_model is not None and np.any(test_status_mask):
        test_statuses = best_statuses[train_val_count:][test_status_mask]
        test_metrics["status"] = _status_metrics(global_status_model, X_test[test_status_mask], test_statuses, status_threshold)

    run_id = new_run_id()
    artifact = {
        "schema_version": 1,
        "model_version": "1.0.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fruit_model": final_fruit_model,
        "global_status_model": global_status_model,
        "status_models": per_fruit_models,
        "fruit_classes": list(final_classes),
        "status_classes": list(STATUS_CLASSES),
        "other_class": OTHER_CLASS,
        "feature_config": asdict(best_feature_config),
        "processing_config": asdict(DEFAULT_PROCESSING),
        "decision": decision_dict,
        "feature_dim": int(feature_dimension(best_feature_config)),
        "selection": {
            "best_experiment": best_name,
            "experiments": results,
            "status_fit": status_fit_info,
            "status_threshold": status_threshold_info,
        },
        "evaluation": {"validation": validation_metrics, "test": test_metrics},
        "training_counts": {
            "train": len(train_records),
            "validation": len(val_records),
            "train_plus_validation_final": train_val_count,
            "test": len(test_records),
        },
        "manifest_path": str(manifest_path),
        "library_versions": library_versions(),
        "label_policy": {
            "scope": "apple, banana, orange, tomato",
            "status": "fresh/rotten chỉ có ý nghĩa theo nhãn dữ liệu, không kết luận an toàn thực phẩm",
            "open_set": "other + ngưỡng xác suất + margin; lớp other không bao phủ mọi ảnh lạ",
            "status_gate": "chỉ suy luận tình trạng sau khi fruit được ACCEPTED",
        },
        "feature_order": [name for name in ("color", "texture", "shape", "hog") if getattr(best_feature_config, name)],
    }
    output_path = save_artifact(artifact, output_path)
    report = {
        "run_id": run_id,
        "artifact": str(output_path),
        "best_experiment": best_name,
        "selection_results": results,
        "validation": validation_metrics,
        "test": test_metrics,
        "warning": "Demo dataset không đại diện ảnh thực tế; chỉ công bố test sau khi thay bằng dữ liệu có nguồn và group_id độc lập.",
    }
    report_path = REPORT_ROOT / f"training_{run_id}.json"
    write_json(report, report_path)
    write_json(report, REPORT_ROOT / "latest_metrics.json")
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Huấn luyện FreshLens với baseline/candidate và đánh giá độc lập")
    parser.add_argument("--root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ARTIFACT_ROOT / "freshlens_model.joblib")
    parser.add_argument("--ablation", action="store_true", help="Chạy thêm color+texture và color+texture+shape")
    args = parser.parse_args()
    run_training(args.root, args.manifest, args.output, args.ablation)


if __name__ == "__main__":
    main()
