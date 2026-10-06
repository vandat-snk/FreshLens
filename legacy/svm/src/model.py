"""Model utilities, open-set decision policy and artifact compatibility."""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import (
    DEFAULT_DECISION,
    DEFAULT_FEATURES,
    DEFAULT_PROCESSING,
    FRUIT_CLASSES,
    FRUIT_CLASSES_WITH_OTHER,
    FRUIT_LABELS_VI,
    OTHER_CLASS,
    DecisionConfig,
    FeatureConfig,
    ProcessingConfig,
    decision_config_from_dict,
    feature_config_from_dict,
    processing_config_from_dict,
)
from .features import feature_dimension

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover - for older scikit-learn versions
    StratifiedGroupKFold = None  # type: ignore[misc,assignment]


STATE_LABELS_VI = {
    "ACCEPTED": "Đã chấp nhận",
    "LOW_CONFIDENCE": "Chưa đủ tin cậy",
    "OUT_OF_SCOPE": "Ngoài phạm vi hỗ trợ",
    "MODEL_MISSING": "Chưa có mô hình",
    "INPUT_ERROR": "Tệp ảnh lỗi",
}


def quality_gate_reason(quality: dict[str, Any]) -> str | None:
    """Reject inputs that contain no usable visual evidence before SVM output.

    The gate is intentionally conservative and small: it catches blank,
    nearly uniform and unusably tiny images without pretending that blur or
    brightness alone identifies freshness.
    """

    if not bool(quality.get("is_usable", False)):
        return "ảnh quá nhỏ hoặc chứa giá trị không hợp lệ"
    if bool(quality.get("is_very_dark", False)):
        return "ảnh quá tối, gần như không có thông tin màu/biên"
    if bool(quality.get("is_very_bright", False)):
        return "ảnh quá sáng, gần như bị cháy sáng"
    blur_score = float(quality.get("blur_score_laplacian_variance", 0.0))
    edge_strength = float(quality.get("edge_strength_mean", 0.0))
    if blur_score < 1.0 and edge_strength < 0.5:
        return "ảnh gần như đồng màu hoặc mất chi tiết"
    return None


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def library_versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }


def make_classifier(
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: str | float = "scale",
    probability: bool = True,
) -> Pipeline:
    if kernel not in {"linear", "rbf"}:
        raise ValueError("FreshLens chỉ hỗ trợ SVM linear hoặc RBF")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(
            kernel=kernel,
            C=float(C),
            gamma=gamma,
            probability=probability,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def _safe_cv(y: np.ndarray, groups: np.ndarray | None, requested: int = 3):
    labels, counts = np.unique(y, return_counts=True)
    if len(labels) < 2:
        return None
    if groups is not None and StratifiedGroupKFold is not None:
        group_counts = [len(np.unique(groups[y == label])) for label in labels]
        n_splits = min(requested, min(group_counts))
        if n_splits >= 2:
            return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    n_splits = min(requested, int(counts.min()))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if n_splits >= 2 else None


def fit_classifier(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: str | float = "scale",
    tune: bool = True,
) -> tuple[Pipeline, dict[str, Any]]:
    """Fit a scaler+SVM pipeline; tune only on train folds when possible."""

    if X.ndim != 2 or len(X) != len(y):
        raise ValueError("X/y không cùng số mẫu")
    if len(np.unique(y)) < 2:
        raise ValueError("Mô hình cần ít nhất hai lớp")
    cv = _safe_cv(y, groups)
    best_params = {"kernel": kernel, "C": C, "gamma": gamma, "tuned": False}
    estimator = make_classifier(kernel, C, gamma, probability=True)
    if tune and cv is not None:
        grid = {"svc__C": [1.0, 10.0], "svc__gamma": ["scale", 0.01]} if kernel == "rbf" else {"svc__C": [0.1, 1.0, 10.0]}
        search = GridSearchCV(
            estimator,
            param_grid=grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        try:
            if groups is not None:
                search.fit(X, y, groups=groups)
            else:
                search.fit(X, y)
            estimator = search.best_estimator_
            params = search.best_params_
            best_params = {
                "kernel": estimator.named_steps["svc"].kernel,
                "C": float(estimator.named_steps["svc"].C),
                "gamma": estimator.named_steps["svc"].gamma,
                "tuned": True,
                "cv_best_f1_macro": float(search.best_score_),
            }
            return estimator, best_params
        except Exception as exc:  # noqa: BLE001 - fallback is recorded in metadata
            best_params["tuning_error"] = f"{type(exc).__name__}: {exc}"
    try:
        estimator.fit(X, y)
    except Exception:
        # SVC probability calibration can be impossible for extremely small
        # classes. A deterministic decision-function model is still usable;
        # predict_proba_safe converts it to a normalized distribution.
        estimator = make_classifier(kernel, C, gamma, probability=False)
        estimator.fit(X, y)
        best_params["probability_mode"] = "decision_function_softmax"
    return estimator, best_params


def estimator_classes(estimator: Any) -> np.ndarray:
    model = estimator.named_steps.get("svc", estimator) if hasattr(estimator, "named_steps") else estimator
    classes = getattr(model, "classes_", None)
    if classes is None:
        raise ValueError("Mô hình không có classes_ — artifact không tương thích")
    return np.asarray(classes)


def predict_proba_safe(estimator: Any, X: np.ndarray) -> np.ndarray:
    """Return rows that sum to one, even when SVC was fit without probability."""

    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(X), dtype=np.float64)
        sums = probabilities.sum(axis=1, keepdims=True)
        return probabilities / np.maximum(sums, 1e-12)
    scores = np.asarray(estimator.decision_function(X), dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores -= scores.max(axis=1, keepdims=True)
    probabilities = np.exp(np.clip(scores, -50, 50))
    return probabilities / np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)


def _top_two(probabilities: np.ndarray) -> tuple[int, int, float, float]:
    order = np.argsort(probabilities)[::-1]
    top = int(order[0])
    second = int(order[1]) if len(order) > 1 else top
    top_prob = float(probabilities[top])
    second_prob = float(probabilities[second]) if len(order) > 1 else 0.0
    return top, second, top_prob, second_prob


def decide_fruit(
    probabilities: np.ndarray,
    classes: Iterable[str],
    decision: DecisionConfig = DEFAULT_DECISION,
) -> dict[str, Any]:
    """Apply the one shared open-set policy used by metrics, CLI and UI."""

    probabilities = np.asarray(probabilities, dtype=float)
    classes = [str(value) for value in classes]
    top_index, _, top_prob, second_prob = _top_two(probabilities)
    predicted = classes[top_index]
    margin = top_prob - second_prob
    reasons: list[str] = []
    if decision.reject_other_class and predicted == OTHER_CLASS:
        state = "OUT_OF_SCOPE"
        reasons.append("Lớp other đứng đầu")
    elif top_prob < decision.min_fruit_probability:
        state = "LOW_CONFIDENCE"
        reasons.append(f"xác suất cao nhất {top_prob:.1%} thấp hơn ngưỡng {decision.min_fruit_probability:.1%}")
    elif margin < decision.min_probability_margin:
        state = "LOW_CONFIDENCE"
        reasons.append(f"khoảng cách hai lớp đầu {margin:.1%} thấp hơn ngưỡng {decision.min_probability_margin:.1%}")
    else:
        state = "ACCEPTED"
        reasons.append("đạt điều kiện lớp, xác suất và khoảng cách")
    return {
        "state": state,
        "accepted": state == "ACCEPTED",
        "fruit": predicted if state == "ACCEPTED" else None,
        "raw_fruit": predicted,
        "confidence": top_prob,
        "margin": margin,
        "reason": "; ".join(reasons),
    }


def select_open_set_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: Iterable[str],
    initial: DecisionConfig = DEFAULT_DECISION,
) -> tuple[DecisionConfig, dict[str, Any]]:
    """Choose probability/margin thresholds using validation only."""

    classes = [str(value) for value in classes]
    labels = np.asarray(labels).astype(str)
    in_scope = np.isin(labels, list(FRUIT_CLASSES))
    out_scope = labels == OTHER_CLASS
    if not np.any(in_scope) and not np.any(out_scope):
        return initial, {"selection": "default_no_validation_labels"}
    best: tuple[float, DecisionConfig, dict[str, Any]] | None = None
    # Validation may raise the thresholds, but it must not make the deployed
    # policy less conservative than the documented baseline.
    for probability_threshold in np.arange(initial.min_fruit_probability, 0.91, 0.05):
        for margin_threshold in np.arange(initial.min_probability_margin, 0.31, 0.05):
            accepted = []
            correct = []
            top_names = []
            for row in probabilities:
                decision = decide_fruit(row, classes, DecisionConfig(float(probability_threshold), float(margin_threshold), initial.min_status_probability, initial.reject_other_class))
                accepted.append(bool(decision["accepted"]))
                top_names.append(decision["raw_fruit"])
            accepted_array = np.asarray(accepted)
            top_array = np.asarray(top_names)
            accepted_in = accepted_array & in_scope
            correct_in = accepted_in & (top_array == labels)
            coverage = float(accepted_in.sum() / max(int(in_scope.sum()), 1))
            accepted_accuracy = float(correct_in.sum() / max(int(accepted_in.sum()), 1))
            false_acceptance = float((accepted_array & out_scope).sum() / max(int(out_scope.sum()), 1)) if np.any(out_scope) else 0.0
            # Reward useful coverage and accepted accuracy while penalising an
            # outside image being accepted. The weights are fixed and logged.
            score = 0.50 * accepted_accuracy + 0.30 * coverage + 0.20 * (1.0 - false_acceptance)
            detail = {
                "validation_score": score,
                "coverage": coverage,
                "accepted_accuracy": accepted_accuracy,
                "false_acceptance_rate": false_acceptance,
                "min_fruit_probability": float(probability_threshold),
                "min_probability_margin": float(margin_threshold),
            }
            candidate = DecisionConfig(float(probability_threshold), float(margin_threshold), initial.min_status_probability, initial.reject_other_class)
            if best is None or score > best[0]:
                best = (score, candidate, detail)
    if best is None:
        return initial, {"selection": "default_empty_validation"}
    return best[1], best[2]


def select_status_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    classes: Iterable[str],
    initial: float = 0.55,
) -> tuple[float, dict[str, Any]]:
    """Choose a status-confidence threshold on validation data only."""

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels).astype(str)
    if probabilities.ndim != 2 or len(probabilities) != len(labels):
        return initial, {"selection": "default_invalid_validation"}
    classes = [str(value) for value in classes]
    top_indices = np.argmax(probabilities, axis=1)
    top_confidence = probabilities[np.arange(len(probabilities)), top_indices]
    top_names = np.asarray([classes[index] for index in top_indices])
    best: tuple[float, float, dict[str, Any]] | None = None
    for threshold in np.arange(max(initial, 0.50), 0.81, 0.025):
        accepted = top_confidence >= threshold
        accuracy = float((top_names[accepted] == labels[accepted]).mean()) if np.any(accepted) else 0.0
        coverage = float(accepted.mean()) if len(accepted) else 0.0
        score = 0.75 * accuracy * coverage + 0.25 * coverage
        detail = {"threshold": float(threshold), "accepted_accuracy": accuracy, "coverage": coverage, "score": score}
        if best is None or score > best[0]:
            best = (score, float(threshold), detail)
    return (best[1], best[2]) if best else (initial, {"selection": "default_empty_validation"})


def artifact_metadata_compatible(artifact: dict[str, Any]) -> tuple[bool, str]:
    required = {"schema_version", "fruit_model", "feature_config", "processing_config", "decision", "feature_dim", "run_id"}
    missing = sorted(required - set(artifact))
    if missing:
        return False, "Thiếu trường artifact: " + ", ".join(missing)
    try:
        features = feature_config_from_dict(artifact.get("feature_config"))
        processing = processing_config_from_dict(artifact.get("processing_config"))
        expected = feature_dimension(features)
        actual = int(artifact["feature_dim"])
    except Exception as exc:  # noqa: BLE001
        return False, f"Không đọc được cấu hình đặc trưng: {exc}"
    if expected != actual:
        return False, f"Sai số chiều đặc trưng: artifact={actual}, code={expected}"
    if processing.image_size != 224:
        return False, f"Artifact dùng image_size={processing.image_size}; FreshLens yêu cầu 224"
    return True, "ok"


def save_artifact(artifact: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    return path


def load_artifact(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Chưa có mô hình: {path}")
    artifact = joblib.load(path)
    if not isinstance(artifact, dict):
        raise ValueError("Artifact không phải dictionary")
    compatible, message = artifact_metadata_compatible(artifact)
    if not compatible:
        raise ValueError(message)
    return artifact


def artifact_configs(artifact: dict[str, Any]) -> tuple[ProcessingConfig, FeatureConfig, DecisionConfig]:
    return (
        processing_config_from_dict(artifact.get("processing_config")),
        feature_config_from_dict(artifact.get("feature_config")),
        decision_config_from_dict(artifact.get("decision")),
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(value: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
