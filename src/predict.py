"""Inference shared by the CLI and Streamlit UI."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ARTIFACT_ROOT, FRUIT_LABELS_VI, STATUS_LABELS_VI
from .features import extract_features
from .image_processing import ArrayLikeSource, preprocess_image
from .model import (
    STATE_LABELS_VI,
    artifact_configs,
    decide_fruit,
    estimator_classes,
    json_safe,
    load_artifact,
    predict_proba_safe,
    quality_gate_reason,
)
from .schemas import PredictionResult, ProcessedImage


class FreshLensPredictor:
    """Load one validated artifact and apply its exact preprocessing contract."""

    def __init__(self, model_path: Path = ARTIFACT_ROOT / "freshlens_model.joblib") -> None:
        self.model_path = Path(model_path)
        self.artifact = load_artifact(self.model_path)
        self.processing_config, self.feature_config, self.decision_config = artifact_configs(self.artifact)
        self.fruit_model = self.artifact["fruit_model"]
        self.global_status_model = self.artifact.get("global_status_model")
        self.status_models = self.artifact.get("status_models") or {}
        self.model_run_id = str(self.artifact.get("run_id", ""))
        self.model_version = str(self.artifact.get("model_version", ""))

    def analyze(self, source: ArrayLikeSource) -> tuple[PredictionResult, ProcessedImage]:
        processed = preprocess_image(source, self.processing_config)
        vector = extract_features(processed, self.feature_config)
        if vector.shape[0] != int(self.artifact["feature_dim"]):
            raise ValueError(
                f"Ảnh tạo vector {vector.shape[0]} chiều nhưng mô hình cần {self.artifact['feature_dim']} chiều"
            )
        probabilities = predict_proba_safe(self.fruit_model, vector.reshape(1, -1))[0]
        classes = estimator_classes(self.fruit_model).astype(str)
        fruit_decision = decide_fruit(probabilities, classes, self.decision_config)
        quality_reason = quality_gate_reason(processed.quality)
        if quality_reason:
            fruit_decision = {
                **fruit_decision,
                "state": "LOW_CONFIDENCE",
                "accepted": False,
                "fruit": None,
                "reason": quality_reason,
            }
        order = np.argsort(probabilities)[::-1]
        top_probabilities = [
            {
                "class": str(classes[index]),
                "label": FRUIT_LABELS_VI.get(str(classes[index]), str(classes[index])),
                "probability": float(probabilities[index]),
            }
            for index in order[: min(3, len(order))]
        ]
        image_hash = hashlib.sha256(processed.processed_bgr.tobytes()).hexdigest()
        fruit = fruit_decision["fruit"]
        status: str | None = None
        status_label: str | None = None
        status_confidence: float | None = None
        reason = str(fruit_decision["reason"])
        status_debug: dict[str, Any] = {}
        if fruit_decision["accepted"] and fruit:
            status_model = self.status_models.get(fruit) or self.global_status_model
            if status_model is None:
                reason += "; chưa có mô hình tình trạng đủ dữ liệu"
            else:
                status_probabilities = predict_proba_safe(status_model, vector.reshape(1, -1))[0]
                status_classes = estimator_classes(status_model).astype(str)
                status_index = int(np.argmax(status_probabilities))
                status_confidence = float(status_probabilities[status_index])
                status_debug["probabilities"] = [
                    {"class": str(name), "label": STATUS_LABELS_VI.get(str(name), str(name)), "probability": float(probability)}
                    for name, probability in zip(status_classes, status_probabilities)
                ]
                if status_confidence >= self.decision_config.min_status_probability:
                    status = str(status_classes[status_index])
                    status_label = STATUS_LABELS_VI.get(status, status)
                    reason += f"; tình trạng đạt ngưỡng {self.decision_config.min_status_probability:.1%}"
                else:
                    reason += f"; tình trạng chưa đủ tin cậy ({status_confidence:.1%})"
        result = PredictionResult(
            state=str(fruit_decision["state"]),
            state_label=STATE_LABELS_VI[str(fruit_decision["state"])],
            fruit=fruit,
            fruit_label=FRUIT_LABELS_VI.get(fruit) if fruit else None,
            fruit_confidence=float(fruit_decision["confidence"]),
            status=status,
            status_label=status_label,
            status_confidence=status_confidence,
            top_fruit_probabilities=top_probabilities,
            accepted=bool(fruit_decision["accepted"]),
            reason=reason,
            model_run_id=self.model_run_id,
            model_version=self.model_version,
            image_sha256=image_hash,
            quality=processed.quality,
            debug={
                "feature_dim": int(vector.shape[0]),
                "feature_order": self.artifact.get("feature_order", []),
                "fruit_probabilities": probabilities.tolist(),
                "fruit_classes": classes.tolist(),
                "status": status_debug,
                "quality_gate": quality_reason,
            },
        )
        return result, processed

    def predict(self, source: ArrayLikeSource) -> PredictionResult:
        result, _ = self.analyze(source)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Dự đoán một ảnh bằng FreshLens")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=ARTIFACT_ROOT / "freshlens_model.joblib")
    args = parser.parse_args()
    try:
        result = FreshLensPredictor(args.model).predict(args.image)
        print(json.dumps(json_safe(result.to_dict()), ensure_ascii=False, indent=2))
    except Exception as exc:  # noqa: BLE001 - CLI prints a useful user-facing error
        print(json.dumps({"state": "INPUT_ERROR", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False, indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
