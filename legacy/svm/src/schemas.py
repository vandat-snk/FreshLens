"""Typed contracts shared by the data, model and UI layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class ProcessedImage:
    """All image representations used by one train/predict pass.

    Arrays are BGR uint8 internally. The UI converts to RGB only at display
    time. Keeping this contract explicit prevents the classic RGB/BGR bug.
    """

    original_bgr: np.ndarray
    letterboxed_bgr: np.ndarray
    clahe_bgr: np.ndarray
    gaussian_bgr: np.ndarray
    processed_bgr: np.ndarray
    edges: np.ndarray
    source_size: tuple[int, int]
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    state: str
    state_label: str
    fruit: str | None
    fruit_label: str | None
    fruit_confidence: float | None
    status: str | None
    status_label: str | None
    status_confidence: float | None
    top_fruit_probabilities: list[dict[str, Any]]
    accepted: bool
    reason: str
    model_run_id: str | None
    model_version: str | None
    image_sha256: str | None
    quality: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

