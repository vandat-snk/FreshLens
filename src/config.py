"""Single source of truth for paths and experiment configuration.

The project intentionally keeps the dataset outside the source tree when the
team has a dedicated data drive. On Windows, the default is E:\\FreshLens\\dataset.
Set FRESHLENS_DATASET_ROOT to override it on any machine.
"""

from __future__ import annotations

import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except ImportError:
    pass  # Explicit environment variables still work without python-dotenv.

ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"
REPORT_ROOT = PROJECT_ROOT / "reports"
LOG_ROOT = PROJECT_ROOT / "logs"


def default_dataset_root() -> Path:
    configured = os.getenv("FRESHLENS_DATASET_ROOT")
    if configured:
        return Path(configured).expanduser()
    if platform.system().lower().startswith("win"):
        return Path(r"E:\FreshLens\dataset")
    return PROJECT_ROOT / "data" / "dataset"


DATASET_ROOT = default_dataset_root()
RAW_DATASET_ROOT = DATASET_ROOT / "raw"
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
MODEL_PATH = ARTIFACT_ROOT / "freshlens_model.joblib"
METRICS_PATH = REPORT_ROOT / "latest_metrics.json"
PREDICTIONS_PATH = REPORT_ROOT / "latest_predictions.csv"


FRUIT_CLASSES = ("apple", "banana", "orange", "tomato")
STATUS_CLASSES = ("fresh", "rotten")
OTHER_CLASS = "other"
FRUIT_CLASSES_WITH_OTHER = FRUIT_CLASSES + (OTHER_CLASS,)

FRUIT_LABELS_VI = {
    "apple": "Táo",
    "banana": "Chuối",
    "orange": "Cam",
    "tomato": "Cà chua",
    "other": "Ngoài phạm vi",
}
STATUS_LABELS_VI = {"fresh": "Tươi", "rotten": "Hỏng"}


@dataclass(frozen=True)
class ProcessingConfig:
    image_size: int = 224
    pad_value: int = 255
    use_clahe: bool = True
    use_gaussian: bool = True
    gaussian_kernel: int = 3
    clahe_clip_limit: float = 2.0
    clahe_grid_size: int = 8


@dataclass(frozen=True)
class FeatureConfig:
    color: bool = True
    texture: bool = True
    shape: bool = True
    hog: bool = True
    lbp_points: int = 16
    lbp_radius: int = 2
    hog_orientations: int = 9
    hog_cell_size: int = 16
    hog_block_cells: int = 2


@dataclass(frozen=True)
class DecisionConfig:
    min_fruit_probability: float = 0.58
    min_probability_margin: float = 0.10
    min_status_probability: float = 0.55
    reject_other_class: bool = True


DEFAULT_PROCESSING = ProcessingConfig()
DEFAULT_FEATURES = FeatureConfig()
DEFAULT_DECISION = DecisionConfig()
RANDOM_SEED = 42


def ensure_runtime_dirs() -> None:
    """Create only project-owned runtime directories."""

    for path in (ARTIFACT_ROOT, REPORT_ROOT, LOG_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def config_as_dict(
    processing: ProcessingConfig = DEFAULT_PROCESSING,
    features: FeatureConfig = DEFAULT_FEATURES,
    decision: DecisionConfig = DEFAULT_DECISION,
) -> dict[str, Any]:
    return {
        "processing": asdict(processing),
        "features": asdict(features),
        "decision": asdict(decision),
        "fruit_classes": list(FRUIT_CLASSES),
        "status_classes": list(STATUS_CLASSES),
        "other_class": OTHER_CLASS,
        "random_seed": RANDOM_SEED,
    }


def feature_config_from_dict(data: dict[str, Any] | None) -> FeatureConfig:
    data = data or {}
    allowed = set(FeatureConfig.__dataclass_fields__)
    return FeatureConfig(**{key: value for key, value in data.items() if key in allowed})


def processing_config_from_dict(data: dict[str, Any] | None) -> ProcessingConfig:
    data = data or {}
    allowed = set(ProcessingConfig.__dataclass_fields__)
    return ProcessingConfig(**{key: value for key, value in data.items() if key in allowed})


def decision_config_from_dict(data: dict[str, Any] | None) -> DecisionConfig:
    data = data or {}
    allowed = set(DecisionConfig.__dataclass_fields__)
    return DecisionConfig(**{key: value for key, value in data.items() if key in allowed})
