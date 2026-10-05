"""Public inference API for FreshLens AI."""

from .app_state import (
    IMAGE_HASH_KEY,
    RESULT_KEY,
    clear_result,
    image_identity,
    result_for_image,
    store_result,
)
from .gate_builder import (
    KNOWN_FOLDERS,
    SUPPORTED_EXTENSIONS,
    build_metadata,
    build_prototypes,
    choose_threshold,
    collect_images,
    collect_known,
    deterministic_split,
    fit_gate,
    infer_paths,
)
from .open_set import (
    analyze_bytes,
    embedding_and_probabilities,
    gate_features,
    load_gate,
    normalize_rows,
    sigmoid,
    supported_probability,
)
from .predictor import (
    predict_bytes,
    predict_image,
)

__all__ = [
    "IMAGE_HASH_KEY",
    "KNOWN_FOLDERS",
    "RESULT_KEY",
    "SUPPORTED_EXTENSIONS",
    "analyze_bytes",
    "build_metadata",
    "build_prototypes",
    "choose_threshold",
    "clear_result",
    "collect_images",
    "collect_known",
    "deterministic_split",
    "embedding_and_probabilities",
    "fit_gate",
    "gate_features",
    "image_identity",
    "infer_paths",
    "load_gate",
    "normalize_rows",
    "predict_bytes",
    "predict_image",
    "result_for_image",
    "sigmoid",
    "store_result",
    "supported_probability",
]
