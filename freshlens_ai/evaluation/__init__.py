"""Public evaluation API for FreshLens AI."""

from .metrics import (
    calculate_metrics,
    classification_metrics,
    evaluate_model,
)
from .reports import (
    plot_confusion,
    plot_history,
    save_predictions,
)

__all__ = [
    "calculate_metrics",
    "classification_metrics",
    "evaluate_model",
    "plot_confusion",
    "plot_history",
    "save_predictions",
]
