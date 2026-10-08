"""Model evaluation dashboard for the FreshLens Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd
import streamlit as st

from freshlens_ai.constants import CLASSES, FRUITS, PROJECT_DIR, STATUSES


MODEL_DIR = PROJECT_DIR / "models" / "cnn_efficientnet_b0"
BEST_VALIDATION = MODEL_DIR / "best_validation.json"
TRAINING_SUMMARY = MODEL_DIR / "training_summary.json"
HISTORY_CSV = MODEL_DIR / "history.csv"
ERRORS_CSV = MODEL_DIR / "validation_errors.csv"
CONFUSION_MATRIX_PNG = MODEL_DIR / "validation_confusion_matrix.png"
EVAL_RESULTS = PROJECT_DIR / "eval_results" / "metrics.json"

FRUIT_LABELS = {
    "apple": "Apple",
    "banana": "Banana",
    "orange": "Orange",
    "tomato": "Tomato",
}
STATUS_LABELS = {"fresh": "Fresh", "rotten": "Rotten"}


def render_evaluation_page() -> None:
    """Render a compact model-evaluation dashboard from repository artifacts."""
    _render_styles()

    artifacts = _load_artifacts()
    best = artifacts["best"]
    summary = artifacts["summary"]
    metrics = _best_metrics(best, summary)

    st.markdown(
        """
        <div class="fl-header">
            <h1>Đánh giá mô hình</h1>
            <p>Hiệu năng của mô hình trên tập validation</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Metrics hiện tại được tính trên tập validation, không phải final test set."
    )

    _render_kpis(metrics, best, summary)
    _render_model_info(metrics, best, summary)

    # 4 main dashboard blocks:
    # 1) train/validation curves
    # 2) confusion matrix
    # 3) per-class F1
    # 4) external benchmark
    _render_training_dashboard(artifacts["history"])
    _render_confusion_matrix(metrics)
    _render_per_class_f1(metrics)
    _render_external_benchmark(artifacts["external"])

    if artifacts["errors"] is not None:
        st.caption(f"Validation errors: {len(artifacts['errors']):,}")


@st.cache_data(show_spinner=False)
def _load_artifacts() -> dict[str, Any]:
    return {
        "best": _read_json(BEST_VALIDATION),
        "summary": _read_json(TRAINING_SUMMARY),
        "history": _read_csv(HISTORY_CSV),
        "errors": _read_csv(ERRORS_CSV),
        "external": _read_json(EVAL_RESULTS),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _best_metrics(
    best: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    if isinstance(best, dict) and isinstance(best.get("metrics"), dict):
        return best["metrics"]
    if isinstance(summary, dict) and isinstance(summary.get("best_validation"), dict):
        return summary["best_validation"]
    return {}


def _render_kpis(
    metrics: dict[str, Any],
    best: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> None:
    cards = [
        ("Joint Accuracy", _fmt_percent(_nested(metrics, "joint", "accuracy"))),
        ("Fruit Accuracy", _fmt_percent(_nested(metrics, "fruit", "accuracy"))),
        (
            "Condition Accuracy",
            _fmt_percent(_nested(metrics, "condition", "accuracy")),
        ),
        ("Joint Macro F1", _fmt_percent(_nested(metrics, "joint", "macro_f1"))),
    ]

    cols = st.columns(4)
    for col, (label, value) in zip(cols, cards):
        with col:
            st.metric(label, value)


def _render_model_info(
    metrics: dict[str, Any],
    best: dict[str, Any] | None,
    summary: dict[str, Any] | None,
) -> None:
    _info_grid(
        [
            ("Validation images", _fmt_plain(metrics.get("images"))),
            ("Validation groups", _fmt_plain(metrics.get("groups"))),
            ("Completed epochs", _fmt_plain(_first(summary, "completed_epochs"))),
            (
                "Best epoch",
                _fmt_plain(
                    _first(best, "epoch", fallback=_first(summary, "best_epoch"))
                ),
            ),
            ("Best validation loss", _fmt_number(metrics.get("loss"))),
        ]
    )


def _render_training_dashboard(history: pd.DataFrame | None) -> None:
    st.subheader("Train / Validation curves")

    if history is None or history.empty:
        st.info("Dữ liệu history.csv chưa có.")
        return

    epoch = _find_col(history, ["epoch"])
    train_loss = _find_col(history, ["train_loss", "training_loss"])
    val_loss = _find_col(history, ["val_loss", "validation_loss"])
    joint_acc = _find_col(history, ["val_joint_accuracy", "joint_accuracy"])
    fruit_acc = _find_col(history, ["val_fruit_accuracy", "fruit_accuracy"])
    condition_acc = _find_col(
        history, ["val_condition_accuracy", "condition_accuracy"]
    )
    macro_f1 = _find_col(history, ["val_joint_macro_f1", "joint_macro_f1"])

    if epoch is None:
        st.info("history.csv chưa có cột epoch phù hợp.")
        return

    tab_loss, tab_acc, tab_f1 = st.tabs(["Loss", "Accuracy", "Macro F1"])

    with tab_loss:
        columns = [col for col in [train_loss, val_loss] if col]
        _line_chart(history, epoch, columns, "Loss")

    with tab_acc:
        columns = [col for col in [joint_acc, fruit_acc, condition_acc] if col]
        _line_chart(history, epoch, columns, "Validation Accuracy")

    with tab_f1:
        columns = [macro_f1] if macro_f1 else []
        _line_chart(history, epoch, columns, "Validation Joint Macro F1")


def _line_chart(
    history: pd.DataFrame,
    epoch: str,
    columns: list[str],
    title: str,
) -> None:
    if not columns:
        st.info(f"Dữ liệu cho {title} chưa có.")
        return

    chart_df = history[[epoch, *columns]].copy().set_index(epoch)
    chart_df.columns = [_pretty_metric_name(col) for col in chart_df.columns]
    st.line_chart(chart_df, width="stretch", height=300)


def _render_confusion_matrix(metrics: dict[str, Any]) -> None:
    st.subheader("Joint Confusion Matrix")

    matrix = _nested(metrics, "joint", "confusion_matrix")

    if not matrix and CONFUSION_MATRIX_PNG.is_file():
        st.image(str(CONFUSION_MATRIX_PNG), width="stretch")
        return

    if not matrix:
        st.info("Dữ liệu confusion matrix chưa có.")
        return

    labels = [_display_class(label) for label in CLASSES]
    df = pd.DataFrame(matrix, index=labels, columns=labels)

    styled = (
        df.style
        .background_gradient(cmap="Blues", axis=None)
        .format("{:.0f}")
        .set_properties(**{"text-align": "center"})
    )

    st.dataframe(styled, width="stretch", height=390)
    st.caption("Rows = true class · Columns = predicted class")


def _render_per_class_f1(metrics: dict[str, Any]) -> None:
    st.subheader("Per-class F1")

    per_class = _nested(metrics, "joint", "per_class")

    if not isinstance(per_class, dict) or not per_class:
        st.info("Dữ liệu per-class F1 chưa có.")
        return

    rows = []

    for name, values in per_class.items():
        if isinstance(values, dict) and values.get("f1") is not None:
            rows.append(
                {
                    "Class": _display_class(name),
                    "F1": float(values["f1"]),
                }
            )

    if not rows:
        st.info("Dữ liệu per-class F1 chưa có.")
        return

    df = pd.DataFrame(rows).sort_values("F1", ascending=True)
    chart_df = df.set_index("Class")[["F1"]]

    st.bar_chart(
        chart_df,
        y="F1",
        horizontal=True,
        width="stretch",
        height=360,
    )

    lowest = df.iloc[0]
    st.caption(f"Lowest F1: {lowest['Class']} · {lowest['F1']:.2%}")


def _render_external_benchmark(external: dict[str, Any] | None) -> None:
    st.subheader("External Benchmark")

    if not isinstance(external, dict) or not external:
        st.info("Chưa có dữ liệu benchmark độc lập.")
        st.caption(
            "Artifact eval_results/metrics.json chưa có trong repository hiện tại."
        )
        return

    values = _extract_numeric_metrics(external)

    if not values:
        st.info(
            "Artifact external benchmark tồn tại nhưng chưa có metric dạng số để vẽ."
        )
        return

    df = pd.DataFrame(
        {"Metric": list(values.keys()), "Value": list(values.values())}
    ).set_index("Metric")

    st.bar_chart(df, y="Value", width="stretch", height=300)


def _extract_numeric_metrics(
    data: dict[str, Any],
    prefix: str = "",
) -> dict[str, float]:
    result: dict[str, float] = {}

    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)

        if isinstance(value, bool):
            continue

        if isinstance(value, (int, float)) and 0 <= float(value) <= 1:
            result[_pretty_metric_name(name)] = float(value)
        elif isinstance(value, dict):
            result.update(_extract_numeric_metrics(value, name))

    return result


def _info_grid(rows: list[tuple[str, str]]) -> None:
    html_rows = "".join(
        f'<div class="fl-info"><div>{label}</div><strong>{value}</strong></div>'
        for label, value in rows
    )

    st.markdown(
        f'<div class="fl-info-grid">{html_rows}</div>',
        unsafe_allow_html=True,
    )


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        .fl-header {
            margin-bottom: 0.8rem;
        }

        .fl-header h1 {
            margin: 0 0 .25rem;
            color: #16324f;
            font-size: 2rem;
        }

        .fl-header p {
            margin: 0;
            color: #62748a;
        }

        .fl-info-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .65rem;
            margin: .7rem 0 1.35rem;
        }

        .fl-info {
            background: #fff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: .65rem .75rem;
        }

        .fl-info div {
            color: #62748a;
            font-size: .78rem;
            margin-bottom: .2rem;
        }

        .fl-info strong {
            color: #16324f;
            font-size: .95rem;
        }

        @media (max-width: 1100px) {
            .fl-info-grid {
                grid-template-columns: repeat(3, minmax(0, 1fr));
            }
        }

        @media (max-width: 700px) {
            .fl-info-grid {
                grid-template-columns: 1fr 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lookup = {col.lower(): col for col in df.columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def _nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)

    return current


def _first(
    data: dict[str, Any] | None,
    key: str,
    fallback: Any = None,
) -> Any:
    if isinstance(data, dict) and data.get(key) is not None:
        return data.get(key)

    return fallback


def _fmt_percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2%}"

    return "N/A"


def _fmt_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"

    return "N/A"


def _fmt_plain(value: Any) -> str:
    return "N/A" if value is None else str(value)


def _pretty_metric_name(name: Any) -> str:
    text = str(name).replace(".", " · ").replace("_", " ")

    replacements = {
        "train loss": "Train Loss",
        "val loss": "Validation Loss",
        "training loss": "Train Loss",
        "validation loss": "Validation Loss",
        "val joint accuracy": "Joint Accuracy",
        "val fruit accuracy": "Fruit Accuracy",
        "val condition accuracy": "Condition Accuracy",
        "val joint macro f1": "Joint Macro F1",
        "joint accuracy": "Joint Accuracy",
        "fruit accuracy": "Fruit Accuracy",
        "condition accuracy": "Condition Accuracy",
        "joint macro f1": "Joint Macro F1",
    }

    return replacements.get(text.lower(), text.title())


def _display_class(name: Any) -> str:
    text = str(name)

    if "::" in text:
        fruit, status = text.split("::", 1)
        return (
            f"{FRUIT_LABELS.get(fruit, fruit.title())} "
            f"{STATUS_LABELS.get(status, status.title())}"
        )

    if text in FRUITS:
        return FRUIT_LABELS.get(text, text.title())

    if text in STATUSES:
        return STATUS_LABELS.get(text, text.title())

    return text
