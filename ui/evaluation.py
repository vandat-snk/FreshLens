"""Model evaluation dashboard for the FreshLens Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

import pandas as pd
import streamlit as st
import altair as alt
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

    
    chart_df = history[[epoch, *columns]].copy()
    chart_df = chart_df.rename(columns={epoch: "Epoch"})
    chart_df = chart_df.melt(
        id_vars="Epoch",
        var_name="Metric",
        value_name="Value",
    )
    chart_df["Metric"] = chart_df["Metric"].map(_pretty_metric_name)

    pastel_colors = [
        "#A8D5B5",  # xanh lá nhạt
        "#79B995",  # xanh lá dịu
        "#D5E8D9",  # xanh bạc hà
        "#B7CDBD",  # xanh xám nhạt
    ]

    chart = (
        alt.Chart(chart_df)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("Epoch:Q", title="Epoch"),
            y=alt.Y("Value:Q", title=title),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(range=pastel_colors),
                legend=alt.Legend(title=None),
            ),
            tooltip=["Epoch:Q", "Metric:N", "Value:Q"],
        )
        .properties(height=300)
        .interactive()
    )

    st.altair_chart(chart, width="stretch")



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
        .background_gradient(cmap="Greens", axis=None, vmin=0)
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

    
    chart = (
        alt.Chart(chart_df.reset_index())
        .mark_bar(
            color="#A8D5B5",
            cornerRadiusEnd=5,
        )
        .encode(
            x=alt.X("F1:Q", title="F1 Score"),
            y=alt.Y("Class:N", sort="-x", title=None),
            tooltip=["Class:N", alt.Tooltip("F1:Q", format=".2%")],
        )
        .properties(height=360)
    )

    st.altair_chart(chart, width="stretch")


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

    
    chart = (
        alt.Chart(df.reset_index())
        .mark_bar(
            color="#C8E6CF",
            cornerRadiusEnd=5,
        )
        .encode(
            x=alt.X("Value:Q", title="Score"),
            y=alt.Y("Metric:N", sort="-x", title=None),
            tooltip=["Metric:N", alt.Tooltip("Value:Q", format=".2%")],
        )
        .properties(height=300)
    )

    st.altair_chart(chart, width="stretch")



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
        :root {
            --fresh-dark: #16352a;
            --fresh-green: #3fa66b;
            --fresh-green-dark: #2f7d50;
            --fresh-green-soft: #e8f5ea;
            --fresh-cream: #faf9f5;
            --fresh-white: #ffffff;
            --fresh-border: #e2e9e4;
            --fresh-muted: #718078;
        }

        .stApp {
            background: #faf9f5;
        }

        .block-container {
            max-width: 1220px;
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        .fl-header {
            margin-bottom: 1.2rem;
        }

        .fl-header h1 {
            margin: 0 0 .35rem;
            color: #16352a;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.035em;
        }

        .fl-header p {
            margin: 0;
            color: #718078;
            font-size: .95rem;
        }

        [data-testid="stCaptionContainer"] p {
            color: #718078;
        }

        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e2e9e4;
            border-radius: 16px;
            padding: 1rem 1.1rem;
            box-shadow: 0 3px 12px rgba(22, 53, 42, .035);
        }

        [data-testid="stMetricLabel"] {
            color: #718078;
            font-size: .83rem;
            font-weight: 600;
        }

        [data-testid="stMetricValue"] {
            color: #2f7d50;
            font-weight: 800;
        }

        [data-testid="stVerticalBlock"] > div:has(> [data-testid="stTabs"]) {
            margin-top: .3rem;
        }

        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: .45rem;
            border-bottom: 1px solid #e2e9e4;
        }

        [data-testid="stTabs"] button[data-baseweb="tab"] {
            color: #718078;
            border-radius: 10px 10px 0 0;
            padding: .6rem 1rem;
        }

        [data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
            color: #2f7d50;
            border-bottom-color: #8bc9a0;
            font-weight: 750;
        }

        h2, h3 {
            color: #16352a !important;
            font-weight: 750 !important;
            letter-spacing: -0.02em;
        }

        .fl-info-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: .75rem;
            margin: .8rem 0 1.5rem;
        }

        .fl-info {
            background: #ffffff;
            border: 1px solid #e2e9e4;
            border-radius: 15px;
            padding: .9rem 1rem;
            box-shadow: 0 3px 12px rgba(22, 53, 42, .03);
        }

        .fl-info div {
            color: #718078;
            font-size: .78rem;
            margin-bottom: .35rem;
        }

        .fl-info strong {
            color: #2f7d50;
            font-size: .98rem;
            font-weight: 750;
        }

        [data-testid="stAlert"] {
            border-radius: 12px;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #e2e9e4;
            border-radius: 14px;
            overflow: hidden;
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

            .fl-header h1 {
                font-size: 1.65rem;
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
