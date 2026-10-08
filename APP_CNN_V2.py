"""FreshLens modular Streamlit application: Nhi UI + TV3 inference/quality policy."""

from __future__ import annotations

import os

import streamlit as st
import torch

from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.inference import clear_result
from freshlens_ai.inference.cnn_predict import FreshLensPredictor
from freshlens_ai.inference.quality import (
    DEFAULT_QUALITY_CONFIG,
    load_quality_config,
    quality_config_hash,
)
from ui.about import render_about_page
from ui.diagnosis import render_diagnosis_page
from ui.evaluation import render_evaluation_page
from ui.explanation import render_explanation_page

CHECKPOINT = PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "best.pt"
GATE_NPZ = PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "open_set_gate.npz"
GATE_META = PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "open_set_gate.json"

st.set_page_config(
    page_title="FreshLens CNN",
    page_icon="🍎",
    layout="wide",
)


@st.cache_resource
def load_runtime(quality_config):
    """Reuse a loaded model/gate only for matching quality configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return FreshLensPredictor(
        CHECKPOINT,
        GATE_NPZ,
        str(device),
        gate_meta_path=GATE_META,
        quality_config=quality_config,
    )


def render_sidebar_navigation() -> str:
    current_page = st.query_params.get("page", "Chẩn đoán")
    pages = ["Chẩn đoán", "Đánh giá", "Giải thích", "About"]

    with st.sidebar:
        st.markdown(
            """
            <style>
            .freshlens-nav {
                margin-top: 18px;
            }
            .freshlens-nav a {
                display: block;
                padding: 9px 14px;
                margin: 3px 0;
                border-radius: 8px;
                text-decoration: none !important;
                color: #24344D !important;
                font-size: 16px;
                font-weight: 400;
                transition: background-color 0.15s ease;
            }
            .freshlens-nav a:hover {
                background-color: #F1F3F6;
            }
            .freshlens-nav a.active {
                background-color: #E8EDF5;
                color: #163B6D !important;
                font-weight: 600;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<h1 style='margin-bottom: 18px;'>FreshLens</h1>", unsafe_allow_html=True)
        nav_html = '<div class="freshlens-nav">'
        for page in pages:
            active = "active" if page == current_page else ""
            nav_html += (
                f'<a class="{active}" href="?page={page}" target="_self">'
                f"{page}</a>"
            )
        nav_html += "</div>"
        st.markdown(nav_html, unsafe_allow_html=True)

    return current_page


selected_page = render_sidebar_navigation()

if selected_page == "Chẩn đoán":
    try:
        quality_path = os.environ.get("FRESHLENS_QUALITY_CONFIG") or DEFAULT_QUALITY_CONFIG
        quality_config = load_quality_config(quality_path)

        quality_enabled = st.sidebar.checkbox(
            "Thử nghiệm kiểm tra chất lượng ảnh",
            value=False,
            key="quality_enabled",
            help=(
                "Các ngưỡng chất lượng đang thử nghiệm và chưa được hiệu chỉnh "
                "trên bộ ảnh thực tế."
            ),
        )

        policy_identity = (bool(quality_enabled), quality_config_hash(quality_config))
        if st.session_state.get("quality_policy_key") != policy_identity:
            clear_result(st.session_state)
            st.session_state["quality_policy_key"] = policy_identity

        predictor = load_runtime(quality_config)
    except Exception as exc:
        st.error(f"Chưa thể khởi động mô hình: {exc}")
        st.stop()

    render_diagnosis_page(
        predictor=predictor,
        show_sidebar=False,
        check_quality=quality_enabled,
    )

elif selected_page == "Đánh giá":
    render_evaluation_page()

elif selected_page == "Giải thích":
    render_explanation_page()

else:
    render_about_page()
