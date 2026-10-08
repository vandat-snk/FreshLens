"""FreshLens modular demo UI entry point."""

from __future__ import annotations

import torch
import streamlit as st

from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.inference import analyze_bytes, load_gate
from freshlens_ai.models import load_model
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
def load_runtime():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_model(CHECKPOINT, device)
    gate = load_gate(GATE_NPZ, GATE_META, CHECKPOINT)
    return model, metadata, gate, device


def render_sidebar_navigation() -> str:
    current_page = st.query_params.get("page", "Chẩn đoán")

    pages = ["Chẩn đoán", "Đánh giá", "Giải thích", "About"]

    with st.sidebar:
        st.markdown(
            """
            <style>
            /* Sidebar navigation */
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

        st.markdown(
            "<h1 style='margin-bottom: 18px;'>FreshLens</h1>",
            unsafe_allow_html=True,
        )

        nav_html = '<div class="freshlens-nav">'

        for page in pages:
            active = "active" if page == current_page else ""
            nav_html += (
                f'<a class="{active}" '
                f'href="?page={page}" target="_self">'
                f'{page}'
                f'</a>'
            )

        nav_html += "</div>"

        st.markdown(nav_html, unsafe_allow_html=True)

    return current_page

selected_page = render_sidebar_navigation()

if selected_page == "Chẩn đoán":
    try:
        model, metadata, gate, device = load_runtime()
    except Exception as exc:
        st.error(f"Chưa thể khởi động mô hình: {exc}")
        st.code(r".\.venv\Scripts\python.exe BUILD_OPENSET_GATE_V2.py --device cuda", language="text")
        st.stop()

    render_diagnosis_page(
        model=model,
        metadata=metadata,
        gate=gate,
        device=device,
        analyze_fn=analyze_bytes,
        show_sidebar=False,
    )
elif selected_page == "Đánh giá":
    render_evaluation_page()
elif selected_page == "Giải thích":
    render_explanation_page()
else:
    render_about_page()
