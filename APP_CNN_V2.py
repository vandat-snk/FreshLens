"""FreshLens modular demo UI entry point."""

from __future__ import annotations

import os

import torch
import streamlit as st

from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.inference import clear_result
from freshlens_ai.inference.quality import DEFAULT_QUALITY_CONFIG, load_quality_config, quality_config_hash
from freshlens_ai.inference.cnn_predict import FreshLensPredictor
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


@st.cache_resource(max_entries=1)
def load_runtime(quality_config):
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

    pages = [
        ("Chẩn đoán", "⌂"),
        ("Đánh giá", "▥"),
        ("Giải thích", "▤"),
        ("About", "ⓘ"),
    ]

    with st.sidebar:
        st.markdown(
            """
            <style>
            /* =========================================================
               FRESHLENS SIDEBAR
               ========================================================= */

            /* ---------- Sidebar background ---------- */

            [data-testid="stSidebar"] {
                background: linear-gradient(
                    180deg,
                    #12352A 0%,
                    #163F31 52%,
                    #102F26 100%
                ) !important;

                border-right: none !important;
                min-width: 260px;
            }

            [data-testid="stSidebar"] > div:first-child {
                background:
                    radial-gradient(
                        circle at 15% 75%,
                        rgba(118, 190, 117, 0.12) 0,
                        rgba(118, 190, 117, 0) 30%
                    ),
                    radial-gradient(
                        circle at 90% 92%,
                        rgba(255, 255, 255, 0.045) 0,
                        rgba(255, 255, 255, 0) 28%
                    );
            }

            [data-testid="stSidebarContent"] {
                padding: 1.05rem 1.05rem 1.2rem 1.05rem;
            }


            /* =========================================================
               BRAND
               ========================================================= */

            .fl-sidebar-brand {
                display: flex;
                align-items: center;
                gap: 12px;

                padding: 0 8px;

                margin-top: 10px;
                margin-bottom: 42px;
            }

            .fl-sidebar-brand-text {
                display: flex;
                flex-direction: column;
                justify-content: center;

                line-height: 1.05;
            }

            .fl-sidebar-brand-name {
                color: #FFFFFF;
                font-size: 1.55rem;
                font-weight: 800;
                letter-spacing: -0.025em;
                white-space: nowrap;
            }

            .fl-sidebar-brand-subtitle {
                color: rgba(255, 255, 255, 0.68);

                font-size: 0.72rem;
                font-weight: 600;

                letter-spacing: 0.075em;

                margin-top: 0.38rem;

                text-transform: uppercase;

                white-space: nowrap;
            }
            .fl-sidebar-logo {
                width: 52px;
                height: 52px;
                flex: 0 0 auto;

                background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath d='M32 20C27 15 19 16 15 21C10 27 11 38 15 46C19 54 25 58 32 55C39 58 45 54 49 46C53 38 54 27 49 21C45 16 37 15 32 20Z' fill='none' stroke='white' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'/%3E%3Cpath d='M32 20C32 14 36 9 42 7' fill='none' stroke='white' stroke-width='4' stroke-linecap='round'/%3E%3Cpath d='M35 12C40 8 46 9 50 11C47 16 42 18 35 17' fill='none' stroke='white' stroke-width='4' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");

                background-repeat: no-repeat;
                background-position: center;
                background-size: 48px 48px;
            }

            /* =========================================================
               NAVIGATION
               ========================================================= */

            .fl-sidebar-nav {
                display: flex;
                flex-direction: column;

                gap: 0.35rem;

                margin-top: 0.15rem;
            }

            .fl-sidebar-nav a {
                position: relative;

                display: flex;
                align-items: center;

                gap: 0.82rem;

                min-height: 54px;

                padding: 0.72rem 0.85rem;

                border-radius: 14px;

                color: rgba(255, 255, 255, 0.82) !important;

                text-decoration: none !important;

                font-size: 0.98rem;
                font-weight: 600;

                transition:
                    background-color 0.18s ease,
                    color 0.18s ease,
                    transform 0.18s ease;
            }

            .fl-sidebar-nav a:hover {
                background: rgba(255, 255, 255, 0.08);

                color: #FFFFFF !important;

                transform: translateX(2px);
            }

            .fl-sidebar-nav a.active {
                background: linear-gradient(
                    135deg,
                    #426D58 0%,
                    #365D4B 100%
                );

                color: #FFFFFF !important;

                box-shadow:
                    inset 0 1px 0 rgba(255, 255, 255, 0.08),
                    0 8px 18px rgba(0, 0, 0, 0.12);
            }

            /* Active green indicator */

            .fl-sidebar-nav a.active::before {
                content: "";

                position: absolute;

                left: 0;
                top: 10px;
                bottom: 10px;

                width: 4px;

                border-radius: 0 5px 5px 0;

                background: #A8E69D;
            }

            .fl-nav-icon {
                width: 30px;
                height: 30px;

                display: flex;
                align-items: center;
                justify-content: center;

                flex: 0 0 auto;

                color: rgba(255, 255, 255, 0.90);

                font-size: 1.18rem;
                font-weight: 700;

                border-radius: 9px;
            }

            .fl-sidebar-nav a.active .fl-nav-icon {
                color: #FFFFFF;
            }
            /* ========================================
            MOVE FRESHLEns BRAND UP
            ======================================== */

            .fl-sidebar-brand {
                position: relative;
                top: -62px;
                margin-bottom: -23px;
            }

            /* =========================================================
               BOTTOM DESCRIPTION
               ========================================================= */

            .fl-sidebar-bottom {
                position: fixed;

                left: 1.35rem;
                bottom: 1.45rem;

                width: 205px;
            }

            .fl-sidebar-line {
                width: 48px;
                height: 4px;

                margin-bottom: 0.78rem;

                border-radius: 999px;

                background: #A8E69D;
            }

            .fl-sidebar-description {
                color: rgba(255, 255, 255, 0.68);

                font-size: 0.79rem;

                line-height: 1.55;

                font-weight: 400;
            }


            /* =========================================================
               STREAMLIT CLEANUP
               ========================================================= */

            [data-testid="stSidebar"] hr {
                border-color: rgba(255, 255, 255, 0.08);
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
                color: inherit;
            }

            </style>
            """,
            unsafe_allow_html=True,
        )


        # =============================================================
        # BRAND
        # =============================================================

        st.html(
            """
            <div class="fl-sidebar-brand">
                <div class="fl-sidebar-logo"></div>

                <div class="fl-sidebar-brand-text">
                    <div class="fl-sidebar-brand-name">
                        FreshLens
                    </div>
                </div>
            </div>
            """
        )


        # =============================================================
        # NAVIGATION
        # =============================================================

        nav_html = '<div class="fl-sidebar-nav">'

        for page, icon in pages:

            active = "active" if page == current_page else ""

            nav_html += (
                f'<a class="{active}" '
                f'href="?page={page}" '
                f'target="_self">'

                f'<span class="fl-nav-icon">'
                f'{icon}'
                f'</span>'

                f'<span>'
                f'{page}'
                f'</span>'

                f'</a>'
            )

        nav_html += "</div>"

        st.markdown(
            nav_html,
            unsafe_allow_html=True,
        )


        # =============================================================
        # BOTTOM DESCRIPTION
        # =============================================================



    return current_page


selected_page = render_sidebar_navigation()


if selected_page == "Chẩn đoán":
    quality_path = os.environ.get("FRESHLENS_QUALITY_CONFIG") or DEFAULT_QUALITY_CONFIG
    try:
        quality_config = load_quality_config(quality_path)
    except (OSError, ValueError) as exc:
        clear_result(st.session_state)
        st.error(f"Không thể đọc cấu hình kiểm tra chất lượng: {exc}")
        st.caption(f"Kiểm tra file {quality_path}: file phải tồn tại, đúng định dạng JSON và có các ngưỡng hợp lệ.")
        st.stop()

    try:
        # Nhi UI always enables quality; changed thresholds invalidate old results.
        policy_identity = (True, quality_config_hash(quality_config))
        if st.session_state.get("quality_policy_key") != policy_identity:
            clear_result(st.session_state)
            st.session_state["quality_policy_key"] = policy_identity
        predictor = load_runtime(quality_config)
    except Exception as exc:
        st.error(f"Chưa thể khởi động mô hình: {exc}")
        st.caption(
            "Kiểm tra checkpoint, bộ gate đi kèm và thiết bị chạy theo lỗi ở trên. "
            "Chỉ dựng lại gate khi đã xác định cần hiệu chỉnh lại và có dữ liệu phù hợp."
        )
        st.stop()

    render_diagnosis_page(
        predictor=predictor,
        show_sidebar=False,
    )

elif selected_page == "Đánh giá":
    render_evaluation_page()

elif selected_page == "Giải thích":
    render_explanation_page()

else:
    render_about_page()
