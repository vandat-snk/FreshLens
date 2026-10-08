"""About page for the FreshLens Streamlit UI."""

from __future__ import annotations

import streamlit as st


def render_about_page() -> None:
    _render_styles()

    _hero()

    left_col, right_col = st.columns([1.05, 1], gap="medium")

    with left_col:
        _purpose()
        _supported_fruits()
        _scope()

    with right_col:
        _team()
        _technology()

    st.caption(
        "FreshLens · AI Fruit Condition Diagnosis · "
        "Built for academic demonstration and evaluation."
    )


# ============================================================
# HEADER
# ============================================================

def _hero() -> None:
    st.markdown(
        """
        <style>
        .about-title {
            color: #16324f;
            font-size: 1.8rem;
            font-weight: 750;
            margin-bottom: 0.15rem;
        }

        .about-subtitle {
            color: #1f8a57;
            font-size: 0.95rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .about-description {
            color: #607386;
            font-size: 0.82rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.markdown(
            '<div class="about-title">FreshLens</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="about-subtitle">'
            'AI-powered fruit condition diagnosis'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="about-description">'
            'Nhận diện loại trái cây và tình trạng Tươi / Hỏng '
            'từ ảnh upload hoặc camera.'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# PROJECT PURPOSE
# ============================================================

def _purpose() -> None:
    with st.container(border=True):

        st.markdown("### Mục tiêu")

        st.markdown(
            """
            FreshLens là hệ thống AI nhận diện loại trái cây và
            tình trạng Fresh / Rotten trên một ảnh có một quả chính.

            Hệ thống không phải công cụ kiểm nghiệm an toàn thực phẩm.
            """
        )


# ============================================================
# SUPPORTED FRUITS
# ============================================================

def _supported_fruits() -> None:
    with st.container(border=True):

        st.markdown("### Phạm vi hỗ trợ")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("🍎")
            st.caption("Apple / Táo")

        with col2:
            st.markdown("🍌")
            st.caption("Banana / Chuối")

        with col3:
            st.markdown("🍊")
            st.caption("Orange / Cam")

        with col4:
            st.markdown("🍅")
            st.caption("Tomato / Cà chua")


# ============================================================
# SCOPE & LIMITATIONS
# ============================================================

def _scope() -> None:
    with st.container(border=True):

        st.markdown("### Hạn chế")

        st.markdown(
            """
            - Chỉ hỗ trợ 4 loại quả: Apple, Banana, Orange, Tomato.
            - Kết quả phụ thuộc vào chất lượng ảnh đầu vào.
            - Input giả định có một quả chính trong ảnh.
            - Fresh / Rotten chỉ mô tả tình trạng nhìn thấy,
              không phải đánh giá an toàn thực phẩm.
            - Camera, ánh sáng và background thực tế có thể khác
              dữ liệu validation.
            - Open-set rejection không đảm bảo nhận diện đúng mọi ảnh không thuộc phạm vi hỗ trợ.
            - Kết quả validation không đồng nghĩa với hiệu năng thực tế.
            """
        )


# ============================================================
# TEAM
# ============================================================

def _team() -> None:
    with st.container(border=True):

        st.markdown("### Thành viên thực hiện")

        _team_member("Thành viên 1", "Data & Image Processing")
        _team_member("Thành viên 2", "AI Training")
        _team_member("Thành viên 3", "Inference & Evaluation")
        _team_member("Thành viên 4", "UI, Integration, Cleanup & QA")


def _team_member(code: str, role: str) -> None:
    col1, col2 = st.columns([0.25, 1])

    with col1:
        st.markdown(f"**{code}**")

    with col2:
        st.markdown(role)


# ============================================================
# TECHNOLOGY
# ============================================================

def _technology() -> None:
    with st.container(border=True):

        st.markdown("### Công nghệ sử dụng")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Python**")
            st.caption("Core application language.")

            st.markdown("**PyTorch**")
            st.caption("Model training and inference runtime.")

            st.markdown("**torchvision**")
            st.caption("EfficientNet-B0 and image transforms.")

            st.markdown("**EfficientNet-B0**")
            st.caption("CNN backbone for 8 joint classes.")

        with col2:
            st.markdown("**Streamlit**")
            st.caption("Upload/camera demo UI.")

            st.markdown("**Open-set gate**")
            st.caption("Supported-vs-unsupported decision layer.")

            st.markdown("**Pandas / NumPy**")
            st.caption("Artifact loading, metrics tables and gate math.")


# ============================================================
# STYLES
# ============================================================

def _render_styles() -> None:
    st.markdown(
        """
        <style>

        /* Page width */
        .block-container {
            max-width: 1200px;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }

        /* Streamlit containers */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 10px;
            border-color: #dce4ec;
            background: #ffffff;
        }

        /* Section headings */
        h3 {
            color: #16324f !important;
            font-size: 0.95rem !important;
            margin-bottom: 0.55rem !important;
        }

        /* Normal text */
        .stMarkdown p {
            color: #52667a;
            font-size: 0.78rem;
            line-height: 1.45;
        }

        /* Captions */
        .stCaption {
            font-size: 0.68rem !important;
        }

        /* Reduce spacing */
        .element-container {
            margin-bottom: 0.25rem;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )