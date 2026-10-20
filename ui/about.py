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



# ============================================================
# HEADER
# ============================================================


def _hero() -> None:
    st.markdown(
        """
        <style>
        .about-title {
            color: #16352a;
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.8px;
            margin-bottom: 0.2rem;
        }

        .about-subtitle {
            color: #3f9465;
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 0.45rem;
        }

        .about-description {
            color: #718078;
            font-size: 0.9rem;
            line-height: 1.6;
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

        fruits = [
            (
                """<svg viewBox="0 0 64 64" width="44" height="44"
                xmlns="http://www.w3.org/2000/svg">
                <g stroke="#111" stroke-width="3.5" stroke-linecap="round"
                stroke-linejoin="round" fill="white">
                <path d="M32 17 C29 9 31 5 29 3
                        M31 13 C22 3 14 6 14 6
                        C14 15 21 18 31 15" />
                <path d="M31 19 C23 13 8 17 7 31
                        C6 41 15 57 24 58
                        C29 58 30 55 33 55
                        C37 55 40 59 45 56
                        C55 50 61 36 57 26
                        C53 16 41 14 33 20 Z" />
                </g></svg>""",
                "Apple / Táo",
            ),
            (
                """<svg viewBox="0 0 64 64" width="44" height="44"
                xmlns="http://www.w3.org/2000/svg">
                <path d="M20 7 C11 6 12 21 16 30
                        C18 35 23 40 30 43
                        C37 47 45 51 51 49
                        C59 46 54 39 47 37
                        C39 35 33 31 29 26
                        C25 21 27 10 20 7 Z"
                fill="white" stroke="#111" stroke-width="3.5"
                stroke-linejoin="round"/>
                <path d="M18 30 C14 37 17 45 24 49
                        C31 53 39 52 45 49"
                fill="none" stroke="#111" stroke-width="3.5"
                stroke-linecap="round"/>
                <path d="M30 34 C35 39 40 41 47 41"
                fill="none" stroke="#111" stroke-width="3.5"
                stroke-linecap="round"/>
                </svg>""",
                "Banana / Chuối",
            ),
            (
                """<svg viewBox="0 0 64 64" width="44" height="44"
                xmlns="http://www.w3.org/2000/svg">
                <g stroke="#111" stroke-width="3.5"
                stroke-linejoin="round" stroke-linecap="round">
                <path d="M31 17 L31 8
                        M31 12 C24 3 15 5 12 5
                        C14 14 21 18 31 16"
                    fill="white"/>
                <circle cx="32" cy="35" r="22" fill="white"/>
                </g>
                <g fill="#111">
                <circle cx="24" cy="27" r="1.7"/>
                <circle cx="39" cy="25" r="1.7"/>
                <circle cx="43" cy="37" r="1.7"/>
                <circle cx="28" cy="42" r="1.7"/>
                <circle cx="20" cy="35" r="1.7"/>
                </g></svg>""",
                "Orange / Cam",
            ),
            (
                """<svg viewBox="0 0 64 64" width="44" height="44"
                xmlns="http://www.w3.org/2000/svg">
                <g stroke="#111" stroke-width="3.5"
                stroke-linejoin="round" stroke-linecap="round"
                fill="white">
                <path d="M32 17 L23 9 L23 17
                        L12 15 L19 25
                        C8 29 6 39 12 48
                        C17 56 25 59 32 58
                        C40 59 49 55 54 47
                        C60 37 54 27 44 24
                        L51 15 L40 17 L32 9 Z"/>
                </g></svg>""",
                "Tomato / Cà chua",
            ),
        ]

        cols = st.columns(4, gap="small")

        for col, (svg, name) in zip(cols, fruits):
            with col:
                st.markdown(
                    f"""
                    <div style="
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        justify-content: center;
                        text-align: center;
                        padding: 12px 2px 8px;
                        gap: 12px;
                        min-height: 120px;
                    ">
                        <div style="
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            height: 56px;
                        ">
                            {svg}
                        </div>
                        <div style="
                            color: #718078;
                            font-size: 0.88rem;
                            font-weight: 500;
                            line-height: 1.4;
                        ">
                            {name}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

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
        /* Page layout */
        .block-container {
            max-width: 1200px;
            padding-top: 0.5rem;
            padding-bottom: 1.5rem;
        }

        /* App background */
        [data-testid="stAppViewContainer"] {
            background: #faf9f5;
        }

        /* Cards */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px;
            border: 1px solid #e2e9e4;
            background: #ffffff;
            box-shadow: 0 3px 12px rgba(22, 53, 42, 0.035);
        }

        /* Card hover */
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #c8ddce;
            box-shadow: 0 5px 16px rgba(22, 53, 42, 0.06);
            transition: border-color 0.2s ease,
                        box-shadow 0.2s ease;
        }

        /* Section headings */
        h3 {
            color: #16352a !important;
            font-size: 1.05rem !important;
            font-weight: 750 !important;
            letter-spacing: -0.2px;
            margin-bottom: 0.75rem !important;
        }

        /* Main text */
        .stMarkdown p {
            color: #52645a;
            font-size: 0.86rem;
            line-height: 1.7;
        }

        /* Bold text */
        .stMarkdown strong {
            color: #254c39;
            font-weight: 700;
        }

        /* Lists */
        .stMarkdown li {
            color: #52645a;
            font-size: 0.84rem;
            line-height: 1.65;
            margin-bottom: 0.3rem;
        }

        /* Emoji and fruit labels */
        .stMarkdown h3 + div {
            color: #52645a;
        }

        /* Captions */
        .stCaption {
            color: #829087 !important;
            font-size: 0.76rem !important;
            line-height: 1.5;
        }

        /* Dividers */
        hr {
            border-color: #e2e9e4;
        }

        /* Reduce excessive spacing */
        [data-testid="stVerticalBlock"] {
            gap: 0.8rem;
        }

        /* Responsive layout */
        @media (max-width: 768px) {
            .block-container {
                padding: 1rem 1rem 1.5rem;
            }

            .about-title {
                font-size: 1.7rem;
            }

            .about-subtitle {
                font-size: 0.92rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )