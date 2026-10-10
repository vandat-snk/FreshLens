"""System explanation page for the FreshLens Streamlit UI."""

from __future__ import annotations

import streamlit as st
from pathlib import Path

from freshlens_ai.constants import PREPROCESS


def render_explanation_page() -> None:
    _render_styles()

    st.markdown(
        """
<div class="fl-header">
  <p class="fl-kicker">Giải thích hệ thống / Explanation</p>
  <h1>FreshLens xử lý ảnh như thế nào?</h1>
</div>
""",
        unsafe_allow_html=True,
    )

    _pipeline()
    _preprocessing()
    _gradcam_status()


def _pipeline() -> None:
    st.markdown(
        """
<section class="fl-section">
  <div class="fl-section-head">
    <h2>Pipeline</h2>
    <p>Toàn bộ quy trình xử lý ảnh từ lúc nhận ảnh đến khi đưa ra kết quả.</p>
  </div>
  <div class="fl-flow">
""",
        unsafe_allow_html=True,
    )

    steps = [
        (
            "User Upload / Camera",
            "Nhận ảnh từ người dùng dưới dạng bytes thông qua upload hoặc camera.",
        ),
        (
            "Image Decoding",
            "Đọc ảnh bằng Pillow, xử lý EXIF orientation, chuyển sang RGB và ghép kênh alpha lên nền trắng nếu cần.",
        ),
        (
            "Quality Check",
            "Đánh giá kích thước, độ mờ và độ sáng của ảnh.",
        ),
        (
            "Image Preprocessing",
            "Letterbox ảnh về 224x224, chuyển thành tensor PyTorch và chuẩn hóa theo cấu hình của model.",
        ),
        (
            "EfficientNet-B0",
            "Trích xuất đặc trưng ảnh và tạo logits cho 8 lớp kết hợp loại quả và tình trạng.",
        ),
        (
            "Softmax & Fruit-first Decoder",
            "Chuyển logits thành xác suất, chọn loại quả theo tổng xác suất Fresh/Rotten, sau đó xác định tình trạng.",
        ),
        (
            "Open-set Gate",
            "Đánh giá mức độ phù hợp của ảnh với phạm vi các loại quả được hỗ trợ.",
        ),
        (
            "Final Result",
            "Hiển thị kết quả dự đoán hoặc trạng thái từ chối do chất lượng ảnh hay ngoài phạm vi hỗ trợ.",
        ),
    ]


    for index, (title, body) in enumerate(steps, start=1):
        st.markdown(
            f"""
  <div class="fl-flow-row">
    <div class="fl-step-index">{index:02d}</div>
    <div class="fl-step-card">
      <strong>{title}</strong>
      <span>{body}</span>
    </div>
  </div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def _preprocessing() -> None:
    size = int(PREPROCESS["image_size"])
    fill = ", ".join(str(value) for value in PREPROCESS["fill_rgb"])
    mean = _fmt_vector(PREPROCESS["mean"])
    std = _fmt_vector(PREPROCESS["std"])

    st.markdown(
        f"""
<section class="fl-section">
  <div class="fl-section-head">
    <h2>Preprocessing</h2>
    <p>Các bước xử lý ảnh trước khi đưa vào mô hình.</p>
  </div>
  <div class="fl-card-grid">
    <div class="fl-card">
      <small>01</small>
      <strong>Decode image</strong>
      <span>Đọc ảnh một frame từ bytes bằng Pillow.</span>
    </div>
    <div class="fl-card">
      <small>02</small>
      <strong>EXIF orientation</strong>
      <span>Áp dụng EXIF transpose để ảnh không bị xoay sai.</span>
    </div>
    <div class="fl-card">
      <small>03</small>
      <strong>RGB conversion</strong>
      <span>Chuyển ảnh sang RGB; nếu ảnh có alpha thì composite lên nền trắng.</span>
    </div>
    <div class="fl-card">
      <small>04</small>
      <strong>Letterbox + Normalize</strong>
      <span>Letterbox về {size}x{size} với fill RGB ({fill}), rồi ToTensor và Normalize mean [{mean}], std [{std}].</span>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def _gradcam_status() -> None:
    project_root = Path(__file__).resolve().parents[1]
    image_path = project_root / "eval_results" / "fruit_input_ai_demo.png"
    gradcam_path = project_root / "eval_results" / "gradcam_fruit_ai_demo.png"

    st.markdown(
        """
        <section class="fl-section">
          <div class="fl-section-head">
            <h2>Grad-CAM</h2>
            <p>
              Minh họa vùng ảnh có đóng góp vào class prediction
              trong phần explainability.
            </p>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col_original, col_gradcam, col_note = st.columns(
        [1, 1, 1.2],
        gap="medium",
    )

    with col_original:
        st.markdown("**Ảnh gốc**")

        if image_path.is_file():
            st.image(
                str(image_path),
                caption="Input image",
                use_container_width=True,
            )
        else:
            st.warning("Không tìm thấy ảnh đầu vào.")

    with col_gradcam:
        st.markdown("**Grad-CAM**")

        if gradcam_path.is_file():
            st.image(
                str(gradcam_path),
                caption="Grad-CAM visualization",
                use_container_width=True,
            )
        else:
            st.warning("Không tìm thấy ảnh Grad-CAM.")

    with col_note:
        st.markdown(
            """
            <div class="fl-note-card">
              <strong>Giải thích dự đoán</strong>
              <p>
                Grad-CAM giúp trực quan hóa những vùng trên ảnh
                có ảnh hưởng đến kết quả dự đoán của mô hình.
              </p>
              <p class="fl-note">
                Bản đồ Grad-CAM thể hiện mức độ đóng góp vào dự đoán,
                không xác định chính xác vị trí hoặc ranh giới vùng
                trái cây bị hỏng.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _fmt_vector(values: list[float]) -> str:
    return ", ".join(f"{value:.3f}" for value in values)



def _render_styles() -> None:
    st.markdown(
        """
<style>
/* =========================
   PAGE LAYOUT
========================= */
.block-container {
    max-width: 1180px;
    padding-top: 0.5rem;
    padding-bottom: 1.5rem;
}

[data-testid="stAppViewContainer"] {
    background: #FAF9F5;
}

/* =========================
   PAGE HEADER
========================= */
.fl-header {
    margin-bottom: 1.25rem;
}

.fl-kicker {
    margin: 0 0 .35rem;
    color: #3F9465;
    font-size: .8rem;
    font-weight: 750;
    letter-spacing: .5px;
}

.fl-header h1 {
    margin: 0 0 .4rem;
    color: #16352A;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -.6px;
    line-height: 1.25;
}

.fl-header p,
.fl-section-head p,
.fl-card span,
.fl-step-card span,
.fl-note-card p {
    color: #718078;
    line-height: 1.65;
}

/* =========================
   SECTION CONTAINERS
========================= */
.fl-section {
    margin: 0 0 1.1rem;
    padding: 1.2rem;
    background: #FFFFFF;
    border: 1px solid #E2E9E4;
    border-radius: 14px;
    box-shadow: 0 3px 12px rgba(22, 53, 42, .035);
}

.fl-section-head {
    margin-bottom: 1rem;
}

.fl-section-head h2 {
    margin: 0 0 .3rem;
    color: #16352A;
    font-size: 1.15rem;
    font-weight: 750;
    letter-spacing: -.2px;
}

.fl-section-head p {
    margin: 0;
    font-size: .86rem;
}

/* =========================
   PIPELINE
========================= */
.fl-flow {
    display: grid;
    gap: .7rem;
}

.fl-flow-row {
    position: relative;
    display: grid;
    grid-template-columns: 2.5rem minmax(0, 1fr);
    gap: .8rem;
    align-items: stretch;
}

.fl-flow-row:not(:last-child)::after {
    content: "↓";
    position: absolute;
    left: .82rem;
    bottom: -.72rem;
    z-index: 1;
    color: #3F9465;
    font-size: .9rem;
    font-weight: 800;
}

.fl-step-index {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.5rem;
    min-height: 2.5rem;
    border-radius: 10px;
    background: #E8F5EA;
    color: #2F7D50;
    font-size: .76rem;
    font-weight: 800;
}

.fl-step-card {
    padding: .85rem 1rem;
    background: #FAFCFA;
    border: 1px solid #E2E9E4;
    border-radius: 11px;
    transition: border-color .2s ease,
                background .2s ease;
}

.fl-step-card:hover {
    background: #F5FAF5;
    border-color: #B7D5BE;
}

.fl-step-card strong,
.fl-card strong,
.fl-preview strong,
.fl-note-card strong {
    display: block;
    margin-bottom: .3rem;
    color: #254C39;
    font-size: .9rem;
    font-weight: 750;
}

.fl-step-card span,
.fl-card span {
    display: block;
    font-size: .82rem;
}

/* =========================
   PREPROCESSING CARDS
========================= */
.fl-card-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .8rem;
}

.fl-card {
    padding: 1rem;
    background: #FAFCFA;
    border: 1px solid #E2E9E4;
    border-radius: 12px;
    transition: transform .2s ease,
                border-color .2s ease,
                box-shadow .2s ease;
}

.fl-card:hover {
    transform: translateY(-2px);
    border-color: #B7D5BE;
    box-shadow: 0 5px 14px rgba(22, 53, 42, .05);
}

.fl-card small {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.85rem;
    height: 1.5rem;
    margin-bottom: .7rem;
    border-radius: 7px;
    background: #E8F5EA;
    color: #2F7D50;
    font-size: .68rem;
    font-weight: 800;
}

/* =========================
   GRAD-CAM
========================= */
.fl-gradcam-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1.2fr;
    gap: .8rem;
    align-items: stretch;
}

.fl-preview,
.fl-note-card {
    min-width: 0;
    padding: 1rem;
    background: #FAFCFA;
    border: 1px solid #E2E9E4;
    border-radius: 12px;
}

.fl-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    aspect-ratio: 4 / 3;
    margin-top: .7rem;
    border: 1px dashed #B7CDBD;
    border-radius: 10px;
    background: #F3F8F3;
    color: #829087;
    font-size: .82rem;
    font-weight: 650;
}

.fl-note-card p {
    margin: .4rem 0 0;
    font-size: .82rem;
}

.fl-note-card .fl-note {
    margin-top: .8rem;
    padding-top: .75rem;
    border-top: 1px solid #E2E9E4;
    color: #829087;
    font-size: .78rem;
}

/* =========================
   RESPONSIVE
========================= */
@media (max-width: 980px) {
    .fl-card-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .fl-gradcam-grid {
        grid-template-columns: 1fr 1fr;
    }

    .fl-note-card {
        grid-column: 1 / -1;
    }
}

@media (max-width: 640px) {
    .block-container {
        padding: .5rem .8rem 1rem;
    }

    .fl-header h1 {
        font-size: 1.55rem;
    }

    .fl-section {
        padding: .9rem;
        border-radius: 12px;
    }

    .fl-card-grid,
    .fl-gradcam-grid {
        grid-template-columns: 1fr;
    }

    .fl-note-card {
        grid-column: auto;
    }

    .fl-flow-row {
        grid-template-columns: 2.2rem minmax(0, 1fr);
        gap: .65rem;
    }

    .fl-step-index {
        width: 2.2rem;
        min-height: 2.2rem;
    }

    .fl-flow-row:not(:last-child)::after {
        left: .7rem;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )

