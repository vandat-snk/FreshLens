"""System explanation page for the FreshLens Streamlit UI."""

from __future__ import annotations

import base64

import streamlit as st

from freshlens_ai.constants import PREPROCESS, PROJECT_DIR


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
            "Người dùng chọn ảnh hoặc chụp ảnh trong giao diện Streamlit.",
        ),
        (
            "Preprocessing",
            "Ảnh được chuẩn hóa theo pipeline production trước khi vào model.",
        ),
        (
            "Tensor",
            "Ảnh sau xử lý được đổi thành tensor PyTorch.",
        ),
        (
            "EfficientNet-B0",
            "EfficientNet-B0 trích xuất đặc trưng và tạo output cho 8 joint classes.",
        ),
        (
            "8 Joint Classes",
            "Softmax tạo xác suất cho các cặp fruit và Fresh/Rotten.",
        ),
        (
            "Fruit-first Decoder",
            "Chọn loại quả trước, sau đó xác định Fresh/Rotten.",
        ),
        (
            "Open-set Gate",
            "Kiểm tra ảnh có thuộc phạm vi các fruit được hỗ trợ hay không.",
        ),
        (
            "Final Result",
            "UI hiển thị kết quả dự đoán hoặc trạng thái ngoài phạm vi hỗ trợ.",
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
    assets = PROJECT_DIR / "eval_results"
    input_image = base64.b64encode((assets / "fruit_input_ai_demo.png").read_bytes()).decode("ascii")
    gradcam_image = base64.b64encode((assets / "gradcam_fruit_ai_demo.png").read_bytes()).decode("ascii")
    st.markdown(
        f"""
<section class="fl-section">
  <div class="fl-section-head">
    <h2>Grad-CAM</h2>
    <p>Minh họa vùng ảnh có đóng góp vào class prediction trong phần explainability.</p>
  </div>
  <div class="fl-gradcam-grid">
    <div class="fl-preview">
      <strong>Ảnh gốc</strong>
      <img class="fl-gradcam-image" src="data:image/png;base64,{input_image}" alt="Ảnh táo mẫu do AI tạo" />
    </div>
    <div class="fl-preview">
      <strong>Grad-CAM</strong>
      <img class="fl-gradcam-image" src="data:image/png;base64,{gradcam_image}" alt="Grad-CAM tính từ ảnh táo mẫu" />
    </div>
    <div class="fl-note-card">
      <strong>Vùng quan trọng</strong>
      <p>Ví dụ minh họa: ảnh táo đầu vào do AI tạo, không thuộc bộ dữ liệu. Grad-CAM được tính bằng mô hình của dự án trên chính ảnh này. Dự đoán: Táo · Tươi.</p>
      <p class="fl-note">Đây là cặp ảnh mẫu cố định, không phải kết quả của ảnh đang tải lên.</p>
      <p class="fl-note">Lưu ý: Grad-CAM minh họa các vùng có đóng góp vào dự đoán của model; không phải segmentation vùng hỏng.</p>
    </div>
  </div>
</section>
""",
        unsafe_allow_html=True,
    )


def _fmt_vector(values: list[float]) -> str:
    return ", ".join(f"{value:.3f}" for value in values)


def _render_styles() -> None:
    st.markdown(
        """
<style>
.block-container {
    max-width: 1180px;
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.fl-header {
    margin-bottom: 1rem;
}

.fl-kicker {
    margin: 0 0 .2rem;
    color: #1f7cc9;
    font-size: .78rem;
    font-weight: 750;
}

.fl-header h1 {
    margin: 0 0 .28rem;
    color: #16324f;
    font-size: 2rem;
    letter-spacing: 0;
}

.fl-header p,
.fl-section-head p,
.fl-card span,
.fl-step-card span,
.fl-note-card p {
    color: #52667a;
    line-height: 1.45;
}

.fl-section {
    margin: 0 0 1rem;
    padding: 1rem;
    background: #ffffff;
    border: 1px solid #dce4ec;
    border-radius: 10px;
}

.fl-section-head {
    margin-bottom: .85rem;
}

.fl-section-head h2 {
    margin: 0 0 .2rem;
    color: #16324f;
    font-size: 1.05rem;
    letter-spacing: 0;
}

.fl-flow {
    display: grid;
    gap: .55rem;
}

.fl-flow-row {
    position: relative;
    display: grid;
    grid-template-columns: 2.25rem minmax(0, 1fr);
    gap: .65rem;
    align-items: stretch;
}

.fl-flow-row:not(:last-child)::after {
    content: "↓";
    position: absolute;
    left: .67rem;
    bottom: -.6rem;
    color: #1f7cc9;
    font-size: .82rem;
    font-weight: 800;
}

.fl-step-index {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 2.25rem;
    min-height: 2.25rem;
    border-radius: 8px;
    background: #e8f3ff;
    color: #1769aa;
    font-size: .72rem;
    font-weight: 800;
}

.fl-step-card,
.fl-card,
.fl-preview,
.fl-note-card {
    background: #f8fbfe;
    border: 1px solid #d9e7f3;
    border-radius: 8px;
    padding: .78rem .86rem;
}

.fl-step-card strong,
.fl-card strong,
.fl-preview strong,
.fl-note-card strong {
    display: block;
    margin-bottom: .22rem;
    color: #16324f;
    font-size: .9rem;
}

.fl-step-card span,
.fl-card span {
    display: block;
    font-size: .8rem;
}

.fl-card-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: .72rem;
}

.fl-card small {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 1.7rem;
    height: 1.25rem;
    margin-bottom: .55rem;
    border-radius: 999px;
    background: #e8f3ff;
    color: #1769aa;
    font-size: .65rem;
    font-weight: 800;
}

.fl-gradcam-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1.2fr;
    gap: .72rem;
}

.fl-gradcam-image {
    display: block;
    width: 100%;
    aspect-ratio: 1;
    object-fit: contain;
    margin-top: .55rem;
    border-radius: 8px;
    background: #ffffff;
}

.fl-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    aspect-ratio: 4 / 3;
    margin-top: .55rem;
    border: 1px dashed #a9c6df;
    border-radius: 8px;
    background: #ffffff;
    color: #6d8194;
    font-size: .8rem;
    font-weight: 650;
}

.fl-note-card p {
    margin: .35rem 0 0;
    font-size: .8rem;
}

.fl-note-card .fl-note {
    padding-top: .55rem;
    border-top: 1px solid #d9e7f3;
}

@media (max-width: 980px) {
    .fl-card-grid,
    .fl-gradcam-grid {
        grid-template-columns: 1fr 1fr;
    }

    .fl-note-card {
        grid-column: 1 / -1;
    }
}

@media (max-width: 640px) {
    .fl-header h1 {
        font-size: 1.5rem;
    }

    .fl-section {
        padding: .82rem;
    }

    .fl-card-grid,
    .fl-gradcam-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""",
        unsafe_allow_html=True,
    )
