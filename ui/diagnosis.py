"""Diagnosis page renderer for FreshLens Streamlit UI."""

from __future__ import annotations

import html
import time
from typing import Any

import streamlit as st

from freshlens_ai.inference import clear_result, result_for_image, store_result
from freshlens_ai.utils.file_io import sha256_bytes


SUPPORTED_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
UPLOAD_HELP = "200MB per file - JPG, PNG, BMP, WEBP"

FRUIT_VI = {
    "apple": "Táo",
    "banana": "Chuối",
    "orange": "Cam",
    "tomato": "Cà chua",
}

STATUS_VI = {
    "fresh": "Tươi",
    "rotten": "Hỏng",
}

GUIDANCE = [
    "Chỉ một quả chính trong ảnh",
    "Đảm bảo đủ ánh sáng",
    "Đưa quả lại gần camera hơn",
    "Không chụp quá xa",
]


CSS = """
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

    --fresh-orange: #ff9f43;
    --fresh-orange-soft: #fff5e8;

    --fresh-red: #ff5b5b;
    --fresh-red-soft: #fff0ef;

    --fresh-blue-soft: #eef6fa;
}


/* =========================================================
   PAGE
   ========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 82% 8%,
            rgba(168, 230, 157, .20),
            transparent 25%
        ),
        radial-gradient(
            circle at 12% 92%,
            rgba(168, 230, 157, .10),
            transparent 25%
        ),
        var(--fresh-cream);
}

.block-container {
    max-width: 1220px;
    padding-top: 1.35rem;
    padding-bottom: 2rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background: var(--fresh-white);
    border-right: 1px solid var(--fresh-border);
}

[data-testid="stSidebar"] h1 {
    color: var(--fresh-dark);
    font-size: 1.55rem;
    margin-bottom: 1.1rem;
}

.fl-nav-item {
    padding: .68rem .8rem;
    border-radius: 10px;
    color: #40534a;
    font-weight: 650;
    margin-bottom: .28rem;
}

.fl-nav-item.active {
    color: var(--fresh-white);
    background: var(--fresh-dark);
}


/* =========================================================
   HEADER
   ========================================================= */

.fl-header {
    margin-bottom: 1.25rem;
}

.fl-header h1 {
    margin: 0 0 .3rem;
    color: var(--fresh-dark);
    font-size: 2rem;
    font-weight: 850;
    letter-spacing: -.025em;
}

.fl-header p {
    margin: 0;
    color: var(--fresh-muted);
    max-width: 760px;
}


/* =========================================================
   GENERAL CARD
   ========================================================= */

.fl-card {
    background: var(--fresh-white);
    border: 1px solid var(--fresh-border);
    border-radius: 20px;
    box-shadow: 0 8px 24px rgba(22, 53, 42, .055);
    padding: 1.05rem;
}

.fl-card-title {
    font-size: .95rem;
    color: var(--fresh-dark);
    font-weight: 800;
    margin-bottom: .6rem;
}


/* =========================================================
   INPUT / UPLOAD
   ========================================================= */

.fl-upload-empty {
    border: 1px dashed #c9d9cf;
    border-radius: 12px;
    padding: .75rem;
    background: #fbfdfb;
    font-size: .88rem;
}

.fl-preview-shell {
    border: 1px solid var(--fresh-border);
    border-radius: 14px;
    background: var(--fresh-white);
    padding: .55rem;
}

.fl-preview-tools {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: .75rem;
    margin-bottom: .5rem;
    color: var(--fresh-muted);
    font-size: .82rem;
}

.fl-preview-title {
    color: var(--fresh-dark);
    font-weight: 800;
}

.fl-preview-image img {
    display: block;
    width: 100%;
    max-height: 330px;
    object-fit: contain;
    border-radius: 10px;
    background: #fbfdfb;
}

.fl-empty-preview {
    min-height: 245px;

    display: flex;
    align-items: center;
    justify-content: center;

    color: var(--fresh-muted);

    border: 1px dashed #c9d9cf;
    border-radius: 12px;

    background:
        linear-gradient(
            135deg,
            #fbfdfb 0%,
            #f7fbf8 100%
        );

    text-align: center;
    padding: 1rem;
}

.fl-empty-title {
    color: var(--fresh-dark);
    font-weight: 800;
    font-size: 1.05rem;
    margin-bottom: .35rem;
}

.fl-empty-meta {
    font-size: .88rem;
    line-height: 1.5;
}


/* =========================================================
   RESULT HEADER
   ========================================================= */

.fl-result-head {
    color: var(--fresh-dark);
    font-weight: 800;
    margin-bottom: .75rem;
    font-size: 1.08rem;
}

.fl-result-panel-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: .75rem;
}

.fl-result-heading {
    display: flex;
    align-items: center;
    gap: .7rem;
}

.fl-step-number {
    width: 42px;
    height: 42px;
    min-width: 42px;

    border-radius: 50%;

    background: var(--fresh-green-soft);
    color: var(--fresh-green-dark);

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 1.15rem;
    font-weight: 850;
}

.fl-step-title {
    color: var(--fresh-dark);
    font-size: 1.3rem;
    font-weight: 850;
    line-height: 1;
}


/* =========================================================
   EMPTY RESULT
   ========================================================= */

.fl-result-empty {
    min-height: 245px;

    border: 1px dashed #cbded2;
    border-radius: 13px;

    background:
        linear-gradient(
            135deg,
            #fcfefd 0%,
            #f7fbf8 100%
        );

    display: flex;
    align-items: center;
    justify-content: center;

    text-align: center;
    padding: 1.5rem;

    color: var(--fresh-muted);
}

.fl-result-empty-inner {
    max-width: 330px;
}

.fl-result-empty-icon {
    width: 64px;
    height: 64px;

    margin: 0 auto .8rem;

    border-radius: 50%;

    background: var(--fresh-green-soft);
    color: var(--fresh-green);

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 1.35rem;
    font-weight: 850;
}

.fl-result-empty-title {
    color: #62736b;
    font-size: .92rem;
    font-weight: 550;
    line-height: 1.5;
}


/* =========================================================
   SUCCESS RESULT
   ========================================================= */

.fl-result-card {
    border: 1px solid #d8e9de;
    background: var(--fresh-white);
    border-radius: 15px;
    padding: 1rem;

    box-shadow: 0 5px 18px rgba(22, 53, 42, .055);
}

.fl-result-flex {
    display: flex;
    gap: 1rem;
    align-items: center;
}

.fl-thumb {
    width: 92px;
    height: 92px;
    min-width: 92px;

    object-fit: cover;

    border-radius: 13px;
    border: 1px solid #dce9df;

    background: #f7faf8;
}

.fl-result-info {
    min-width: 0;
    flex: 1;
}

.fl-fruit {
    color: var(--fresh-dark);
    font-size: 1.5rem;
    font-weight: 850;
    line-height: 1.15;
}

.fl-condition {
    color: var(--fresh-green);
    font-size: 1rem;
    font-weight: 700;
    margin-top: .35rem;
}

.fl-badge {
    display: inline-flex;
    align-items: center;

    margin-top: .55rem;

    padding: .3rem .65rem;

    border-radius: 999px;

    background: var(--fresh-green-soft);
    color: var(--fresh-green-dark);

    font-size: .76rem;
    font-weight: 750;
}


/* =========================================================
   METRICS
   ========================================================= */

div[data-testid="stMetric"] {
    background: var(--fresh-white);

    border: 1px solid #e1ebe4;
    border-radius: 12px;

    padding: .7rem .8rem;

    box-shadow: none;
}

div[data-testid="stMetricLabel"] {
    color: var(--fresh-muted) !important;
    font-size: .72rem !important;
}

div[data-testid="stMetricValue"] {
    color: var(--fresh-dark) !important;
    font-size: 1.2rem !important;
    font-weight: 850 !important;
}


/* =========================================================
   PRIMARY BUTTON
   ========================================================= */

div.stButton > button[kind="primary"] {
    background: var(--fresh-red);
    border: 1px solid var(--fresh-red);
    color: #ffffff;

    border-radius: 11px;

    min-height: 44px;

    font-size: .94rem;
    font-weight: 800;

    box-shadow: 0 5px 14px rgba(255, 91, 91, .18);
}

div.stButton > button[kind="primary"]:hover {
    background: #f24d4d;
    border-color: #f24d4d;
    color: #ffffff;
}


/* =========================================================
   SECONDARY BUTTON
   ========================================================= */

button[kind="secondary"] {
    border-radius: 10px;
}


/* =========================================================
   WARNING / ERROR
   ========================================================= */

.fl-warning {
    border: 1px solid #f2c979;
    background: var(--fresh-orange-soft);
    border-radius: 13px;
    padding: 1rem;
    color: #68400e;
}

.fl-warning h3 {
    margin: 0 0 .45rem;
    color: #d17b16;
    font-size: 1.05rem;
}

.fl-error {
    border: 1px solid #f2b8b3;
    background: var(--fresh-red-soft);
    border-radius: 13px;
    padding: 1rem;
    color: #6b201a;
}

.fl-error h3 {
    margin: 0 0 .45rem;
    color: var(--fresh-red);
    font-size: 1.05rem;
}


/* =========================================================
   PHOTO GUIDANCE
   ========================================================= */

.fl-guidance-box {
    margin-top: .8rem;
    padding: .85rem 1rem;

    background: #fffdf7;
    border: 1px solid #ead9a8;
    border-radius: 14px;

    box-shadow: 0 3px 10px rgba(120, 95, 35, .035);
}

.fl-guidance-title {
    color: #5f512d;
    font-size: .92rem;
    font-weight: 850;
    margin-bottom: .6rem;
}

.fl-guidance-list {
    display: flex;
    flex-direction: column;
    gap: .35rem;
}

.fl-guidance-item {
    display: flex;
    align-items: center;
    gap: .6rem;

    padding: .18rem 0;
}

.fl-guidance-number {
    width: 27px;
    height: 27px;
    min-width: 27px;

    display: flex;
    align-items: center;
    justify-content: center;

    border-radius: 50%;

    background: #e8f5ea;
    color: #2f7d50;

    font-size: .75rem;
    font-weight: 850;
}

.fl-guidance-text {
    color: #40534a;
    font-size: .82rem;
    font-weight: 600;
    line-height: 1.25;
}

/* =========================================================
   RESPONSIVE
   ========================================================= */

@media (max-width: 900px) {

    .fl-guidance-grid {
        grid-template-columns: repeat(
            2,
            minmax(0, 1fr)
        );
    }

}

@media (max-width: 640px) {

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .fl-header h1 {
        font-size: 1.55rem;
    }

    .fl-guidance-grid {
        grid-template-columns: 1fr;
    }

    .fl-result-flex {
        align-items: flex-start;
    }

    .fl-thumb {
        width: 72px;
        height: 72px;
        min-width: 72px;
    }

    .fl-result-panel-head {
        align-items: flex-start;
    }

    .fl-step-title {
        font-size: 1.15rem;
    }

}

</style>
"""


def render_diagnosis_page(predictor, show_sidebar: bool = True) -> None:
    """Render the Diagnosis page while preserving the existing inference flow."""
    _init_ui_state()
    _render_styles()
    if show_sidebar:
        _render_sidebar()
    _render_header()

    left, right = st.columns([0.45, 0.55], gap="large")

    with left:
        _render_input_panel()

    image_bytes = st.session_state.get("current_image_bytes")
    image_meta = st.session_state.get("current_image_meta")

    with right:
        _render_result_panel(image_bytes, image_meta, predictor)
        _render_guidance()

    _render_footer()


def _init_ui_state() -> None:
    defaults = {
        "input_source": "upload",
        "current_image_bytes": None,
        "current_image_meta": None,
        "last_image_sha256_ui": None,
        "uploader_version": 0,
        "camera_version": 0,
        "input_source_radio": "upload",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _render_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("<h1>FreshLens</h1>", unsafe_allow_html=True)
        for label, active in [
            ("Chẩn đoán", True),
            ("Giải thích", False),
            ("Đánh giá", False),
            ("About / Team", False),
        ]:
            cls = "fl-nav-item active" if active else "fl-nav-item"
            st.markdown(f'<div class="{cls}">{label}</div>', unsafe_allow_html=True)

def _render_header() -> None:
    st.markdown(
        """
<div class="fl-header">
  <h1>Nhận diện trái cây và tình trạng</h1>
  <p>Hỗ trợ: Táo · Chuối · Cam · Cà chua</p>
</div>
""",
        unsafe_allow_html=True,
    )

def _render_input_panel() -> None:
    st.markdown(
        '<div class="fl-card-title">Ảnh đầu vào</div>',
        unsafe_allow_html=True,
    )

    selected_source = st.radio(
        "Nguồn ảnh",
        ["upload", "camera"],
        format_func=lambda value: "Tải ảnh lên" if value == "upload" else "Chụp bằng camera",
        horizontal=True,
        key="input_source_radio",
        label_visibility="collapsed",
    )

    if selected_source != st.session_state.input_source:
        st.session_state.input_source = selected_source
        _remove_current_image()

    if st.session_state.input_source == "upload":
        _render_upload_control()
        _render_preview()
    else:
        _render_camera_control()

def _render_upload_control() -> None:
    uploaded = st.file_uploader(
        "Upload",
        type=SUPPORTED_TYPES,
        key=f"diagnosis_upload_{st.session_state.uploader_version}",
        label_visibility="collapsed",
    )

    if uploaded is None:
        current_meta = st.session_state.get("current_image_meta") or {}
        if current_meta.get("source") == "upload" and st.session_state.current_image_bytes is not None:
            _remove_current_image(keep_source=True)
        return

    uploaded_bytes = uploaded.getvalue()
    _set_current_image(
        uploaded_bytes,
        {
            "name": getattr(uploaded, "name", "uploaded-image.png"),
            "size": int(getattr(uploaded, "size", 0) or len(uploaded_bytes)),
            "type": getattr(uploaded, "type", None),
            "source": "upload",
        },
    )

def _render_camera_control() -> None:
    captured = st.camera_input(
        "Chụp một quả chính trong khung hình",
        key=f"diagnosis_camera_{st.session_state.camera_version}",
    )

    current_meta = st.session_state.get("current_image_meta") or {}
    if captured is None:
        if current_meta.get("source") == "camera" and st.session_state.current_image_bytes is not None:
            _remove_current_image(keep_source=True)
        return

    captured_bytes = captured.getvalue()
    _set_current_image(
        captured_bytes,
        {
            "name": getattr(captured, "name", "camera_capture.jpg"),
            "size": int(getattr(captured, "size", 0) or len(captured_bytes)),
            "type": getattr(captured, "type", "image/jpeg"),
            "source": "camera",
        },
    )

def _render_preview() -> None:
    image_bytes = st.session_state.get("current_image_bytes")

    if image_bytes is None:
        st.markdown(
            """
<div class="fl-preview-shell">
  <div class="fl-empty-preview">
    <div>
      <div class="fl-empty-title">Chọn một ảnh</div>
      <div class="fl-empty-meta">JPG, PNG, BMP, WEBP<br />≤ 200MB</div>
    </div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    meta = st.session_state.get("current_image_meta") or {}
    name = html.escape(str(meta.get("name") or "selected-image"))
    size = _format_size(int(meta.get("size") or len(image_bytes)))

    title_col, clear_col = st.columns([0.88, 0.12], vertical_alignment="center")
    with title_col:
        st.markdown(
            f'<div class="fl-preview-tools"><span class="fl-preview-title">Ảnh đầu vào</span><span>{name} - {size}</span></div>',
            unsafe_allow_html=True,
        )
    with clear_col:
        if st.button("×", key="clear_upload_preview", help="Xóa ảnh", width="stretch"):
            _remove_current_image()
            st.rerun()

    st.markdown(
        f"""
<div class="fl-preview-shell">
  <div class="fl-preview-image">
    <img src="data:image/jpeg;base64,{_image_base64(image_bytes)}" alt="Ảnh đầu vào" />
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

def _render_result_panel(image_bytes, image_meta, predictor) -> None:
    st.markdown('<div class="fl-card-title">Kết quả</div>', unsafe_allow_html=True)
    if image_bytes is None:
        st.markdown(
            '<div class="fl-empty-preview">Kết quả nhận diện sẽ xuất hiện sau khi bạn chọn ảnh và phân tích.</div>',
            unsafe_allow_html=True,
        )
        return

    if st.button("Phân tích ảnh", type="primary", width="stretch"):
        _analyze_current_image(image_bytes, predictor)

    result = result_for_image(st.session_state, image_bytes)
    if not result:
        st.info("Nhấn Phân tích ảnh để bắt đầu chẩn đoán ảnh hiện tại.")
        return

    status = _result_status(result)
    if status == "success":
        _render_success_result(result, image_bytes)
    elif status == "quality_rejected":
        _render_quality_rejection(result)
    else:
        _render_openset_rejection(result)



def _analyze_current_image(image_bytes, predictor) -> None:
    try:
        with st.spinner("Đang phân tích..."):
            start_time = time.perf_counter()

            analyzed = dict(
                predictor.predict(
                    image_bytes,
                    use_tta=False,
                    check_quality=True,
                )
            )

            measured_latency_ms = (time.perf_counter() - start_time) * 1000
            analyzed["latency_ms"] = (
                analyzed.get("latency_ms") or measured_latency_ms
            )

            store_result(st.session_state, image_bytes, analyzed)

    except Exception as exc:
        clear_result(st.session_state)
        st.error(f"Không đọc/nhận diện được ảnh: {exc}")


def _render_success_result(result: dict[str, Any], image_bytes: bytes) -> None:
    fruit = _display_fruit(result.get("fruit"))
    condition = _display_condition(result.get("condition"))
    st.markdown('<div class="fl-result-head">✓ Kết quả nhận diện</div>', unsafe_allow_html=True)
    fruit_html = html.escape(str(fruit))
    condition_html = html.escape(str(condition))
    st.markdown(
        f"""
<div class="fl-result-card">
  <div class="fl-result-flex">
    <img class="fl-thumb" src="data:image/jpeg;base64,{_image_base64(image_bytes)}" alt="Ảnh đầu vào" />
    <div>
      <div class="fl-fruit">{fruit_html}</div>
      <div class="fl-condition">Tình trạng: <b>{condition_html}</b></div>
      <div class="fl-badge">✓ Đã nhận diện</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    fruit_confidence = result.get("fruit_confidence", result.get("fruit_score"))
    condition_confidence = result.get(
        "condition_confidence",
        result.get("condition_score_given_fruit"),
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Độ tin cậy loại quả", _percent(fruit_confidence))
    c2.metric("Độ tin cậy tình trạng", _percent(condition_confidence))
    c3.metric("Latency", _latency(result.get("latency_ms")))


def _render_quality_rejection(result: dict[str, Any]) -> None:
    reason = result.get("rejection_reason") or "Ảnh chưa đạt chất lượng. Vui lòng chụp lại rõ nét và đủ sáng."
    st.markdown(
        "<div class=\"fl-warning\"><h3>Ảnh chưa đạt chất lượng</h3></div>",
        unsafe_allow_html=True,
    )
    st.warning(str(reason))
    _render_latency_caption(result)


def _render_openset_rejection(result: dict[str, Any]) -> None:
    reason = result.get("rejection_reason") or "Hệ thống hiện hỗ trợ Táo, Chuối, Cam và Cà chua."
    st.markdown(
        "<div class=\"fl-error\"><h3>Ảnh ngoài phạm vi nhận diện</h3></div>",
        unsafe_allow_html=True,
    )
    st.error(str(reason))
    _render_latency_caption(result)


def _render_technical_details(result: dict[str, Any]) -> None:
    fields = {
        "fruit": result.get("fruit"),
        "condition": result.get("condition"),
        "fruit confidence": result.get("fruit_score"),
        "condition confidence": result.get("condition_score_given_fruit"),
        "joint confidence": result.get("joint_confidence"),
        "latency": result.get("latency_ms"),
        "rejection type": result.get("rejection_type"),
        "rejection reason": result.get("rejection_reason"),
        "supported": result.get("supported"),
        "is_supported": result.get("is_supported"),
        "support probability": result.get("support_probability"),
        "support threshold": result.get("support_threshold"),
        "fruit_scores": result.get("fruit_scores"),
        "gate_detail": result.get("gate_detail"),
    }
    details = {key: value for key, value in fields.items() if value is not None}
    if details:
        with st.expander("Chi tiết kỹ thuật"):
            st.json(details)


def _render_guidance() -> None:
    st.write("")

    items = []

    for index, item in enumerate(GUIDANCE, 1):
        items.append(
            f"""
            <div class="fl-guidance-item">
                <div class="fl-guidance-number">{index}</div>
                <div class="fl-guidance-text">
                    {html.escape(item)}
                </div>
            </div>
            """
        )

    st.html(
        f"""
        <div class="fl-guidance-box">

            <div class="fl-guidance-title">
                Hướng dẫn chụp ảnh
            </div>

            <div class="fl-guidance-list">
                {"".join(items)}
            </div>

        </div>
        """
    )

def _render_footer() -> None:
    st.divider()
    st.caption(
        "FreshLens đánh giá dấu hiệu nhìn thấy trong ảnh. Kết quả không thay thế kiểm nghiệm an toàn thực phẩm."
    )


def _set_current_image(image_bytes: bytes, meta: dict[str, Any]) -> None:
    image_hash = sha256_bytes(image_bytes)
    if st.session_state.get("last_image_sha256_ui") != image_hash:
        clear_result(st.session_state)
        st.session_state.current_image_bytes = image_bytes
        st.session_state.current_image_meta = meta
        st.session_state.last_image_sha256_ui = image_hash


def _remove_current_image(keep_source: bool = False) -> None:
    source = st.session_state.get("input_source")
    st.session_state.current_image_bytes = None
    st.session_state.current_image_meta = None
    st.session_state.last_image_sha256_ui = None
    clear_result(st.session_state)
    if not keep_source:
        st.session_state.uploader_version += 1
        st.session_state.camera_version += 1
    st.session_state.input_source = source


def _result_status(result: dict[str, Any]) -> str:
    status = result.get("status")
    if status in {"success", "quality_rejected", "unsupported"}:
        return status

    rejection_type = result.get("rejection_type")
    if rejection_type in {"quality_rejection", "quality_rejected"}:
        return "quality_rejected"
    if rejection_type in {"openset_rejection", "open_set_rejection", "unsupported"}:
        return "unsupported"
    if result.get("is_supported") is False or result.get("supported") is False:
        return "unsupported"
    return "success"


def _display_fruit(value: Any) -> Any:
    return FRUIT_VI.get(str(value).lower(), value)


def _display_condition(value: Any) -> Any:
    return STATUS_VI.get(str(value).lower(), value)


def _format_size(size: int) -> str:
    if size < 1024:
        return f"{size}B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f}KB"
    return f"{size / (1024 * 1024):.1f}MB"


def _extension_label(name: str) -> str:
    return name.rsplit(".", 1)[-1].upper() if "." in name else "image"


def _percent(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1%}"
    return "-"


def _latency(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.0f} ms"
    return "-"


def _render_latency_caption(result: dict[str, Any]) -> None:
    if result.get("latency_ms") is not None:
        st.caption(f"Latency: {_latency(result.get('latency_ms'))}")


def _image_base64(image_bytes: bytes) -> str:
    import base64

    return base64.b64encode(image_bytes).decode("ascii")