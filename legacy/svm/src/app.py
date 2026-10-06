"""FreshLens Streamlit application — Professional UI Edition.

Run from the project root with: ``streamlit run src/app.py``
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import streamlit as st
except Exception:  # pragma: no cover
    st = None  # type: ignore[assignment]

try:
    from .config import DATASET_ROOT, METRICS_PATH, MODEL_PATH
    from .features import diagnostic_views, extract_features
    from .image_processing import bgr_to_rgb
    from .model import json_safe
    from .predict import FreshLensPredictor
except ImportError:
    from src.config import DATASET_ROOT, METRICS_PATH, MODEL_PATH
    from src.features import diagnostic_views, extract_features
    from src.image_processing import bgr_to_rgb
    from src.model import json_safe
    from src.predict import FreshLensPredictor


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

FRUIT_ICONS = {
    "apple":  "🍎",
    "banana": "🍌",
    "orange": "🍊",
    "tomato": "🍅",
    "other":  "❓",
}
STATUS_ICONS   = {"fresh": "✅", "rotten": "🟤"}
STATUS_COLORS  = {"fresh": "#22d3a0", "rotten": "#f97316"}


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _model_path() -> Path:
    return Path(os.getenv("FRESHLENS_MODEL_PATH", str(MODEL_PATH))).expanduser()


# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────

def _inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

        *, *::before, *::after { box-sizing: border-box; }
        html, body, .stApp { font-family: 'Inter', sans-serif; }

        /* Background */
        .stApp {
            background:
                radial-gradient(ellipse 70% 55% at 80% -10%, rgba(16,185,129,.18) 0%, transparent 60%),
                radial-gradient(ellipse 50% 40% at -10% 90%, rgba(6,78,59,.25) 0%, transparent 55%),
                linear-gradient(160deg, #050f0d 0%, #071a16 40%, #04120f 100%);
            color: #e2f5ee;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #071f1a 0%, #041310 100%) !important;
            border-right: 1px solid rgba(16,185,129,.18) !important;
        }
        [data-testid="stSidebar"] * { color: #c8e8dc !important; }

        /* Header */
        header[data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }

        /* Scrollbar */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: #071a16; }
        ::-webkit-scrollbar-thumb { background: #1d6b52; border-radius: 4px; }

        /* Keyframes */
        @keyframes gradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50%       { transform: translateY(-7px); }
        }
        @keyframes fadeSlideIn {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        @keyframes shimmer {
            0%   { background-position: -800px 0; }
            100% { background-position: 800px 0; }
        }

        /* Hero */
        .hero-banner {
            background: linear-gradient(270deg, #064e3b, #065f46, #047857, #059669, #064e3b);
            background-size: 400% 400%;
            animation: gradientShift 8s ease infinite;
            border-radius: 24px;
            padding: 36px 44px;
            margin-bottom: 28px;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(52,211,153,.25);
            box-shadow: 0 24px 80px rgba(0,0,0,.45), 0 0 0 1px rgba(52,211,153,.08);
        }
        .hero-banner::before {
            content: '';
            position: absolute; inset: 0;
            background: radial-gradient(circle at 85% 50%, rgba(52,211,153,.12) 0%, transparent 55%);
            pointer-events: none;
        }
        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.8rem; font-weight: 800;
            letter-spacing: -.05em; color: #ecfdf5;
            margin: 0 0 8px; line-height: 1.1;
        }
        .hero-title span {
            background: linear-gradient(90deg, #6ee7b7, #34d399, #10b981);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        .hero-sub { color: #a7f3d0; font-size: 1.05rem; font-weight: 400; margin: 0; line-height: 1.6; }
        .hero-badges { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }
        .badge {
            display: inline-flex; align-items: center; gap: 6px;
            background: rgba(6,78,59,.65); border: 1px solid rgba(52,211,153,.30);
            border-radius: 50px; padding: 5px 14px;
            font-size: .78rem; font-weight: 600; color: #6ee7b7;
            letter-spacing: .04em;
        }
        .hero-icon {
            position: absolute; right: 44px; top: 50%; transform: translateY(-50%);
            font-size: 5.5rem; line-height: 1;
            animation: float 3.5s ease-in-out infinite;
            filter: drop-shadow(0 10px 30px rgba(16,185,129,.5));
            opacity: .85;
        }

        /* Section title */
        .section-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.1rem; font-weight: 700; color: #6ee7b7;
            letter-spacing: -.01em; margin: 28px 0 14px;
            display: flex; align-items: center; gap: 10px;
        }
        .section-title::after {
            content: ''; flex: 1; height: 1px;
            background: linear-gradient(90deg, rgba(52,211,153,.35) 0%, transparent 100%);
        }

        /* Metric cards */
        .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 16px 0; }
        .metric-card {
            background: linear-gradient(135deg, rgba(6,78,59,.55) 0%, rgba(4,47,36,.65) 100%);
            border: 1px solid rgba(52,211,153,.22); border-radius: 20px;
            padding: 22px 20px; position: relative; overflow: hidden;
            transition: transform .25s, box-shadow .25s, border-color .25s;
            animation: fadeSlideIn .5s ease both;
        }
        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 20px 50px rgba(0,0,0,.35), 0 0 30px rgba(16,185,129,.12);
            border-color: rgba(52,211,153,.42);
        }
        .metric-card::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, #059669, #34d399, #6ee7b7);
            border-radius: 20px 20px 0 0;
        }
        .metric-card-icon { font-size: 1.9rem; margin-bottom: 10px; animation: float 4s ease-in-out infinite; }
        .metric-label { font-size: .72rem; font-weight: 600; text-transform: uppercase; letter-spacing: .10em; color: #6ee7b7; margin-bottom: 6px; }
        .metric-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 700; color: #ecfdf5; line-height: 1.15; }

        /* Result banners */
        .result-accepted {
            background: linear-gradient(135deg, rgba(6,78,59,.7), rgba(4,47,36,.8));
            border: 1px solid #34d399; border-radius: 18px; padding: 20px 24px;
            display: flex; align-items: center; gap: 16px;
            animation: fadeSlideIn .4s ease; box-shadow: 0 0 40px rgba(52,211,153,.12);
        }
        .result-warning {
            background: linear-gradient(135deg, rgba(120,53,15,.55), rgba(92,38,8,.7));
            border: 1px solid #f97316; border-radius: 18px; padding: 20px 24px;
            display: flex; align-items: center; gap: 16px; animation: fadeSlideIn .4s ease;
        }
        .result-error {
            background: linear-gradient(135deg, rgba(127,29,29,.55), rgba(69,10,10,.7));
            border: 1px solid #ef4444; border-radius: 18px; padding: 20px 24px;
            animation: fadeSlideIn .4s ease;
        }
        .result-icon { font-size: 2.8rem; flex-shrink: 0; }
        .result-state { font-family: 'Space Grotesk', sans-serif; font-size: 1.25rem; font-weight: 700; color: #ecfdf5; }
        .result-detail { font-size: .88rem; color: #a7f3d0; margin-top: 4px; }

        /* Progress bar */
        .progress-wrap {
            background: rgba(6,78,59,.35); border-radius: 50px; height: 12px;
            margin: 14px 0 4px; overflow: hidden; border: 1px solid rgba(52,211,153,.18);
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #059669, #34d399, #6ee7b7);
            border-radius: 50px; transition: width .8s cubic-bezier(.4,0,.2,1); position: relative;
        }
        .progress-fill::after {
            content: ''; position: absolute; inset: 0;
            background: linear-gradient(90deg, transparent 0%, rgba(255,255,255,.25) 50%, transparent 100%);
            background-size: 200px 100%; animation: shimmer 2s infinite;
        }
        .progress-label { display: flex; justify-content: space-between; font-size: .78rem; color: #6ee7b7; margin-top: 4px; }

        /* Probability table */
        .prob-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        .prob-table th {
            text-align: left; padding: 8px 12px; font-size: .72rem; font-weight: 600;
            text-transform: uppercase; letter-spacing: .08em; color: #6ee7b7;
            border-bottom: 1px solid rgba(52,211,153,.2);
        }
        .prob-table td { padding: 10px 12px; font-size: .88rem; color: #d1fae5; border-bottom: 1px solid rgba(52,211,153,.08); }
        .prob-table tr:last-child td { border-bottom: none; }
        .prob-table tr:hover td { background: rgba(16,185,129,.06); }
        .prob-mini-bar { background: rgba(6,78,59,.45); border-radius: 4px; height: 6px; overflow: hidden; width: 80px; display: inline-block; vertical-align: middle; margin-left: 8px; }
        .prob-mini-fill { height: 100%; background: linear-gradient(90deg, #059669, #34d399); border-radius: 4px; }

        /* File uploader */
        [data-testid="stFileUploader"] {
            border: 2px dashed rgba(52,211,153,.35) !important; border-radius: 18px !important;
            background: rgba(6,78,59,.15) !important; transition: border-color .25s, background .25s !important;
        }
        [data-testid="stFileUploader"]:hover { border-color: rgba(52,211,153,.65) !important; background: rgba(6,78,59,.25) !important; }

        /* Camera */
        [data-testid="stCameraInput"] { border-radius: 18px !important; overflow: hidden; }

        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, #059669, #047857) !important;
            color: #ecfdf5 !important; border: 1px solid rgba(52,211,153,.35) !important;
            border-radius: 14px !important; font-weight: 600 !important; letter-spacing: .02em !important;
            transition: all .25s !important; box-shadow: 0 4px 20px rgba(5,150,105,.3) !important;
        }
        .stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 30px rgba(5,150,105,.45) !important; }

        /* Expander */
        div[data-testid="stExpander"] { background: rgba(6,78,59,.18) !important; border: 1px solid rgba(52,211,153,.2) !important; border-radius: 16px !important; }

        /* Images */
        [data-testid="stImage"] img { border-radius: 14px !important; border: 1px solid rgba(52,211,153,.18) !important; box-shadow: 0 8px 30px rgba(0,0,0,.3) !important; }

        /* Info / warning boxes */
        .info-box {
            background: rgba(6,78,59,.35); border: 1px solid rgba(52,211,153,.25);
            border-left: 4px solid #10b981; border-radius: 12px; padding: 14px 18px;
            font-size: .88rem; color: #a7f3d0; margin: 12px 0; animation: fadeSlideIn .4s ease;
        }
        .warn-box {
            background: rgba(120,53,15,.3); border: 1px solid rgba(249,115,22,.35);
            border-left: 4px solid #f97316; border-radius: 12px; padding: 14px 18px;
            font-size: .88rem; color: #fed7aa; margin: 12px 0;
        }
        .err-box {
            background: rgba(127,29,29,.35); border: 1px solid rgba(239,68,68,.35);
            border-left: 4px solid #ef4444; border-radius: 12px; padding: 14px 18px;
            font-size: .88rem; color: #fca5a5; margin: 12px 0;
        }

        /* Diagnostic captions */
        .diag-caption { text-align: center; font-size: .75rem; color: #6ee7b7; margin-top: 6px; font-weight: 500; }

        /* Sidebar logo */
        .sidebar-logo { display: flex; align-items: center; gap: 10px; padding: 8px 0 20px; }
        .sidebar-logo-icon { font-size: 2rem; animation: float 4s ease-in-out infinite; filter: drop-shadow(0 0 12px rgba(16,185,129,.6)); }
        .sidebar-logo-text { font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 800; color: #ecfdf5 !important; letter-spacing: -.04em; }
        .sidebar-logo-text span { color: #34d399 !important; }

        /* Evaluation cards */
        .eval-card {
            background: linear-gradient(135deg, rgba(6,78,59,.5), rgba(4,47,36,.6));
            border: 1px solid rgba(52,211,153,.2); border-radius: 20px; padding: 20px;
            text-align: center; transition: transform .25s, box-shadow .25s;
        }
        .eval-card:hover { transform: translateY(-4px); box-shadow: 0 16px 40px rgba(0,0,0,.3); }
        .eval-num {
            font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 800;
            background: linear-gradient(90deg, #6ee7b7, #34d399);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        .eval-label { font-size: .75rem; font-weight: 600; text-transform: uppercase; letter-spacing: .09em; color: #6ee7b7; margin-top: 4px; }

        /* Team table */
        .team-table { width: 100%; border-collapse: collapse; }
        .team-table th { background: rgba(6,78,59,.55); padding: 12px 16px; text-align: left; font-size: .78rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: #6ee7b7; border-bottom: 1px solid rgba(52,211,153,.25); }
        .team-table td { padding: 14px 16px; font-size: .88rem; color: #d1fae5; border-bottom: 1px solid rgba(52,211,153,.1); }
        .team-table tr:last-child td { border-bottom: none; }
        .team-table tr:hover td { background: rgba(16,185,129,.06); }
        .member-badge { display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; background: linear-gradient(135deg, #059669, #34d399); border-radius: 50%; font-size: .8rem; font-weight: 700; color: #064e3b; }

        /* Misc overrides */
        .stMarkdown p { color: #c8e8dc; }
        .stCaption { color: #6ee7b7 !important; }
        label[data-testid="stWidgetLabel"] { color: #a7f3d0 !important; }
        .stJson { background: rgba(4,47,36,.7) !important; border-radius: 12px !important; }
        [data-testid="stDownloadButton"] button { background: rgba(6,78,59,.5) !important; border: 1px solid rgba(52,211,153,.35) !important; border-radius: 10px !important; color: #6ee7b7 !important; }
        .stCode, code { background: rgba(4,47,36,.8) !important; border: 1px solid rgba(52,211,153,.2) !important; border-radius: 12px !important; }
        hr { border-color: rgba(52,211,153,.18) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Hero banner
# ─────────────────────────────────────────────

def _hero(page: str = "Chẩn đoán") -> None:
    icons_map = {
        "Chẩn đoán":       ("🍏", "Chẩn đoán thực phẩm",  "Nhận dạng táo · chuối · cam · cà chua và dấu hiệu tươi / hỏng từ một ảnh"),
        "Phòng học thuật": ("🔬", "Phòng học thuật",       "Xem ảnh trung gian, vector đặc trưng và hợp đồng mô hình theo từng bước"),
        "Đánh giá":        ("📊", "Đánh giá mô hình",      "So sánh baseline / candidate và xem các chỉ số test cuối"),
        "Phân công nhóm":  ("👥", "Phân công nhóm",        "Bàn giao 4 thành viên và lệnh chạy nhanh cho từng bước"),
    }
    icon, title, sub = icons_map.get(page, ("🍏", page, ""))
    badges = ["SVM Kernel RBF", "HOG + LBP + Màu", "Cơ chế từ chối", "OpenCV · Scikit-learn"]
    badge_html = "".join(f'<span class="badge">⬡ {b}</span>' for b in badges)
    st.markdown(
        f"""
        <div class="hero-banner">
          <div class="hero-icon">{icon}</div>
          <p class="hero-title">Fresh<span>Lens</span></p>
          <h1 style="margin:0 0 6px;font-family:'Space Grotesk',sans-serif;font-size:1.25rem;
                     font-weight:600;color:#a7f3d0;">{title}</h1>
          <p class="hero-sub">{sub}</p>
          <div class="hero-badges">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def _load_predictor(path: Path) -> FreshLensPredictor:
    return _cached_predictor(str(path), path.stat().st_mtime_ns)


if st is not None:
    @st.cache_resource(show_spinner=False)
    def _cached_predictor(path_string: str, modified_ns: int) -> FreshLensPredictor:
        return FreshLensPredictor(Path(path_string))
else:  # pragma: no cover
    def _cached_predictor(path_string: str, modified_ns: int) -> FreshLensPredictor:
        return FreshLensPredictor(Path(path_string))


# ─────────────────────────────────────────────
# Helper components
# ─────────────────────────────────────────────

def _progress_bar(value: float, label_left: str = "", label_right: str = "") -> None:
    pct = min(max(value, 0.0), 1.0) * 100
    st.markdown(
        f"""
        <div class="progress-wrap">
          <div class="progress-fill" style="width:{pct:.1f}%"></div>
        </div>
        <div class="progress-label"><span>{label_left}</span><span>{label_right}</span></div>
        """,
        unsafe_allow_html=True,
    )


def _metric_cards(items: list[tuple[str, str, str]]) -> None:
    """items = [(icon, label, value), ...]"""
    cols_html = ""
    for i, (icon, label, value) in enumerate(items):
        delay = i * 0.1
        cols_html += f"""
        <div class="metric-card" style="animation-delay:{delay:.1f}s">
          <div class="metric-card-icon">{icon}</div>
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
        </div>"""
    st.markdown(f'<div class="metric-grid">{cols_html}</div>', unsafe_allow_html=True)


def _section(title: str, emoji: str = "") -> None:
    prefix = emoji + " " if emoji else ""
    st.markdown(f'<div class="section-title">{prefix}{title}</div>', unsafe_allow_html=True)


def _info(msg: str) -> None:
    st.markdown(f'<div class="info-box">ℹ️ {msg}</div>', unsafe_allow_html=True)


def _warn(msg: str) -> None:
    st.markdown(f'<div class="warn-box">⚠️ {msg}</div>', unsafe_allow_html=True)


def _err(msg: str) -> None:
    st.markdown(f'<div class="err-box">❌ {msg}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Page: Chẩn đoán
# ─────────────────────────────────────────────

def _render_input_and_prediction(model_path: Path) -> tuple[Any | None, Any | None]:
    _section("Đưa ảnh vào", "📷")
    left, right = st.columns(2, gap="medium")
    with left:
        uploaded = st.file_uploader(
            "Tải ảnh sản phẩm lên",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="upload",
            help="Hỗ trợ JPG, PNG, WEBP, BMP",
        )
    with right:
        camera = st.camera_input("Chụp trực tiếp bằng camera", key="camera")

    source = uploaded if uploaded is not None else camera
    if source is None:
        _info("Hãy tải ảnh hoặc chụp một ảnh. Nên chụp một quả chính, đủ sáng, không che khuất.")
        return None, None

    payload = source.getvalue()
    digest = hashlib.sha256(payload).hexdigest()
    if st.session_state.get("input_hash") != digest:
        for key in ("input_hash", "prediction", "processed", "prediction_error"):
            st.session_state.pop(key, None)
        st.session_state["input_hash"] = digest

    if "prediction" not in st.session_state and "prediction_error" not in st.session_state:
        if not model_path.exists():
            st.session_state["prediction_error"] = (
                f"Chưa có mô hình tại {model_path}. Hãy chạy lệnh train trong README."
            )
        else:
            with st.spinner("🔍 Đang phân tích ảnh..."):
                try:
                    predictor = _load_predictor(model_path)
                    result, processed = predictor.analyze(payload)
                    st.session_state["prediction"] = result
                    st.session_state["processed"] = processed
                except Exception as exc:  # noqa: BLE001
                    st.session_state["prediction_error"] = f"{type(exc).__name__}: {exc}"

    _section("Ảnh đầu vào", "🖼️")
    img_data = (
        bgr_to_rgb(st.session_state["processed"].original_bgr)
        if st.session_state.get("processed") is not None
        else payload
    )
    st.image(img_data, caption="Ảnh được đưa vào hệ thống", use_container_width=True)

    if st.session_state.get("prediction_error"):
        _err(st.session_state["prediction_error"])
        return None, st.session_state.get("processed")

    result = st.session_state.get("prediction")
    if result is None:
        return None, st.session_state.get("processed")

    _section("Kết quả chẩn đoán", "🎯")

    fruit_icon   = FRUIT_ICONS.get(result.fruit or "", "❓")
    status_icon  = STATUS_ICONS.get(result.status or "", "—")
    status_color = STATUS_COLORS.get(result.status or "", "#9ca3af")

    if result.state == "ACCEPTED":
        st.markdown(
            f"""
            <div class="result-accepted">
              <div class="result-icon">{fruit_icon}</div>
              <div>
                <div class="result-state">✅ {result.state_label} — {result.fruit_label or "—"}</div>
                <div class="result-detail">
                  Tình trạng: <strong style="color:{status_color}">{status_icon} {result.status_label or "Đang xác định"}</strong>
                  &nbsp;·&nbsp; Mô hình v{result.model_version or "?"}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif result.state == "OUT_OF_SCOPE":
        st.markdown(
            f"""
            <div class="result-warning">
              <div class="result-icon">🚫</div>
              <div>
                <div class="result-state">⚠️ {result.state_label}</div>
                <div class="result-detail">Không khẳng định loại quả hay tình trạng.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-warning">
              <div class="result-icon">🔍</div>
              <div>
                <div class="result-state">⚠️ {result.state_label}</div>
                <div class="result-detail">Hãy thử ảnh rõ hơn hoặc đặt vật thể trên nền đơn giản.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    _metric_cards([
        (fruit_icon,  "Loại quả",    result.fruit_label or "Chưa chấp nhận"),
        ("🎯",         "Độ tin cậy",  f"{(result.fruit_confidence or 0):.1%}"),
        (status_icon, "Tình trạng",  result.status_label or "Không kết luận"),
    ])

    conf = float(result.fruit_confidence or 0.0)
    _progress_bar(conf, "Độ tin cậy loại quả", f"{conf:.1%}")

    st.markdown(
        f'<div class="info-box" style="margin-top:16px">💬 <strong>Lý do:</strong> {result.reason}</div>',
        unsafe_allow_html=True,
    )

    _section("Top xác suất các lớp", "📈")
    rows_html = ""
    for item in result.top_fruit_probabilities:
        pct = item["probability"] * 100
        rows_html += f"""
        <tr>
          <td>{FRUIT_ICONS.get(item['class'], '❓')} {item['label']}</td>
          <td style="color:#6ee7b7;font-family:'Space Grotesk',sans-serif;font-weight:600">{item['probability']:.2%}</td>
          <td><div class="prob-mini-bar"><div class="prob-mini-fill" style="width:{pct:.1f}%"></div></div></td>
        </tr>"""
    st.markdown(
        f"""
        <table class="prob-table">
          <thead><tr><th>Lớp</th><th>Xác suất</th><th>Biểu đồ</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("🔧 Chi tiết kỹ thuật & JSON kết quả"):
        st.json(json_safe(result.to_dict()))
        st.download_button(
            "⬇️ Tải JSON kết quả",
            data=json.dumps(json_safe(result.to_dict()), ensure_ascii=False, indent=2),
            file_name=f"freshlens_{result.image_sha256[:12] if result.image_sha256 else 'result'}.json",
            mime="application/json",
        )

    return result, st.session_state.get("processed")


# ─────────────────────────────────────────────
# Page: Phòng học thuật
# ─────────────────────────────────────────────

def _render_diagnostics(processed: Any | None) -> None:
    _section("Ảnh trung gian", "🧪")
    if processed is None:
        _info("Hãy phân tích một ảnh trước để xem các bước xử lý.")
        return
    views = diagnostic_views(processed)

    items = [
        ("Letterbox 224×224", processed.letterboxed_bgr, True),
        ("CLAHE (Lab-L)",     processed.clahe_bgr,       True),
        ("Gaussian 3×3",      processed.gaussian_bgr,    True),
        ("Canny (biên)",      processed.edges,           False),
    ]
    cols = st.columns(4, gap="small")
    for col, (caption, image, is_bgr) in zip(cols, items):
        with col:
            st.image(bgr_to_rgb(image) if is_bgr else image, use_container_width=True)
            st.markdown(f'<div class="diag-caption">{caption}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cols2 = st.columns(3, gap="small")
    with cols2[0]:
        st.image(views["gray"], use_container_width=True)
        st.markdown('<div class="diag-caption">Ảnh xám</div>', unsafe_allow_html=True)
    with cols2[1]:
        st.image(views["lbp"], use_container_width=True)
        st.markdown('<div class="diag-caption">LBP P=16 R=2</div>', unsafe_allow_html=True)
    with cols2[2]:
        st.image(views["gradient_magnitude"], use_container_width=True)
        st.markdown('<div class="diag-caption">Độ lớn gradient</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="info-box" style="font-size:.78rem;margin-top:12px">'
        '🔬 Canny/gradient/LBP là ảnh trung gian mô tả biến thiên và kết cấu; không phải mặt nạ vùng hỏng.'
        '</div>',
        unsafe_allow_html=True,
    )

    _section("Vector đặc trưng", "📐")
    dims = views["feature_dimensions"]
    total = sum(dims.values())
    dim_cols = st.columns(len(dims))
    for col, (name, dim) in zip(dim_cols, dims.items()):
        with col:
            st.markdown(
                f'<div class="eval-card"><div class="eval-num">{dim}</div><div class="eval-label">{name}</div></div>',
                unsafe_allow_html=True,
            )
    st.markdown(
        f'<div class="info-box" style="margin-top:12px">🔢 Tổng số chiều: <strong style="color:#34d399">{total}</strong></div>',
        unsafe_allow_html=True,
    )
    with st.expander("📋 Quality & chi tiết JSON"):
        st.json({"feature_dimensions": dims, "total": total, "quality": processed.quality})


def _render_learning(processed: Any | None, model_path: Path) -> None:
    _render_diagnostics(processed)
    if model_path.exists():
        try:
            predictor = _load_predictor(model_path)
            vector = extract_features(processed, predictor.feature_config) if processed is not None else None
            _section("Hợp đồng mô hình", "📜")
            st.json({
                "run_id":         predictor.model_run_id,
                "model_version":  predictor.model_version,
                "feature_order":  predictor.artifact.get("feature_order", []),
                "feature_dim":    predictor.artifact.get("feature_dim"),
                "decision":       predictor.artifact.get("decision"),
                "vector_preview": vector[:32].tolist() if vector is not None else None,
            })
        except Exception as exc:  # noqa: BLE001
            _err(f"Artifact không tương thích: {type(exc).__name__}: {exc}")


# ─────────────────────────────────────────────
# Page: Đánh giá
# ─────────────────────────────────────────────

def _render_evaluation() -> None:
    if not METRICS_PATH.exists():
        _info(f"Chưa có báo cáo tại {METRICS_PATH}. Hãy chạy train hoặc evaluate.")
        return
    try:
        report = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        _err(f"Không đọc được báo cáo: {exc}")
        return

    selection = report.get("selection_results", [])
    if selection:
        _section("So sánh Baseline vs Candidate", "⚖️")
        rows_html = ""
        for i, item in enumerate(selection):
            score = item.get("selection_score")
            score_str = f"{float(score):.4f}" if score is not None else "—"
            highlight = "color:#34d399;font-weight:700" if i == 0 else ""
            rows_html += f"""
            <tr>
              <td style="{highlight}">{"🏆 " if i == 0 else ""}{item.get("name", "—")}</td>
              <td style="text-align:center">{item.get("feature_dim", "—")}</td>
              <td style="text-align:center">{item.get("kernel", "—")}</td>
              <td style="text-align:center;{highlight}">{score_str}</td>
            </tr>"""
        st.markdown(
            f"""
            <table class="prob-table">
              <thead><tr><th>Thí nghiệm</th><th>Số chiều</th><th>Kernel</th><th>Điểm validation</th></tr></thead>
              <tbody>{rows_html}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )

    test = report.get("test", {})
    if test:
        fruit = test.get("fruit", test)
        _section("Chỉ số Test cuối", "🏁")
        coverage = float(fruit.get("coverage_in_scope", 0))
        accuracy = float(fruit.get("accuracy_on_accepted_in_scope") or 0)
        far_raw = fruit.get("false_acceptance_rate_other")
        far = float(far_raw) if far_raw is not None else None

        cols = st.columns(3, gap="medium")
        with cols[0]:
            st.markdown(f'<div class="eval-card"><div class="eval-num">{coverage:.1%}</div><div class="eval-label">Coverage trong phạm vi</div></div>', unsafe_allow_html=True)
        with cols[1]:
            st.markdown(f'<div class="eval-card"><div class="eval-num">{accuracy:.1%}</div><div class="eval-label">Đúng trên ảnh được nhận</div></div>', unsafe_allow_html=True)
        with cols[2]:
            far_display = "—" if far is None else f"{far:.1%}"
            st.markdown(f'<div class="eval-card"><div class="eval-num">{far_display}</div><div class="eval-label">Nhận nhầm ngoài phạm vi</div></div>', unsafe_allow_html=True)
        _progress_bar(coverage, "Coverage", f"{coverage:.1%}")

    with st.expander("📋 Toàn bộ JSON báo cáo"):
        st.json(report)


# ─────────────────────────────────────────────
# Page: Phân công nhóm
# ─────────────────────────────────────────────

def _render_team() -> None:
    members = [
        ("TV1", "Dữ liệu · group split · tiền xử lý",      "`dataset.py` · `check_dataset.py` · `image_processing.py`"),
        ("TV2", "Màu · texture · shape · HOG · ablation",   "`features.py`"),
        ("TV3", "SVM · ngưỡng · lớp other · model artifact","`model.py` · `train.py` · `predict.py`"),
        ("TV4", "UI · nghiệp vụ · metrics · test độc lập",  "`app.py` · `metrics.py` · `tests/`"),
    ]
    _section("Bàn giao 4 thành viên", "👥")
    rows_html = ""
    for num, (member, task, files) in enumerate(members, 1):
        rows_html += f"""
        <tr>
          <td><div class="member-badge">{num}</div></td>
          <td style="font-weight:600;color:#ecfdf5">{member}</td>
          <td style="color:#a7f3d0">{task}</td>
          <td style="font-size:.8rem;color:#6ee7b7">{files}</td>
        </tr>"""
    st.markdown(
        f"""
        <table class="team-table">
          <thead><tr><th>#</th><th>Thành viên</th><th>Phụ trách</th><th>File chính</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)
    _warn("<em>fresh / rotten</em> mô tả nhãn dấu hiệu bề ngoài theo dataset, <strong>không phải</strong> kết luận trái cây ăn được hay an toàn thực phẩm.")
    _section("Lệnh chạy nhanh", "⚡")
    st.code(
        "# 1. Tạo demo dataset\n"
        "python scripts\\generate_demo_dataset.py\n\n"
        "# 2. Quét & kiểm tra dữ liệu\n"
        "python -m src.dataset --root data/demo\n"
        "python -m src.check_dataset --root data/demo\n\n"
        "# 3. Train mô hình\n"
        "python -m src.train --root data/demo --manifest data/demo/manifest.csv \\\n"
        "    --output artifacts/freshlens_model.joblib\n\n"
        "# 4. Chạy ứng dụng\n"
        "streamlit run src/app.py",
        language="powershell",
    )


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def _render_sidebar(model_path: Path) -> str:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
              <span class="sidebar-logo-icon">🍏</span>
              <span class="sidebar-logo-text">Fresh<span>Lens</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        page = st.radio(
            "Điều hướng",
            ["Chẩn đoán", "Phòng học thuật", "Đánh giá", "Phân công nhóm"],
            label_visibility="collapsed",
        )
        st.divider()
        model_ok    = model_path.exists()
        status_color = "#34d399" if model_ok else "#f97316"
        status_icon  = "🟢" if model_ok else "🟡"
        status_text  = "Sẵn sàng" if model_ok else "Chưa train"
        st.markdown(
            f"""
            <div style="background:rgba(6,78,59,.35);border:1px solid rgba(52,211,153,.2);
                        border-radius:12px;padding:14px 16px;margin:8px 0">
              <div style="font-size:.7rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:.08em;color:#6ee7b7;margin-bottom:8px">Trạng thái hệ thống</div>
              <div style="display:flex;align-items:center;gap:8px;font-size:.85rem;color:{status_color}">
                {status_icon} Mô hình: <strong>{status_text}</strong>
              </div>
              <div style="font-size:.75rem;color:#6ee7b7;margin-top:8px;word-break:break-all">
                📂 {str(model_path)[-40:]}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="font-size:.72rem;color:#4ade80;margin-top:12px;line-height:1.7;padding:0 4px">
              📦 Dataset: <code style="font-size:.72rem">{str(DATASET_ROOT)[-32:]}</code><br>
              🌿 Một ảnh · một quả · không đếm nhiều vật thể
            </div>
            """,
            unsafe_allow_html=True,
        )
    return page  # type: ignore[return-value]


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    if st is None:
        raise RuntimeError("Chưa cài Streamlit. Chạy: pip install -r requirements.txt")

    st.set_page_config(
        page_title="FreshLens — Nhận dạng thực phẩm",
        page_icon="🍏",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    model_path = _model_path()
    page = _render_sidebar(model_path)
    _hero(page)

    if page == "Chẩn đoán":
        _render_input_and_prediction(model_path)
        if st.session_state.get("processed") is not None:
            with st.expander("🔬 Xem ảnh trung gian & vector đặc trưng", expanded=False):
                _render_diagnostics(st.session_state["processed"])

    elif page == "Phòng học thuật":
        _render_learning(st.session_state.get("processed"), model_path)

    elif page == "Đánh giá":
        _render_evaluation()

    else:
        _render_team()


if __name__ == "__main__":
    main()
