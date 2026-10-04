"""FreshLens final demo UI: upload/camera -> supported gate -> fruit + condition."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch
import streamlit as st

from cnn_data import DataError
from cnn_model import load_model
from open_set import analyze_bytes, load_gate

CHECKPOINT = PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "best.pt"
GATE_NPZ = PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "open_set_gate.npz"
GATE_META = PROJECT_DIR / "models" / "cnn_efficientnet_b0" / "open_set_gate.json"

FRUIT_VI = {"apple": "Táo", "banana": "Chuối", "orange": "Cam", "tomato": "Cà chua"}
STATUS_VI = {"fresh": "Tươi", "rotten": "Hỏng / có dấu hiệu hỏng"}

st.set_page_config(page_title="FreshLens CNN", page_icon="🍎", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {padding: 1.2rem 1.35rem; border: 1px solid rgba(128,128,128,.25); border-radius: 18px; margin-bottom: 1rem;}
.hero h1 {margin: 0 0 .25rem 0; font-size: 2rem;}
.hero p {margin: 0; opacity: .78;}
.good {padding: 1rem 1.15rem; border-radius: 14px; border: 1px solid rgba(70,180,110,.35);}
.unsupported {padding: 1.1rem 1.2rem; border-radius: 14px; border: 1px solid rgba(240,150,70,.45);}
.big {font-size: 1.65rem; font-weight: 750; margin-bottom: .2rem;}
.muted {opacity: .72;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>FreshLens · Nhận diện trái cây</h1>
  <p>Hỗ trợ Táo · Chuối · Cam · Cà chua — nhận diện loại quả và tình trạng tươi/hỏng từ ảnh tải lên hoặc camera.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_runtime():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = load_model(CHECKPOINT, device)
    gate = load_gate(GATE_NPZ, GATE_META, CHECKPOINT)
    return model, metadata, gate, device


try:
    model, metadata, gate, device = load_runtime()
except Exception as exc:
    st.error(f"Chưa thể khởi động mô hình: {exc}")
    st.code("FreshLens_Buoc5_AppThucTe\\SETUP_STEP5.cmd", language="text")
    st.stop()

with st.sidebar:
    st.subheader("Mô hình")
    st.write("EfficientNet-B0")
    st.caption(f"Checkpoint epoch {metadata.get('epoch', '?')} · device: {device}")
    st.caption("Open-set gate dùng để từ chối ảnh không đủ giống 4 loại quả được hỗ trợ.")
    with st.expander("Thông tin gate"):
        st.json({
            "known_validation_accept_rate": gate["meta"].get("known_validation_accept_rate"),
            "unknown_calibration_reject_rate": gate["meta"].get("unknown_calibration_reject_rate"),
            "threshold": gate["decision_threshold"],
        })

source = st.radio("Nguồn ảnh", ["Tải ảnh", "Chụp bằng camera"], horizontal=True)
if source == "Tải ảnh":
    item = st.file_uploader("Chọn một ảnh", type=["jpg", "jpeg", "png", "bmp", "webp"])
else:
    item = st.camera_input("Chụp một quả chính trong khung hình")

if item is None:
    st.info("Chọn hoặc chụp một ảnh để bắt đầu.")
    st.stop()

image_bytes = item.getvalue()
left, right = st.columns([1.05, .95], gap="large")
with left:
    st.image(image_bytes, caption="Ảnh đầu vào", use_container_width=True)

with right:
    if st.button("🔎 Phân tích ảnh", type="primary", use_container_width=True):
        try:
            with st.spinner("Đang phân tích..."):
                result = analyze_bytes(model, image_bytes, device, gate)
            st.session_state["last_result"] = result
        except Exception as exc:
            st.session_state.pop("last_result", None)
            st.error(f"Không đọc/nhận diện được ảnh: {exc}")

    result = st.session_state.get("last_result")
    if result:
        if not result["supported"]:
            st.markdown("""
            <div class="unsupported">
              <div class="big">Loại quả này hiện chưa được FreshLens hỗ trợ</div>
              <div class="muted">Hệ thống hiện hỗ trợ: Táo, Chuối, Cam và Cà chua.</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption("Nếu đây thực sự là một trong 4 loại trên, hãy thử chụp lại một quả chính, rõ nét và đủ sáng.")
        else:
            fruit = FRUIT_VI[result["fruit"]]
            status = STATUS_VI[result["condition"]]
            st.markdown(f"""
            <div class="good">
              <div class="big">{fruit}</div>
              <div>Tình trạng: <b>{status}</b></div>
            </div>
            """, unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.metric("Tin cậy loại quả", f"{result['fruit_score']:.1%}")
            c2.metric("Tin cậy tình trạng", f"{result['condition_score_given_fruit']:.1%}")

        with st.expander("Chi tiết kỹ thuật"):
            st.write(f"Supported score: {result['support_probability']:.3f} / threshold {result['support_threshold']:.3f}")
            st.json({"fruit_scores": result["fruit_scores"], "gate": result["gate_detail"]})

st.divider()
st.caption("FreshLens đánh giá dấu hiệu nhìn thấy trong ảnh. Kết quả không thay thế kiểm nghiệm an toàn thực phẩm. Open-set detection giúp từ chối ảnh ngoài phạm vi nhưng không thể bảo đảm tuyệt đối cho mọi ảnh có thể xảy ra.")
