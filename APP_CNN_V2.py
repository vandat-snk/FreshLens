"""FreshLens modular demo UI entry point."""

from __future__ import annotations

import torch
import streamlit as st

from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.inference import analyze_bytes, load_gate
from freshlens_ai.models import load_model
from ui.diagnosis import render_diagnosis_page


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
)
