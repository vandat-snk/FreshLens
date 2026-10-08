"""FreshLens modular demo UI: upload/camera -> open-set gate -> fruit + condition."""

from __future__ import annotations

import os

import torch
import streamlit as st

from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.data import rgb_from_bytes
from freshlens_ai.inference import (
    clear_result,
    result_for_image,
    store_result,
)
from freshlens_ai.inference.cnn_predict import FreshLensPredictor
from freshlens_ai.inference.quality import DEFAULT_QUALITY_CONFIG, load_quality_config, quality_config_hash


CHECKPOINT = (
    PROJECT_DIR
    / "models"
    / "cnn_efficientnet_b0"
    / "best.pt"
)

GATE_NPZ = (
    PROJECT_DIR
    / "models"
    / "cnn_efficientnet_b0"
    / "open_set_gate.npz"
)

GATE_META = (
    PROJECT_DIR
    / "models"
    / "cnn_efficientnet_b0"
    / "open_set_gate.json"
)


FRUIT_VI = {
    "apple": "Táo",
    "banana": "Chuối",
    "orange": "Cam",
    "tomato": "Cà chua",
}

STATUS_VI = {
    "fresh": "Tươi",
    "rotten": "Hỏng / có dấu hiệu hỏng",
}


st.set_page_config(
    page_title="FreshLens CNN",
    page_icon="🍎",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {
    padding: 1.2rem 1.35rem;
    border: 1px solid rgba(128,128,128,.25);
    border-radius: 18px;
    margin-bottom: 1rem;
}
.hero h1 {margin: 0 0 .25rem 0; font-size: 2rem;}
.hero p {margin: 0; opacity: .78;}
.good {
    padding: 1rem 1.15rem;
    border-radius: 14px;
    border: 1px solid rgba(70,180,110,.35);
}
.unsupported {
    padding: 1.1rem 1.2rem;
    border-radius: 14px;
    border: 1px solid rgba(240,150,70,.45);
}
.big {
    font-size: 1.65rem;
    font-weight: 750;
    margin-bottom: .2rem;
}
.muted {opacity: .72;}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="hero">
  <h1>FreshLens · Nhận diện trái cây</h1>
  <p>
    Hỗ trợ Táo · Chuối · Cam · Cà chua —
    nhận diện loại quả và tình trạng tươi/hỏng
    từ ảnh tải lên hoặc camera.
  </p>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_runtime(quality_config):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    return FreshLensPredictor(
        CHECKPOINT, GATE_NPZ, str(device), gate_meta_path=GATE_META, quality_config=quality_config,
    )


try:
    quality_config = load_quality_config(os.environ.get("FRESHLENS_QUALITY_CONFIG", DEFAULT_QUALITY_CONFIG))
    runtime = load_runtime(quality_config)
    metadata, gate, device = runtime.metadata, runtime.openset_gate, runtime.device

except Exception as exc:
    st.error(
        f"Chưa thể khởi động mô hình: {exc}"
    )

    st.caption("Kiểm tra checkpoint và hai file gate tương ứng trước khi chạy lại.")

    st.stop()


with st.sidebar:
    st.subheader(
        "Mô hình"
    )

    st.write(
        "EfficientNet-B0"
    )

    st.caption(
        f"Checkpoint epoch "
        f"{metadata.get('epoch', '?')} "
        f"· device: {device}"
    )

    st.caption(
        "Open-set gate dùng để từ chối ảnh "
        "không đủ giống 4 loại quả được hỗ trợ."
    )

    with st.expander(
        "Thông tin gate"
    ):
        st.json(
            {
                "gate_version": gate[
                    "meta"
                ].get(
                    "version"
                ),
                "known_calibration_accept_rate": gate[
                    "meta"
                ].get(
                    "known_calibration_accept_rate"
                ),
                "unknown_calibration_reject_rate": gate[
                    "meta"
                ].get(
                    "unknown_calibration_reject_rate"
                ),
                "threshold": gate[
                    "decision_threshold"
                ],
            }
        )


quality_enabled = st.sidebar.checkbox(
    "Thử nghiệm kiểm tra chất lượng ảnh", value=False,
    help="Ngưỡng chất lượng đang thử nghiệm; cần đánh giá trên ảnh thực tế trước khi dùng mặc định.",
)
quality_key = "quality_policy_enabled"
quality_policy = (quality_enabled, quality_config_hash(quality_config))
if st.session_state.get(quality_key) != quality_policy:
    clear_result(st.session_state)
    st.session_state[quality_key] = quality_policy

source = st.radio(
    "Nguồn ảnh",
    [
        "Tải ảnh",
        "Chụp bằng camera",
    ],
    horizontal=True,
)


if source == "Tải ảnh":
    item = st.file_uploader(
        "Chọn một ảnh",
        type=[
            "jpg",
            "jpeg",
            "png",
            "bmp",
            "webp",
        ],
    )

else:
    item = st.camera_input(
        "Chụp một quả chính trong khung hình"
    )


if item is None:
    st.info(
        "Chọn hoặc chụp một ảnh để bắt đầu."
    )
    st.stop()


image_bytes = item.getvalue()
try:
    preview = rgb_from_bytes(image_bytes)
except Exception as exc:
    clear_result(st.session_state)
    st.error(f"Không đọc được ảnh: {exc}")
    st.stop()

left, right = st.columns(
    [
        1.05,
        0.95,
    ],
    gap="large",
)


with left:
    st.image(
        preview,
        caption="Ảnh đầu vào",
        width="stretch",
    )


with right:
    if st.button(
        "🔎 Phân tích ảnh",
        type="primary",
        width="stretch",
    ):
        try:
            with st.spinner(
                "Đang phân tích..."
            ):
                analyzed = runtime.predict(image_bytes, check_quality=quality_enabled)

            store_result(
                st.session_state,
                image_bytes,
                analyzed,
            )

        except Exception as exc:
            clear_result(
                st.session_state
            )

            st.error(
                "Không đọc/nhận diện được ảnh: "
                f"{exc}"
            )

    # Important: a result is shown only for the exact image bytes that
    # produced it. Selecting/capturing a new image can no longer display
    # the previous image's result.
    result = result_for_image(
        st.session_state,
        image_bytes,
    )

    if result:
        if result["status"] == "quality_rejection":
            st.warning("Ảnh chưa đạt chất lượng — vui lòng chụp lại.")
            st.write(result["rejection_reason"])
        elif not result["supported"]:
            st.markdown(
                """
<div class="unsupported">
  <div class="big">
    Chưa đủ cơ sở nhận diện trong phạm vi hỗ trợ
  </div>
  <div class="muted">
    Hệ thống hiện hỗ trợ: Táo, Chuối, Cam và Cà chua.
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            st.caption(
                "Nếu đây thực sự là một trong 4 loại trên, "
                "hãy thử chụp lại một quả chính, rõ nét và đủ sáng."
            )

        else:
            fruit = FRUIT_VI[
                result["fruit"]
            ]

            status = STATUS_VI[
                result["condition"]
            ]

            st.markdown(
                f"""
<div class="good">
  <div class="big">{fruit}</div>
  <div>
    Tình trạng: <b>{status}</b>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(
                2
            )

            c1.metric(
                "Tin cậy loại quả",
                f"{result['fruit_score']:.1%}",
            )

            c2.metric(
                "Tin cậy tình trạng",
                (
                    f"{result['condition_score_given_fruit']:.1%}"
                ),
            )

        st.caption(f"Thời gian phân tích: {result['latency_ms']:.0f} ms")
        with st.expander(
            "Chi tiết kỹ thuật"
        ):
            st.write(
                "Supported score: "
                f"{result['support_probability']:.3f} "
                "/ threshold "
                f"{result['support_threshold']:.3f}"
            )

            st.json(
                {
                    "fruit_scores": result[
                        "fruit_scores"
                    ],
                    "quality": result["quality"],
                    "gate": result[
                        "gate_detail"
                    ],
                }
            )


st.divider()

st.caption(
    "FreshLens đánh giá dấu hiệu nhìn thấy trong ảnh. "
    "Kết quả không thay thế kiểm nghiệm an toàn thực phẩm. "
    "Open-set detection giúp từ chối ảnh ngoài phạm vi "
    "nhưng không thể bảo đảm tuyệt đối cho mọi ảnh có thể xảy ra."
)
