"""FreshLens modular demo UI: upload/camera -> open-set gate -> fruit + condition."""

from __future__ import annotations

import torch
import streamlit as st

from freshlens_ai.constants import PROJECT_DIR
from freshlens_ai.inference import (
    analyze_bytes,
    clear_result,
    load_gate,
    result_for_image,
    store_result,
)
from freshlens_ai.models import load_model


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
def load_runtime():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model, metadata = load_model(
        CHECKPOINT,
        device,
    )

    gate = load_gate(
        GATE_NPZ,
        GATE_META,
        CHECKPOINT,
    )

    return (
        model,
        metadata,
        gate,
        device,
    )


try:
    (
        model,
        metadata,
        gate,
        device,
    ) = load_runtime()

except Exception as exc:
    st.error(
        f"Chưa thể khởi động mô hình: {exc}"
    )

    st.code(
        r".\.venv\Scripts\python.exe BUILD_OPENSET_GATE_V2.py --device cuda",
        language="text",
    )

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
                "known_validation_accept_rate": gate[
                    "meta"
                ].get(
                    "known_validation_accept_rate"
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

left, right = st.columns(
    [
        1.05,
        0.95,
    ],
    gap="large",
)


with left:
    st.image(
        image_bytes,
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
                analyzed = analyze_bytes(
                    model,
                    image_bytes,
                    device,
                    gate,
                )

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
        if not result[
            "supported"
        ]:
            st.markdown(
                """
<div class="unsupported">
  <div class="big">
    Loại quả này hiện chưa được FreshLens hỗ trợ
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
