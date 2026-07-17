"""Streamlit app: predict diabetic-retinopathy / blindness level from a fundus photo."""

from __future__ import annotations

import os

import streamlit as st
import torch
from PIL import Image

from model import CLASS_DESCRIPTIONS, CLASS_NAMES, load_model, predict

# Checkpoint that lives one level up in the repo.
MODEL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "best_model.pth")
)

# Color per severity grade (green -> red).
GRADE_COLORS = {
    "No DR": "#2e7d32",
    "Mild": "#9e9d24",
    "Moderate": "#f9a825",
    "Severe": "#ef6c00",
    "Proliferative": "#c62828",
}

st.set_page_config(page_title="Blindness Level Detection", page_icon="👁️", layout="centered")


@st.cache_resource(show_spinner="Loading model…")
def get_model(path: str, device: str):
    return load_model(path, device=device)


st.title("👁️ Blindness Level Detection")
st.caption(
    "Upload a retinal **fundus photograph** to estimate the diabetic-retinopathy grade "
    "using a from-scratch convolutional neural network."
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = None
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: `{MODEL_PATH}`.")
else:
    try:
        model = get_model(MODEL_PATH, device)
    except Exception as exc:  # noqa: BLE001 - surface any load failure to the user
        st.error(f"Failed to load model: {exc}")

# --- Main: image upload + prediction ---
uploaded = st.file_uploader(
    "Upload a fundus image", type=["png", "jpg", "jpeg"], disabled=model is None
)

if uploaded is not None and model is not None:
    image = Image.open(uploaded)

    col_img, col_pred = st.columns(2)
    with col_img:
        st.image(image, caption="Input fundus image", use_container_width=True)

    results = predict(model, image, device=device)
    top_name, top_prob = max(results, key=lambda r: r[1])
    color = GRADE_COLORS.get(top_name, "#333")

    with col_pred:
        st.markdown("#### Prediction")
        st.markdown(
            f"<div style='padding:1rem;border-radius:0.5rem;background:{color};color:white;'>"
            f"<div style='font-size:1.6rem;font-weight:700;'>{top_name}</div>"
            f"<div style='opacity:0.9;'>{CLASS_DESCRIPTIONS[top_name]}</div>"
            f"<div style='margin-top:0.5rem;'>Confidence: {top_prob:.1%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### Class probabilities")
    for name in CLASS_NAMES:
        prob = dict(results)[name]
        st.write(f"**{name}** — {prob:.1%}")
        st.progress(min(max(prob, 0.0), 1.0))

    st.info(
        "This tool is for educational and research purposes only and is **not** a medical "
        "device. It should not be used for diagnosis. Consult an ophthalmologist for care."
    )
elif model is not None:
    st.info("Upload a fundus image to get a prediction.")
