"""Streamlit app: predict diabetic-retinopathy / blindness level from a fundus photo."""

from __future__ import annotations

import os

import streamlit as st
import torch
from PIL import Image

from explain import compute_gradcam, overlay_heatmap
from model import (
    CLASS_DESCRIPTIONS,
    CLASS_NAMES,
    IMG_SIZE,
    load_model,
    model_input_image,
    predict,
)

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
    # Ben Graham preprocessing, the same pipeline the training corpus went
    # through. Computed once here and reused for display and Grad-CAM.
    processed = model_input_image(image)

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

    # --- Explainability: Grad-CAM heatmap ---
    st.markdown("#### Why this prediction? (Grad-CAM)")
    st.caption(
        "The heatmap highlights the retinal regions that most influenced the "
        "model's score for the selected grade — red/yellow areas contributed "
        "the most, transparent areas the least."
    )

    ctrl_class, ctrl_alpha = st.columns(2)
    with ctrl_class:
        explain_name = st.selectbox(
            "Grade to explain",
            CLASS_NAMES,
            index=CLASS_NAMES.index(top_name),
            help="Defaults to the predicted grade; pick another to see what evidence the model finds for it.",
        )
    with ctrl_alpha:
        heat_alpha = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)

    with st.spinner("Computing Grad-CAM…"):
        cam = compute_gradcam(
            model, image, CLASS_NAMES.index(explain_name), device=device
        )
    overlay = overlay_heatmap(processed, cam, alpha=heat_alpha)

    col_orig, col_cam = st.columns(2)
    with col_orig:
        st.image(
            processed,
            caption=f"Model input — Ben Graham preprocessed ({IMG_SIZE}×{IMG_SIZE})",
            use_container_width=True,
        )
    with col_cam:
        st.image(
            overlay,
            caption=f"Evidence for “{explain_name}” ({dict(results)[explain_name]:.1%})",
            use_container_width=True,
        )

    with st.expander("How to read this map"):
        st.markdown(
            "**Grad-CAM** (Gradient-weighted Class Activation Mapping) traces the "
            "gradients of the selected grade's score back to the network's last "
            "convolutional layer, showing *where* in the image the evidence for "
            "that grade came from.\n\n"
            "- For DR grades, warm regions ideally align with clinical signs such "
            "as microaneurysms, hemorrhages, hard exudates, or neovascularization.\n"
            f"- The map is coarse (upsampled from a {IMG_SIZE // 32}×{IMG_SIZE // 32} "
            "grid), so it localizes broad regions, not individual lesions.\n"
            "- A heatmap concentrated outside the retina (e.g. on the black "
            "border) is a sign the prediction may not be trustworthy for this image."
        )

    st.info(
        "This tool is for educational and research purposes only and is **not** a medical "
        "device. It should not be used for diagnosis. Consult an ophthalmologist for care."
    )
elif model is not None:
    st.info("Upload a fundus image to get a prediction.")
