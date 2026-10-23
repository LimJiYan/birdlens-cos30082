"""BirdLens Streamlit fallback: fully functional public bird classification UI.

This is a fallback for accounts ineligible for free Hugging Face ZeroGPU.
The assignment explicitly asks for a Hugging Face demonstration, so obtain
instructor approval before using a Streamlit URL as the sole deployment link.
"""
from __future__ import annotations

from io import BytesIO

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from inference import BirdPredictor

st.set_page_config(
    page_title="BirdLens | 200 Bird Species",
    page_icon="🐦",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.hero {padding: 23px 29px; background: linear-gradient(110deg, #124b3d, #1c6270);
       color: white; border-radius: 16px; margin: 0 0 22px;}
.hero h1 {color: white; margin: 5px 0 6px;}
.hero p {margin: 0; color: #e0f5e7;}
.small {color: #84938d; font-size: .86rem;}
</style>
<div class="hero"><div>COS30082 · Applied Machine Learning</div>
<h1>🐦 BirdLens</h1><p>Identify one of 200 bird species using ResNet18 + Mixup</p></div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading the verified Model 8 checkpoint...")
def load_model():
    # Cache once to avoid repeated CPU memory allocations on every button click.
    return BirdPredictor(device="cpu")

left, right = st.columns([1, 1], gap="large")
with left:
    uploaded = st.file_uploader("Bird photograph", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        try:
            photo = ImageOps.exif_transpose(Image.open(BytesIO(uploaded.getvalue()))).convert("RGB")
            st.image(photo, caption="Uploaded bird photograph", use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError):
            photo = None
            st.error("The uploaded file is not a valid image.")
    else:
        photo = None
        st.info("Upload a JPG, PNG, or WebP bird photo to begin.")
    run = st.button("Identify bird", type="primary", disabled=photo is None, use_container_width=True)

with right:
    st.subheader("Top 5 predicted species")
    if run and photo is not None:
        with st.spinner("Classifying image..."):
            predictions = load_model().predict_top5(photo)
        best, score = predictions[0]
        st.success(f"Predicted species: **{best}** — model score: **{score:.1%}**")
        for species, probability in predictions:
            st.markdown(f"**{species}** · {probability:.1%}")
            st.progress(min(1.0, max(0.0, probability)))
    else:
        st.write("The top-five predictions will appear after selecting **Identify bird**.")
    st.caption("Scores are model probabilities, not guarantees. The model only knows the 200 dataset species.")

st.divider()
st.caption("**200 species** · **Model 8 / epoch 18** · **ResNet18 + Mixup** · **224 × 224** inference")
st.caption("**Held-out test:** 58.55% Top-1 accuracy, 57.96% mean class accuracy on 1,204 images. "
           "Predictions may be incorrect, including when model scores are high. "
           "This is an educational demonstration, not verified wildlife identification.")
