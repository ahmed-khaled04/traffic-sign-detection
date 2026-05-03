"""
app.py — Streamlit web app for traffic sign recognition demo.

Run:
    streamlit run app.py

Requires a trained model at results/svm_model.pkl (run train.py first).
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess_image
from segmentation  import segment_image, find_sign_bbox
from features      import extract_hog_visual, extract_hog
from classifier    import load_model, predict
from utils         import GTSRB_CLASSES


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚦 Traffic Sign Recognition")
    st.markdown("**Pipeline**")
    st.markdown("""
1. Resize → 64×64
2. BGR → HSV + Normalize
3. HOG + Color Histogram features
4. SVM classification
""")
    st.divider()
    model_path = st.text_input("Model path", value="results/svm_model.pkl")
    st.divider()
    st.caption("GTSRB dataset · 43 classes · SVM + HOG")

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def get_model(path):
    if not os.path.exists(path):
        return None
    return load_model(path)


model = get_model(model_path)

if model is None:
    st.warning(
        f"No model found at **{model_path}**. "
        "Run `python train.py` first to train and save the model."
    )

# ── Main area ─────────────────────────────────────────────────────────────────
st.header("Upload a Traffic Sign Image")
uploaded = st.file_uploader(
    "Supported formats: JPG, PNG, PPM",
    type=["jpg", "jpeg", "png", "ppm"],
)

if uploaded is not None:
    # Decode upload to BGR numpy array
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if bgr is None:
        st.error("Could not decode the uploaded image.")
        st.stop()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    # Crop to detected sign region before resizing so background doesn't
    # pollute HOG — GTSRB training images were already tight ROI crops.
    x1, y1, x2, y2 = find_sign_bbox(bgr)
    bgr_cropped     = bgr[y1:y2, x1:x2] if (x2 - x1) > 10 and (y2 - y1) > 10 else bgr

    hsv_norm        = preprocess_image(bgr_cropped)
    segmented, mask = segment_image(hsv_norm)
    # HOG is extracted from the full preprocessed image (not segmented) to
    # match training — train.py never applies segmentation before HOG.
    features, hog_img = extract_hog_visual(hsv_norm)

    # ── Pipeline visualization strip ──────────────────────────────────────────
    st.subheader("Pipeline Steps")
    col1, col2, col3, col4, col5 = st.columns(5)

    # Original — show cropped region so user sees what the model received
    with col1:
        st.markdown("**1 · Auto-cropped**")
        rgb_orig = cv2.cvtColor(bgr_cropped, cv2.COLOR_BGR2RGB)
        st.image(rgb_orig, use_container_width=True)

    # Preprocessed HSV — display V channel as grayscale for readability
    with col2:
        st.markdown("**2 · Preprocessed (HSV)**")
        # Rescale each channel to uint8 for display
        h_disp = (hsv_norm[:, :, 0] * 179).astype(np.uint8)
        s_disp = (hsv_norm[:, :, 1] * 255).astype(np.uint8)
        v_disp = (hsv_norm[:, :, 2] * 255).astype(np.uint8)
        hsv_display = np.stack([h_disp, s_disp, v_disp], axis=-1)
        # Treat as RGB so Streamlit shows colour variation across channels
        st.image(hsv_display, use_container_width=True)

    # Color mask
    with col3:
        st.markdown("**3 · Color Mask**")
        st.image(mask, use_container_width=True, clamp=True)

    # HOG visualization
    with col4:
        st.markdown("**4 · HOG Features**")
        hog_disp = (hog_img / hog_img.max() * 255).astype(np.uint8) if hog_img.max() > 0 else hog_img.astype(np.uint8)
        st.image(hog_disp, use_container_width=True)

    # Prediction
    with col5:
        st.markdown("**5 · Prediction**")
        if model is not None:
            class_id, confidence = predict(model, features)
            class_name = GTSRB_CLASSES.get(class_id, f"Class {class_id}")
            st.metric("Class", f"#{class_id}")
            st.write(f"**{class_name}**")
            st.progress(confidence)
            st.caption(f"Confidence: {confidence*100:.1f}%")
        else:
            st.info("Load a model to see predictions.")

    # ── Pipeline details expander ─────────────────────────────────────────────
    with st.expander("Pipeline Details"):
        st.markdown("""
**HSV Color Ranges Used for Segmentation** *(normalized [0, 1])*

| Color  | Hue range        | Saturation | Value  |
|--------|-----------------|------------|--------|
| Red    | 0.000 – 0.056   | ≥ 0.40     | ≥ 0.40 |
| Red    | 0.894 – 1.000   | ≥ 0.40     | ≥ 0.40 |
| Blue   | 0.559 – 0.726   | ≥ 0.40     | ≥ 0.20 |
| Yellow | 0.084 – 0.196   | ≥ 0.40     | ≥ 0.40 |

**HOG Parameters**

| Parameter        | Value   |
|-----------------|---------|
| Orientations    | 9       |
| Pixels per cell | 8 × 8   |
| Cells per block | 2 × 2   |
| Block norm      | L2-Hys  |
| Transform sqrt  | True    |

**Classifier:** SVM with RBF kernel · C=10 · γ=0.001 · probability=True
**Scaler:** StandardScaler (zero mean, unit variance)
""")
