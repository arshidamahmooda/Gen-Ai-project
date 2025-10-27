# app.py
# 🚧 Construction Safety Detector - Streamlit App

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# ------------------------------------------------------------
# ⚙️ Page Configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Construction Safety Detector", page_icon="🦺", layout="centered")
st.title("🦺 Construction Site Safety Detector")
st.write("Upload an image to check whether the site is **Safe ✅** or **Unsafe 🚧**")

# ------------------------------------------------------------
# 🔗 Google Drive Model Link
# ------------------------------------------------------------
# 👉 Replace with your OWN Google Drive share link (Make it 'Anyone with the link')
# Example:
# https://drive.google.com/file/d/1ABCxyz123/view?usp=sharing
# ↓ becomes:
# https://drive.google.com/uc?id=1ABCxyz123
MODEL_URL = "https://drive.google.com/uc?id=1eug-EkN_7KH2MVClBwc8VGi16XnftYCd"
MODEL_PATH = "efficientnet_hazard_model.h5"

# ------------------------------------------------------------
# ⬇️ Download model if not already present
# ------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Downloading model... Please wait..."):
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to download model: {e}")
            st.stop()

# ------------------------------------------------------------
# 🧠 Load Model
# ------------------------------------------------------------
@st.cache_resource
def load_hazard_model():
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()

model = load_hazard_model()
st.success("✅ Model ready for prediction!")

# ------------------------------------------------------------
# 📤 Upload Image
# ------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a construction site image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((128, 128))
    x = np.array(img_resized) / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    with st.spinner("🔍 Analyzing image..."):
        pred = model.predict(x)[0][0]
        confidence = (pred if pred > 0.5 else 1 - pred) * 100

    # Result Display
    if pred > 0.5:
        st.markdown("<h2 style='color:red;text-align:center;'>🚧 UNSAFE: Hazard Detected!</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:green;text-align:center;'>✅ SAFE: No Hazard Detected!</h2>", unsafe_allow_html=True)

    st.write(f"**Model Confidence:** {confidence:.2f}%")
else:
    st.info("📸 Please upload an image to start prediction.")
