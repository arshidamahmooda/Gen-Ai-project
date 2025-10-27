# ====================================================
# 🧠 Generative AI Hazard Detection System
# ====================================================
# Author: Your Name
# Description:
# - Loads a trained hazard detection model (EfficientNet)
# - Classifies uploaded images as Safe or Unsafe
# - Uses Generative AI (FLAN-T5) to generate a safety report
# ====================================================

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import requests, os

# ----------------------------------------------------
# STEP 1: Download model from Google Drive or Hugging Face
# ----------------------------------------------------
MODEL_URL = "https://drive.google.com/uc?id=1eug-EkN_7KH2MVClBwc8VGi16XnftYCd"  # 🔹 Replace this link
MODEL_PATH = "efficientnet_hazard_model.h5"

if not os.path.exists(MODEL_PATH):
    st.write("📥 Downloading model file...")
    with open(MODEL_PATH, "wb") as f:
        f.write(requests.get(MODEL_URL).content)
    st.success("✅ Model downloaded successfully!")

# ----------------------------------------------------
# STEP 2: Load model
# ----------------------------------------------------
st.write("🔄 Loading hazard detection model...")
model = tf.keras.models.load_model(MODEL_PATH)
st.success("✅ Model loaded successfully!")

# ----------------------------------------------------
# STEP 3: Streamlit UI
# ----------------------------------------------------
st.title("🧠 Generative AI Hazard Detection System")
st.write("Upload a worksite image to detect if it’s **Safe or Unsafe**, and let AI generate a safety report.")

uploaded_file = st.file_uploader("📸 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]
    label = "Unsafe" if pred > 0.5 else "Safe"
    confidence = float(pred) if label == "Unsafe" else 1 - float(pred)

    st.subheader(f"🔍 Prediction: **{label}**")
    st.write(f"Confidence: {confidence:.2f}")

    # ----------------------------------------------------
    # STEP 4: Generative AI Explanation (FLAN-T5)
    # ----------------------------------------------------
    from transformers import pipeline

    st.write("🧠 Generating AI-based safety report...")
    gen_ai = pipeline("text2text-generation", model="google/flan-t5-small")

    prompt = f"Write a short safety report for a {label.lower()} construction worksite."
    report = gen_ai(prompt, max_length=80)[0]['generated_text']

    st.subheader("📝 AI-Generated Safety Report:")
    st.write(report)

# ----------------------------------------------------
# STEP 5: Footer
# ----------------------------------------------------
st.markdown("""
---
**Project:** Generative AI Hazard Detection  
**Model:** EfficientNetB0 + FLAN-T5  
**Developed with ❤️ using Streamlit**
""")
