from transformers import pipeline
import tensorflow as tf
import numpy as np
import streamlit as st

# Load your trained hazard detection model
model = tf.keras.models.load_model("hazard_model.h5")

# Load a small generative model (e.g., T5 for report generation)
gen_ai = pipeline("text2text-generation", model="google/flan-t5-small")

st.title("🧠 Generative AI Hazard Detection")
st.write("Detect hazards and generate automatic reports!")

# Example input fields
feature_names = ["feature1", "feature2", "feature3"]  # update as per dataset
inputs = []
for f in feature_names:
    val = st.number_input(f"Enter {f}:", value=0.0)
    inputs.append(val)

if st.button("Analyze Hazard"):
    X_input = np.array([inputs])
    pred = model.predict(X_input)
    hazard_idx = np.argmax(pred)

    if hazard_idx == 1:
        hazard_type = "Hazardous"
    else:
        hazard_type = "Safe"

    st.write(f"🔍 Prediction: **{hazard_type}**")

    # Generate AI-based explanation
    prompt = f"Generate a short safety report for a {hazard_type} situation."
    report = gen_ai(prompt, max_length=60)[0]['generated_text']

    st.subheader("📝 AI-Generated Report:")
    st.write(report)
