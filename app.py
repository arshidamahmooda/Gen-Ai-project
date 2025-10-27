import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

st.title("🧠 Hazard Detection System")
st.write("Upload data and predict whether it’s hazardous or safe.")

# Load model
model = tf.keras.models.load_model("hazard_model.h5")

# Example input (based on your dataset features)
feature_names = ["feature1", "feature2", "feature3"]  # update as per your dataset

inputs = []
for name in feature_names:
    val = st.number_input(f"Enter {name}:", value=0.0)
    inputs.append(val)

if st.button("Predict Hazard"):
    X_input = np.array([inputs])
    prediction = model.predict(X_input)
    class_idx = np.argmax(prediction)
    st.success(f"Predicted Class: {class_idx}")
