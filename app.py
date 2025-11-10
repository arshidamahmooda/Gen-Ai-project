import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageDraw, ImageFont
import textwrap

# -------------------------
# 🎨 Streamlit App Setup
# -------------------------
st.set_page_config(page_title="🎨 Offline AI Comic Generator", layout="centered")
st.title("🎨 Offline AI Comic Generator")
st.markdown("Create comic panels using **GPT-2 + Stable Diffusion**")

# -------------------------
# 🧠 User Input
# -------------------------
prompt = st.text_area(
    "✍️ Enter your comic idea:",
    """Frog Prince’s Day Off

Panel 1: The frog prince sneaks out of his castle wearing sunglasses.
Panel 2: He rides a skateboard through the city streets.
Panel 3: He waves to surprised people as he zooms by fountains.
Panel 4: The frog prince jumps into his favorite pond with a happy splash."""
)

if st.button("🎬 Generate Comic"):
    st.info("⏳ Generating comic... please wait 1–2 minutes for first-time model load.")

    # -------------------------
    # 🧠 GPT-2 Text Generation
    # -------------------------
    st.subheader("💬 Comic Storyline")
    text_gen = pipeline("text-generation", model="gpt2")
    story = text_gen(prompt, max_new_tokens=60)[0]["generated_text"]
    st.write(story)

    # -------------------------
    # 🎨 Stable Diffusion Image Generation
    # -------------------------
    st.subheader("🖼️ Comic Panel")

    # Use the Hugging Face model (you can choose others)
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    image = pipe(prompt).images[0]

    # -------------------------
    # 🗨️ Add Caption Text
    # -------------------------
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    wrapped = textwrap.fill(prompt, width=30)
    draw.text((10, 10), wrapped, fill="white", font=font)

    st.image(image, caption="✨ AI-generated Comic Panel", use_container_width=True)

st.caption("🚀 Powered by GPT-2 + Stable Diffusion + Streamlit ")
