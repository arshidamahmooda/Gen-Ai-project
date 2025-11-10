import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image, ImageDraw, ImageFont
import textwrap
import openai
import os

st.set_page_config(page_title="🎨 AI Comic Generator", layout="centered")

st.title("🎨 AI Comic Generator")
st.markdown("Generate a short comic scene from your text prompt using Generative AI models!")

prompt = st.text_area(
    "✍️ Enter your comic idea:",
    """Frog Prince’s Day Off

Panel 1: The frog prince sneaks out of his castle wearing sunglasses.
Panel 2: He rides a skateboard through the city streets.
Panel 3: He waves to surprised people as he zooms by fountains.
Panel 4: The frog prince jumps into his favorite pond with a happy splash."""
)

if st.button("🎬 Generate Comic"):
    st.info("⏳ Generating your comic... please wait 20–40 seconds")

    # --- Step 1: Generate short storyline using GPT ---
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")

        story_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a witty comic writer."},
                {"role": "user", "content": f"Write a 3-line comic scene about: {prompt}"}
            ]
        )
        story = story_resp.choices[0].message["content"].strip()
    except Exception as e:
        story = f"This comic shows: {prompt}. Our hero faces laughter, chaos, and triumph!"
        st.warning(f"Story generation fallback (error: {e})")

    st.subheader("💬 Comic Storyline")
    st.write(story)

    # --- Step 2: Generate comic image using Stable Diffusion ---
    st.subheader("🖼️ Comic Panel")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
            image = pipe(prompt, guidance_scale=7.5).images[0]

        # Add caption on top
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        wrapped = textwrap.fill(prompt, width=30)
        draw.text((10, 10), wrapped, fill="white", font=font)

        st.image(image, caption="✨ AI-generated Comic Panel", use_container_width=True)

    except Exception as e:
        st.error(f"Image generation failed: {e}")

st.caption("🚀 Powered by OpenAI + Stable Diffusion + Streamlit")
