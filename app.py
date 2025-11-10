import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import openai
import os

st.set_page_config(page_title="🎨 AI Comic Generator", layout="centered")

st.title("🎨 AI Comic Generator")
st.markdown("Create short comic scenes using GPT + AI Image Generation")

prompt = st.text_area(
    "✍️ Enter your comic idea:",
    """Frog Prince’s Day Off

Panel 1: The frog prince sneaks out of his castle wearing sunglasses.
Panel 2: He rides a skateboard through the city streets.
Panel 3: He waves to surprised people as he zooms by fountains.
Panel 4: The frog prince jumps into his favorite pond with a happy splash."""
)

if st.button("🎬 Generate Comic"):
    st.info("⏳ Generating your comic... please wait 10–15 seconds")

    # --- GPT Text Generation ---
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        story_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative comic writer."},
                {"role": "user", "content": f"Write a 3-line funny comic scene about: {prompt}"}
            ]
        )
        story = story_resp.choices[0].message["content"].strip()
    except Exception as e:
        story = f"This comic shows: {prompt}. (Error: {e})"

    st.subheader("💬 Comic Storyline")
    st.write(story)

    # --- Pollinations Image Generation (Lightweight) ---
    st.subheader("🖼️ Comic Panel")
    img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
    response = requests.get(img_url)
    image = Image.open(BytesIO(response.content))

    # --- Add caption text on image ---
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    wrapped = textwrap.fill(prompt, width=30)
    draw.text((10, 10), wrapped, fill="white", font=font)

    st.image(image, caption="✨ AI-generated Comic Panel", use_container_width=True)

st.caption("🚀 Powered by OpenAI + Pollinations + Streamlit")
