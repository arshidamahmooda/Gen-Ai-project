import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import os

st.set_page_config(page_title="🎨 AI Comic Creator", layout="centered")

st.title("🎨 AI Comic Creator")
st.markdown("Generate a short comic scene from your text prompt!")

prompt = st.text_area("✍️ Enter your comic idea:", "A dog learns to play the guitar on stage!")

if st.button("Generate Comic"):
    st.info("⏳ Generating... please wait 10–15 seconds")

    # --- Generate short storyline using OpenAI API ---
    try:
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY", "your_api_key_here")  # replace if running locally

        story_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a creative comic writer."},
                      {"role": "user", "content": f"Write a funny 3-line comic scene about: {prompt}"}]
        )
        story = story_resp.choices[0].message["content"].strip()
    except Exception:
        # fallback if no API key or offline — use simple placeholder story
        story = f"This comic shows: {prompt}. Our hero faces laughter, chaos, and triumph!"

    st.subheader("💬 Comic Storyline")
    st.write(story)

    # --- Generate comic image using Pollinations API ---
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

st.caption("🚀 Powered by Streamlit + OpenAI + Pollinations API")
