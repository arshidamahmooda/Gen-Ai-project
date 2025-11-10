import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import os

# -----------------------------
# 🎨 App Configuration
# -----------------------------
st.set_page_config(page_title="🎨 AI Comic Creator", layout="centered")

st.title("🎨 AI Comic Creator")
st.markdown("Generate a short comic scene from your text prompt using AI!")

# -----------------------------
# ✍️ User Input
# -----------------------------
prompt = st.text_area("✍️ Enter your comic idea:", """Frog Prince’s Day Off

Panel 1: The frog prince sneaks out of his castle wearing sunglasses.
Panel 2: He rides a skateboard through the city streets.
Panel 3: He waves to surprised people as he zooms by fountains.
Panel 4: The frog prince jumps into his favorite pond with a happy splash.""")

# -----------------------------
# 🚀 Generate Button
# -----------------------------
if st.button("Generate Comic"):
    st.info("⏳ Generating... please wait 10–15 seconds")

    # --- Generate short storyline using OpenAI API ---
    try:
        import openai

        # ✅ Load your OpenAI API key safely
        openai.api_key = os.getenv("OPENAI_API_KEY")  # Set in environment variable

        if not openai.api_key or openai.api_key == "your_api_key_here":
            raise ValueError("No valid API key found")

        story_resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a creative comic writer."},
                {"role": "user", "content": f"Write a funny 3-line comic scene about: {prompt}"}
            ]
        )

        story = story_resp.choices[0].message["content"].strip()

    except Exception as e:
        # 🧩 Fallback if offline or missing API key
        story = f"This comic shows: {prompt}. Our hero faces laughter, chaos, and triumph!"
        st.warning(f"⚠️ Could not use OpenAI API ({e}). Using offline text instead.")

    # --- Display Story ---
    st.subheader("💬 Comic Storyline")
    st.write(story)

    # -----------------------------
    # 🖼️ Generate comic image using Pollinations API
    # -----------------------------
    st.subheader("🖼️ Comic Panel")

    try:
        img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
        response = requests.get(img_url, timeout=30)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # --- Add caption text on image ---
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        wrapped = textwrap.fill(prompt, width=30)
        draw.text((10, 10), wrapped, fill="white", font=font)

        st.image(image, caption="✨ AI-generated Comic Panel", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Image generation failed: {e}")
        st.info("💡 Try again with a different prompt or check your internet connection.")

# -----------------------------
# 📘 Footer
# -----------------------------
st.caption("🚀 Powered by Streamlit + OpenAI + Pollinations API")
