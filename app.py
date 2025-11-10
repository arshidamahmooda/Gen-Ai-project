import streamlit as st
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import textwrap
import google.generativeai as genai

# ------------------------------
# 🎨 Streamlit Page Setup
# ------------------------------
st.set_page_config(page_title="🎨 Gemini Comic Generator", layout="centered")
st.title("🎨 AI Comic Generator (Gemini Powered)")
st.markdown("Create short comic stories using **Google Gemini + Pollinations API** 🎭")

# ------------------------------
# 🧠 User Input
# ------------------------------
prompt = st.text_area(
    "✍️ Enter your comic idea:",
    """Frog Prince’s Day Off

Panel 1: The frog prince sneaks out of his castle wearing sunglasses.
Panel 2: He rides a skateboard through the city streets.
Panel 3: He waves to surprised people as he zooms by fountains.
Panel 4: The frog prince jumps into his favorite pond with a happy splash."""
)

# ------------------------------
# 🚀 Generate Comic
# ------------------------------
if st.button("🎬 Generate Comic"):
    st.info("⏳ Generating comic using Google Gemini...")

    # --- Gemini Text Generation ---
    try:
        genai.configure(api_key="YOUR_GEMINI_API_KEY")  # 🔑 Replace with your Gemini key

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            f"Write a short, funny 3-line comic scene based on this idea:\n{prompt}"
        )
        story = response.text
    except Exception as e:
        story = f"⚠️ Error generating story: {e}"

    st.subheader("💬 Comic Storyline")
    st.write(story)

    # --- Generate Image (Pollinations API) ---
    st.subheader("🖼️ Comic Panel")
    img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
    response = requests.get(img_url)
    image = Image.open(BytesIO(response.content))

    # --- Add caption text ---
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    wrapped = textwrap.fill(prompt, width=30)
    draw.text((10, 10), wrapped, fill="white", font=font)

    st.image(image, caption="✨ AI-generated Comic Panel", use_container_width=True)

st.caption("🚀 Powered by Google Gemini + Pollinations + Streamlit")
