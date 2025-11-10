# =====================================================
# 🎨 Lightweight Offline AI Comic Generator (Streamlit)
# =====================================================
import streamlit as st
from transformers import pipeline, set_seed
from PIL import Image, ImageDraw, ImageFont
import torch, textwrap, requests
from io import BytesIO

# -----------------------------------------------------
# ⚙️ App Configuration
# -----------------------------------------------------
st.set_page_config(page_title="🎨 AI Comic Creator", layout="centered")
st.title("🎨 AI Comic Creator")
st.markdown("Generate a short comic scene from your idea ")

# -----------------------------------------------------
# ✍️ User Input
# -----------------------------------------------------
prompt = st.text_area(
    "💭 Enter your comic idea:",
    """Frog Prince’s Day Off

Panel 1: The frog prince sneaks out of his castle wearing sunglasses.
Panel 2: He rides a skateboard through the city streets.
Panel 3: He waves to surprised people as he zooms by fountains.
Panel 4: The frog prince jumps into his favorite pond with a happy splash.""",
    height=150
)

# -----------------------------------------------------
# 🚀 Generate Comic
# -----------------------------------------------------
if st.button("🎬 Generate Comic"):
    st.info("⏳ Generating story... please wait")

    # === GPT-2 TEXT GENERATION ===
    try:
        generator = pipeline("text-generation", model="distilgpt2")
        set_seed(42)
        story_output = generator(
            f"Create a 3-line funny comic scene about: {prompt}",
            max_length=80,
            num_return_sequences=1
        )
        story = story_output[0]["generated_text"].strip()
    except Exception as e:
        story = f"This comic shows: {prompt}."
        st.warning(f"⚠️ Text generation fallback due to: {e}")

    st.subheader("💬 Comic Storyline")
    st.write(story)

    # === IMAGE GENERATION (Pollinations API) ===
    st.subheader("🖼️ Comic Panel")
    try:
        url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))

        # Add caption
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        wrapped = textwrap.fill(prompt, width=30)
        draw.text((10, 10), wrapped, fill="white", font=font)

        st.image(image, caption="✨ AI-generated Comic Panel", use_container_width=True)
    except Exception as e:
        st.error(f"❌ Image generation failed: {e}")

st.caption("🚀 Powered by GPT-2 + Pollinations + Streamlit ")
