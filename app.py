import streamlit as st
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import textwrap, requests
from io import BytesIO

st.set_page_config(page_title="AI Comic Creator", layout="centered")

st.title("🎨 AI Comic Creator")
st.markdown("Generate a short comic scene from your text prompt!")

# --- Input box ---
prompt = st.text_area("Enter your comic idea:", "A dog learns to play the guitar on stage!")

if st.button("Generate Comic"):
    st.info("⏳ Generating... please wait 10–15 seconds")

    # --- Generate short story text ---
    story_gen = pipeline("text-generation", model="gpt2", device=-1)
    story = story_gen(prompt, max_length=80, num_return_sequences=1)[0]["generated_text"]
    st.subheader("💬 Comic Story")
    st.write(story)

    # --- Generate comic image (using Pollinations API — no key required) ---
    img_url = f"https://image.pollinations.ai/prompt/{requests.utils.quote(prompt)}"
    response = requests.get(img_url)
    image = Image.open(BytesIO(response.content))

    # Add title text
    draw = ImageDraw.Draw(image)
    wrapped = textwrap.fill(prompt, width=25)
    font = ImageFont.load_default()
    draw.text((10, 10), wrapped, fill="white", font=font)

    st.subheader("🖼️ Comic Panel")
    st.image(image, use_container_width=True)

st.caption("Built with ❤️ Streamlit + GPT-2 + Pollinations AI")
