# =====================================================
# 🎨 Lightweight Offline AI Comic Generator (Streamlit)
# =====================================================
import streamlit as st
from transformers import pipeline, set_seed
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import torch, textwrap, tempfile

# -----------------------------------------------------
# ⚙️ App Configuration
# -----------------------------------------------------
st.set_page_config(page_title="🎨 AI Comic Creator", layout="centered")
st.title("🎨 Lightweight AI Comic Creator")
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
        generator = pipeline("text-generation", model="gpt2")
        set_seed(42)
        story_output = generator(
            f"Create a 3-line funny comic scene about: {prompt}",
            max_length=80,
            num_return_sequences=1
        )
        story = story_output[0]["generated_text"].strip()
        st.subheader("💬 Comic Storyline")
        st.write(story)
    except Exception as e:
        st.warning(f"Text generation failed: {e}")
        story = prompt

    # === IMAGE GENERATION ===
    st.subheader("🖼️ Comic Panel")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Use a lighter diffusion model
        model_id = "stabilityai/sd-turbo"  # smaller and faster
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe = pipe.to(device)

        with torch.inference_mode():
            image = pipe(prompt).images[0]

        # Add text overlay
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        wrapped = textwrap.fill(story[:200], width=40)
        draw.text((10, 10), wrapped, fill="white", font=font)

        st.image(image, caption="✨ Comic-Style AI Image", use_container_width=True)

        # Save + download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            image.save(tmpfile.name)
            st.download_button("📥 Download Comic", tmpfile.name, "comic.png")
    except Exception as e:
        st.warning(f"Stable Diffusion not supported here: {e}")
        st.info("💡 Using lightweight online generator (Pollinations)...")
        import requests
        from io import BytesIO
        try:
            url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}"
            response = requests.get(url)
            img = Image.open(BytesIO(response.content))
            st.image(img, caption="✨ Comic (Generated via Pollinations)")
        except:
            st.error("❌ Image generation failed. Try again later.")

st.caption("🚀 Powered by GPT-2 + sd-turbo + Streamlit (no API key needed)")
