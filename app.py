import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import textwrap, torch

st.set_page_config(page_title="AI Comic Generator", layout="centered")

st.title("🎨 AI Comic Generator")
st.markdown("Generate a mini comic scene using AI (GPT-2 for story + Stable Diffusion for images).")

# --- Text Input ---
prompt = st.text_area("Enter your comic idea or scene:", "A superhero cat saves the city from robot dogs!")

if st.button("Generate Comic"):
    with st.spinner("Generating comic story..."):
        # --- Text Generation ---
        text_gen = pipeline("text-generation", model="gpt2")
        story = text_gen(prompt, max_length=80, num_return_sequences=1)[0]["generated_text"]
        st.subheader("🗨️ Comic Story")
        st.write(story)

    with st.spinner("Generating comic image... (takes ~30s on CPU)"):
        # --- Image Generation ---
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float32
        )
        pipe.to("cpu")

        image = pipe(prompt).images[0]

        # Add caption on image
        draw = ImageDraw.Draw(image)
        wrapped_text = textwrap.fill(prompt, width=30)
        font = ImageFont.load_default()
        draw.text((10, 10), wrapped_text, fill="white", font=font)

        st.subheader("🖼️ Generated Comic Panel")
        st.image(image, use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + Hugging Face (GPT-2 + Stable Diffusion)")
