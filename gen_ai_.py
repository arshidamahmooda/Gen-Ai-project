import streamlit as st
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont
import torch

st.set_page_config(page_title="AI Comic Generator 🎨", layout="wide")

st.title("🎭 AI Comic Generator")
st.write("Create your own mini comic by entering a short story or prompt below!")

# User input
story_prompt = st.text_area("✍️ Enter a short story or scene:", 
                            "A robot finds a flower in the desert and learns what emotions mean.")

if st.button("✨ Generate Comic"):
    with st.spinner("Generating captions and images..."):
        # Generate text captions
        text_gen = pipeline("text-generation", model="gpt2")
        captions = text_gen(story_prompt, max_new_tokens=60, num_return_sequences=3)
        captions = [c['generated_text'].strip().split('.')[:2] for c in captions]

        # Initialize diffusion pipeline (low VRAM mode)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "runwayml/stable-diffusion-v1-5"
        image_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
        image_pipe = image_pipe.to(device)

        comic_panels = []
        for idx, cap in enumerate(captions):
            text = '. '.join(cap)
            image = image_pipe(text).images[0]
            comic_panels.append((image, text))

    st.success("✅ Comic generated!")
    st.write("---")

    # Display generated panels
    for i, (img, txt) in enumerate(comic_panels, start=1):
        st.image(img, caption=f"🖼️ Panel {i}: {txt}", use_container_width=True)
