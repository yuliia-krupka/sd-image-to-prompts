import gc
import torch
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer

BLIP_PATH = "/Users/juliakrupka/.cache/kagglehub/datasets/yuliiakrupka/my-blip-models/versions/1/blip-large"
MINILM_PATH = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_TOKENS = 85
NUM_BEAMS = 10


@st.cache_resource(show_spinner=False)
def load_blip():
    processor = BlipProcessor.from_pretrained(BLIP_PATH)
    model = BlipForConditionalGeneration.from_pretrained(
        BLIP_PATH, torch_dtype=DTYPE, ignore_mismatched_sizes=True
    ).to(DEVICE).eval()
    return processor, model


@st.cache_resource(show_spinner=False)
def load_minilm():
    return SentenceTransformer(MINILM_PATH)


def generate_caption(image: Image.Image, processor, model) -> str:
    inputs = processor(image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            num_beams=NUM_BEAMS,
            repetition_penalty=1.5,
            length_penalty=1.2,
            early_stopping=True,
        )
    gc.collect()
    return processor.decode(out[0], skip_special_tokens=True)


st.set_page_config(page_title="Image → Prompt", page_icon="🔍", layout="centered")

st.title("🔍 Image → Prompt")
st.caption("Upload an image and the model will guess what prompt could have generated it.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.subheader("Input image")
        st.image(image, width="stretch")

    with col2:
        st.subheader("Generated prompt")

        if st.button("🔍 Guess the prompt", use_container_width=True):
            with st.spinner("Analysing image…"):
                blip_processor, blip_model = load_blip()
                caption = generate_caption(image, blip_processor, blip_model)

                st_model = load_minilm()
                embedding = st_model.encode(caption)

            st.session_state["caption"] = caption
            st.session_state["embedding"] = embedding

        if "caption" in st.session_state:
            st.text_area("", value=st.session_state["caption"], height=200, label_visibility="collapsed")

            with st.expander("Embedding (384-dim vector)"):
                st.write(st.session_state["embedding"])
else:
    st.info("⬆️ Upload an image to get started", icon="🖼️")
