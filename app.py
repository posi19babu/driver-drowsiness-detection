import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
from PIL import Image

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
        html, body, [class*="css"] {
            height: 100%;
            overflow: hidden;
        }

        .block-container {
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 90vh;
            text-align: center;
        }

        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# Google Drive Model Details
MODEL_PATH = "drowsiness_model.h5"

FILE_ID = "1Ni2jnBIyyVkCGQ_NjcuLNftSUMhEl84y"

MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"


# Custom Dense Compatibility
class _CompatibleDense(tf.keras.layers.Dense):

    @classmethod
    def from_config(cls, config):
        config = config.copy()

        if config.get("quantization_config") is None:
            config.pop("quantization_config", None)

        return super().from_config(config)


# Download and Load Model
@st.cache_resource
def load_model():

    # Download model if not available
    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading AI Model... Please wait"):

            downloaded_file = gdown.download(
                MODEL_URL,
                MODEL_PATH,
                quiet=False
            )

            if downloaded_file is None:
                raise RuntimeError(
                    "Model download failed. Check Google Drive permissions."
                )

    # Check model file
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found.")

    if os.path.getsize(MODEL_PATH) == 0:
        raise RuntimeError("Downloaded model file is empty.")

    # Load model
    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"Dense": _CompatibleDense},
        compile=False
    )


# Load Model
model = load_model()


# Application UI
st.title("🚗 Driver Drowsiness Detection")

uploaded_file = st.file_uploader(
    "Upload Driver Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, width=300)

    # Preprocess Image
    img = image.resize((224, 224))

    img_array = np.array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)[0][0]

    confidence = prediction if prediction > 0.5 else 1 - prediction

    confidence = round(float(confidence) * 100, 2)

    st.subheader("Prediction Result")

    if prediction > 0.5:
        st.error(f"😴 Drowsy ({confidence}%)")

    else:
        st.success(f"😊 Awake ({confidence}%)")
