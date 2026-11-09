import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="Driver Drowsiness Detection",
    layout="centered"
)

# Custom CSS to remove scroll and center everything
st.markdown("""
    <style>
        html, body, [class*="css"]  {
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

# Google Drive Model Download
MODE_PATH = "drowsiness_model.h5"
FILLED_ID = "https://drive.google.com/file/d/1Ni2jnBIyyVkCGQ_NjcuLNftSUMhEl84y/view?usp=sharing"

# Load model
class _CompatibleDense(tf.keras.layers.Dense):
    @classmethod
    def from_config(cls, config):
        config = config.copy()
        if config.get("quantization_config") is None:
            config.pop("quantization_config", None)
        return super().from_config(config)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "drowsiness_model.h5",
        custom_objects={"Dense": _CompatibleDense},
        compile=False,
    )

model = load_model()

st.title("🚗 Driver Drowsiness Detection")

uploaded_file = st.file_uploader("Upload Driver Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    confidence = prediction if prediction > 0.5 else 1 - prediction
    confidence = round(confidence * 100, 2)

    st.subheader("Prediction Result")

    if prediction > 0.5:
        st.error(f"😴 Drowsy ({confidence}%)")
    else:
        st.success(f"😊 Awake ({confidence}%)")
