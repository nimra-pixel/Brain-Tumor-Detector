import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.keras")

model = load_model()
IMG_SIZE = 128

st.title("🧠 Brain Tumor Classification")
st.markdown("Upload a **brain MRI scan** to detect whether a tumor is present.")
st.divider()

uploaded_file = st.file_uploader("📤 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)

    with st.spinner("Analyzing..."):
        prediction = model.predict(img)[0][0]

    confidence = float(prediction) if prediction > 0.5 else float(1 - prediction)

    st.divider()
    if prediction > 0.5:
        st.error(f"🔴 **Tumor Detected**")
    else:
        st.success(f"🟢 **No Tumor Detected**")

    st.metric("Confidence", f"{confidence * 100:.2f}%")
    st.caption("⚠️ For educational purposes only. Always consult a medical professional.")