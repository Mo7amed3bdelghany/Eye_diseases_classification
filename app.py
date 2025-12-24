import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2

# ==================================================
# Page Configuration
# ==================================================
st.set_page_config(
    page_title="Eye Disease Classification",
    page_icon="👁️",
    layout="centered"
)

# ==================================================
# Custom CSS (Professional UI)
# ==================================================
st.markdown("""
<style>

body {
    background-color: #f5f7fa;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #1f3c88;
    text-align: center;
}

.sub-title {
    font-size: 20px;
    color: #4a4a4a;
    text-align: center;
    margin-bottom: 35px;
}

.section-title {
    font-size: 26px;
    font-weight: 600;
    color: #1f3c88;
    margin-top: 30px;
}

.card {
    background-color: white;
    padding: 30px;
    border-radius: 18px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.12);
    margin-top: 25px;
}

.result {
    font-size: 30px;
    font-weight: bold;
    color: #006d77;
}

.confidence {
    font-size: 22px;
    color: #2a9d8f;
    margin-top: 10px;
}

.footer {
    text-align: center;
    font-size: 14px;
    color: gray;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# Title
# ==================================================
st.markdown('<div class="main-title">👁️ Eye Disease Classification</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-powered retinal disease detection using Deep Learning & Transfer Learning</div>',
    unsafe_allow_html=True
)


# ==================================================
# Load Model
# ==================================================
@st.cache_resource
def load_model():
    return keras.models.load_model("BestModel.h5")


model = load_model()

# ==================================================
# Class Names
# ==================================================
class_names = {
    0: "Glaucoma",
    1: "Normal",
    2: "Diabetic Retinopathy",
    3: "Cataract"
}


# ==================================================
# Image Preprocessing
# ==================================================
def preprocess_image(image):
    image = np.array(image)

    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    image = cv2.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    return image


# ==================================================
# Upload Section
# ==================================================
st.markdown('<div class="section-title">📤 Upload Retinal Image</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an eye image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

# ==================================================
# Prediction
# ==================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Retinal Image", use_column_width=True)

    if st.button("🔍 Analyze Image"):
        with st.spinner("Analyzing image..."):
            progress = st.progress(0)
            for i in range(100):
                progress.progress(i + 1)

            processed_image = preprocess_image(image)
            predictions = model.predict(processed_image)

            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions) * 100

        # ==================================================
        # Result Card
        # ==================================================
        st.markdown(f"""
        <div class="card">
            <div class="result">🧠 Diagnosis: {class_names[predicted_class]}</div>
            <div class="confidence">Confidence: {confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # ==================================================
        # Probabilities
        # ==================================================
        st.markdown('<div class="section-title">📊 Prediction Probabilities</div>', unsafe_allow_html=True)

        for i, prob in enumerate(predictions[0]):
            st.metric(
                label=class_names[i],
                value=f"{prob*100:.2f}%"
            )

        # ==================================================
        # Disclaimer
        # ==================================================
        st.warning(
            "⚠️ This application is for educational purposes only and should not be used as a substitute for professional medical diagnosis."
        )

# ==================================================
# Footer
# ==================================================
st.markdown("""
<div class="footer">
<h6>Developed by Mohamed Abdelghany | Deep Learning & Medical AI</h6>
</div>
""", unsafe_allow_html=True)
