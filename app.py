
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model(
    '/content/drive/MyDrive/tomato_disease_model.h5'
)

# Nama kelas
class_names = [
    "Healthy",
    "Miner",
    "Bacterial Spot",
    "Powdery Mildew",
    "Powdery Mildew Miner",
    "Bacterial Spot Miner",
    "White Fly"
]

# Konfigurasi halaman
st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🍅",
    layout="centered"
)

# CSS STYLE
st.markdown("""
<style>

.main {
    background: linear-gradient(to bottom right, #0f172a, #14532d);
    border-radius: 20px;
    padding: 20px;
}

h1 {
    color: white;
    text-align: center;
}

p {
    color: #e2e8f0;
    text-align: center;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #1e293b;
    color: white;
    text-align: center;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# TITLE
st.markdown(
    "<h1>🍅 Tomato Disease Detection</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p>Upload gambar daun tomat untuk mendeteksi penyakit menggunakan AI CNN EfficientNetB0.</p>",
    unsafe_allow_html=True
)

# Upload gambar
uploaded_file = st.file_uploader(
    "📸 Upload Gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption='Gambar Upload',
        use_column_width=True
    )

    # preprocessing
    image = image.resize((224,224))

    image = np.array(image) / 255.0

    image = np.expand_dims(image, axis=0)

    # prediksi
    prediction = model.predict(image)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    # hasil
    st.markdown(f'''
    <div class="result-box">
        <h2>🩺 Hasil Prediksi</h2>
        <h3>{predicted_class}</h3>
        <p>Akurasi: {confidence:.2f}%</p>
    </div>
    ''', unsafe_allow_html=True)
