# app.py
import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Judul aplikasi
st.title("😊 Deteksi Ekspresi Wajah")
st.write("Upload gambar wajah untuk deteksi emosi!")

# Load model CNN
model = load_model('emotion_model.h5')  # Pastikan file model ada di folder yang sama
emotion_labels = ['Marah', 'Jijik', 'Takut', 'Senang', 'Sedih', 'Terkejut', 'Netral']

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Baca gambar
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (48, 48)) / 255.0  # Sesuaikan dengan input model
    gray = np.expand_dims(gray, axis=[0, -1])  # Tambah dimensi batch & channel

    # Prediksi
    pred = model.predict(gray)[0]
    emotion = emotion_labels[np.argmax(pred)]
    confidence = np.max(pred) * 100

    # Tampilkan hasil
    st.image(image, caption=f"Hasil: {emotion}", width=300)
    st.write(f"**Ekspresi:** {emotion}")
    st.write(f"**Akurasi:** {confidence:.2f}%")
