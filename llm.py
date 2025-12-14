# ==========================
# streamlit_app.py
# ==========================
import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf
import os

# Load your trained model
model = tf.keras.models.load_model("C:/Users/lohit/OneDrive/Desktop/IMG_CLASSIFICATION/Cattle Breeds")

# Class labels (from training)
class_indices = {
    0: "Ayrshire cattle",
    1: "Holstein Friesian",
    2: "Jersey",
    3: "Sahiwal",
    4: "Gir",
    # 👉 Add all breeds as per your dataset folders
}

img_size = 224

# ==========================
# Prediction Function
# ==========================
def predict_cattle(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(img_size, img_size))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]

    return class_indices[predicted_class]

# ==========================
# Streamlit UI
# ==========================
st.title("🐄 Cattle Breed Classifier")
st.write("Upload a cattle photo and I’ll tell you the breed.")

uploaded_file = st.file_uploader("Choose a cattle image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("🔎 Analyzing...")
    breed = predict_cattle(uploaded_file)
    st.success(f"✅ Predicted Cattle Breed: **{breed}**")
