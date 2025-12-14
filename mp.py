import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image  # PIL is used by Streamlit for image handling

# Set the page title and layout
st.set_page_config(page_title="breed classiier", layout="centered")

# 1. DEFINE THE IMAGE SIZE (MUST MATCH TRAINING SIZE)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 2. Load the trained model and class labels
@st.cache_resource  # This decorator caches the model so it's only loaded once
def load_my_model():
    model = load_model('my_image_classifier.h5')
    return model

@st.cache_data  # This decorator caches the class labels
def load_class_labels():
    with open('class_indices.json', 'r') as f:
        class_indices = json.load(f)
    # Convert the dictionary to a list of class names, sorted by their index
    class_labels = [class_name for class_name, index in sorted(class_indices.items(), key=lambda x: x[1])]
    return class_labels

# Load the model and labels
try:
    model = load_my_model()
    class_labels = load_class_labels()
except Exception as e:
    st.error(f"Error loading model or class labels: {e}")
    st.stop()

# 3. Function to make a prediction
def predict_image(img):
    # Resize the image to the target size
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    # Convert the image to a numpy array
    img_array = image.img_to_array(img)
    # Add a batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image pixels
    img_array /= 255.

    # Make the prediction
    prediction = model.predict(img_array, verbose=0)  # verbose=0 silences the output
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    predicted_class = class_labels[predicted_class_index]
    
    return predicted_class, confidence

# 4. Streamlit UI Components
st.title("🖼️ Breed Classification")
st.write("Upload an image and the AI model will predict what it is.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_display = Image.open(uploaded_file)
    st.image(image_display, caption="Uploaded Image", use_container_width=True)
    
    # Add a button to trigger prediction
    if st.button("Classify Image"):
        # Show a spinner while predicting
        with st.spinner('Analyzing the image...'):
            predicted_class, confidence = predict_image(image_display)
        
        # Display the results
        st.success("Prediction Complete!")
        st.markdown(f"**Prediction:** {predicted_class}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        
        # Optional: Show a progress bar for confidence
        st.progress(int(confidence) / 100)