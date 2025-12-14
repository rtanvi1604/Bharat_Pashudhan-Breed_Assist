import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import os

# Set the page title and layout
st.set_page_config(page_title="Breed Classifier", layout="centered")

# 1. DEFINE THE IMAGE SIZE (MUST MATCH TRAINING SIZE)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# 2. Load the trained model and class labels
@st.cache_resource
def load_my_model():
    try:
        model = load_model('my_image_classifier.h5')
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_labels():
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_labels = [class_name for class_name, index in sorted(class_indices.items(), key=lambda x: x[1])]
        st.sidebar.success(f"✅ Loaded {len(class_labels)} classes!")
        return class_labels
    except Exception as e:
        st.sidebar.error(f"❌ Error loading class labels: {e}")
        return []

# Load the model and labels
model = load_my_model()
class_labels = load_class_labels()

# 3. Function to make a prediction
def predict_image(img):
    try:
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        prediction = model.predict(img_array, verbose=0)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100
        
        predicted_class = class_labels[predicted_class_index]
        
        return predicted_class, confidence, True
    except Exception as e:
        return f"Error: {str(e)}", 0, False

# 4. Streamlit UI Components
st.title("🐕 Breed Classification")
st.write("Upload a dog image and the AI model will predict the breed.")

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    try:
        image_display = Image.open(uploaded_file)
        st.image(image_display, caption="Uploaded Image", use_container_width=True)
        
        # Add a button to trigger prediction
        if st.button("🔍 Classify Breed", type="primary"):
            with st.spinner('Analyzing the image...'):
                predicted_class, confidence, success = predict_image(image_display)
            
            if success:
                # Display the results
                st.success("🎯 Prediction Complete!")
                
                # Create columns for better layout
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Results")
                    st.markdown(f"**Predicted Breed:** {predicted_class}")
                    st.markdown(f"**Confidence Score:** {confidence:.2f}%")
                
                with col2:
                    st.markdown("### 🎯 Accuracy Assessment")
                    # Display accuracy status with appropriate color
                    if confidence >= 90:
                        accuracy_status = "🎯 Very High Accuracy"
                        st.success(accuracy_status)
                    elif confidence >= 75:
                        accuracy_status = "✅ High Accuracy"
                        st.success(accuracy_status)
                    elif confidence >= 60:
                        accuracy_status = "⚠️ Moderate Accuracy"
                        st.warning(accuracy_status)
                    elif confidence >= 40:
                        accuracy_status = "🔍 Low Accuracy"
                        st.warning(accuracy_status)
                    else:
                        accuracy_status = "❌ Very Low Accuracy"
                        st.error(accuracy_status)
                
                
                # Progress bar
                st.markdown("### 📈 Confidence Level")
                st.progress(int(confidence )/ 100)
                st.caption(f"Model confidence: {confidence:.1f}%")
                
                # Additional feedback based on confidence
                if confidence >= 80:
                    st.info("💡 The model is very confident in this prediction!")
                elif confidence >= 50:
                    st.info("💡 The model is somewhat confident in this prediction.")
                else:
                    st.warning("⚠️ The model has low confidence. Consider using a clearer image.")
                
            else:
                st.error(f"Prediction failed: {predicted_class}")
                
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

else:
    st.info("👆 Please upload a dog image to get started")
    st.write("Supported formats: JPG, JPEG, PNG")

# Footer
st.markdown("---")
st.caption("Dog Breed Classification App | Powered by TensorFlow & Streamlit")