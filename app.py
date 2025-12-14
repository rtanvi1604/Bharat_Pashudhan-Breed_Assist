import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
import time
import base64
import urllib.request

# Set page configuration
st.set_page_config(
    page_title="Bharat Pashudhan • Breed Assist",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "https://via.placeholder.com/78x78/1E40AF/FFFFFF?text=BP",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define the image size (must match training size)
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Function to encode image to base64
def get_base64_encoded_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    return None

# Function to load Google Fonts
def load_google_fonts():
    # Load Poppins font from Google Fonts
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# Custom CSS to match the design
# Custom CSS to match the design - NEON VERSION
def inject_custom_css():
    # Try to load logo as base64 if it exists
    logo_base64 = get_base64_encoded_image("assets/logo.png")
    logo_url = "data:image/png;base64," + logo_base64 if logo_base64 else "https://via.placeholder.com/78x78/1E40AF/FFFFFF?text=BP"
    
    # Load Google Fonts
    load_google_fonts()
    
    st.markdown(f"""
    <style>
    :root {{
        --neon-blue: #00f3ff;
        --neon-pink: #ff00ff;
        --neon-purple: #bd00ff;
        --neon-green: #00ff66;
        --neon-cyan: #00ffea;
        --dark-bg: #0a0a1a;
        --darker-bg: #050510;
        --card-bg: rgba(15, 15, 35, 0.8);
        --text-light: #ffffff;
        --text-muted: #a0a0c0;
        --border-neon: rgba(0, 243, 255, 0.3);
    }}
    
    /* Neon glow animations */
    @keyframes neonGlow {{
        0%, 100% {{ 
            text-shadow: 0 0 5px var(--neon-blue), 
                        0 0 10px var(--neon-blue),
                        0 0 15px var(--neon-blue),
                        0 0 20px var(--neon-purple);
        }}
        50% {{ 
            text-shadow: 0 0 10px var(--neon-cyan), 
                        0 0 20px var(--neon-cyan),
                        0 0 30px var(--neon-cyan),
                        0 0 40px var(--neon-pink);
        }}
    }}
    
    @keyframes pulse {{
        0%, 100% {{ 
            box-shadow: 0 0 10px var(--neon-blue),
                       0 0 20px var(--neon-blue);
            transform: scale(1);
        }}
        50% {{ 
            box-shadow: 0 0 20px var(--neon-cyan),
                       0 0 40px var(--neon-cyan);
            transform: scale(1.05);
        }}
    }}
    
    @keyframes float {{
        0%, 100% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-8px); }}
    }}
    
    @keyframes shimmer {{
        0% {{ background-position: -200% 0; }}
        100% {{ background-position: 200% 0; }}
    }}
    
    @keyframes borderGlow {{
        0%, 100% {{ 
            border-color: var(--neon-blue);
            box-shadow: 0 0 10px var(--neon-blue);
        }}
        50% {{ 
            border-color: var(--neon-cyan);
            box-shadow: 0 0 20px var(--neon-cyan);
        }}
    }}
    
    * {{ 
        box-sizing: border-box; 
        font-family: 'Poppins', sans-serif !important;
    }}
    
    html, body {{
        height: 100%;
        margin: 0;
        background: linear-gradient(135deg, var(--darker-bg) 0%, var(--dark-bg) 100%);
        color: var(--text-light) !important;
        font-family: 'Poppins', sans-serif !important;
    }}
    
    /* Target all Streamlit text elements */
    .stApp, .stMarkdown, .stText, .stTitle, .stHeader, .stSubheader,
    h1, h2, h3, h4, h5, h6, p, div, span {{
        color: var(--text-light) !important;
        font-family: 'Poppins', sans-serif !important;
    }}
    
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 95% !important;
        background: transparent !important;
    }}
    
    /* Neon topbar */
    .topbar {{
        background: rgba(10, 10, 30, 0.95) !important; /* Less transparent */
        backdrop-filter: blur(5px) !important; /* Reduced blur */
        border-bottom: 2px solid var(--neon-blue);
        padding: 12px 30px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.3);
        width: 100vw;
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        z-index: 1000;
        animation: borderGlow 3s infinite;
    }}
    
    .navbar-center h1 {{
        background: linear-gradient(45deg, #00ffea, #bd00ff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        font-size: 26px !important;
        font-weight: 800 !important;
        white-space: nowrap;
        text-shadow: none !important;
        filter: none !important;
        animation: none !important;
    }}
                
    .navbar {{
        display: flex;
        width: 100%;
        align-items: center;
        justify-content: space-between;
        position: relative;
    }}

    .navbar-left {{
        flex: 1;
        display: flex;
        justify-content: flex-start;
    }}

    .navbar-center {{
        flex: 2;
        text-align: center;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
    }}
                
    .navbar-right {{
        flex: 1;
        display: flex;
        justify-content: flex-end;
        gap: 20px;
    }}
    
    .navbar-center p {{
        color: #00ffea !important;
        font-size: 14px;
        margin: 5px 0 0 0;
        text-shadow: 0 0 10px #00ffea;
        white-space: nowrap;
        animation: none !important;
    }}
    
    .navbar-left img {{
        width: 78px !important;  /* Bigger logo */
        height: 78px !important;
        transition: all 0.3s ease;
        object-fit: contain !important; /* Ensure no cropping */
    }}

    .navbar-left img:hover {{
        transform: scale(1.1) rotate(5deg);
        filter: drop-shadow(0 0 10px var(--neon-cyan));
    }}

    /* Neon containers */
    .container {{
        background: var(--card-bg) !important;
        border: 2px solid var(--neon-blue);
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0, 243, 255, 0.2),
        inset 0 0 20px rgba(0, 243, 255, 0.1);
        padding: 25px 30px;
        margin-bottom: 24px;
        font-family: 'Poppins', sans-serif !important;
        color: var(--text-light) !important;
        backdrop-filter: blur(10px);
        animation: borderGlow 4s infinite;
        transition: all 0.3s ease;
    }}
    
    .container:hover {{
        transform: translateY(-5px);
        box-shadow: 0 0 30px rgba(0, 243, 255, 0.4),
        inset 0 0 30px rgba(0, 243, 255, 0.15);
    }}
    
    .card__title {{
        color: #ffffff !important;  /* White color */
        background: none !important;
        -webkit-text-fill-color: #ffffff !important;
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.7) !important;
        font-size: 20px !important;
        margin: 0 0 10px 0 !important;
    }}

    
    .muted {{
        color: var(--text-muted) !important;
        font-size: 14px;
    }}
    
    /* Neon file uploader */
    [data-testid="stFileUploader"] section {{
        background: rgba(15, 15, 35, 0.6) !important;
        border: 2px dashed var(--neon-purple) !important;
        border-radius: 16px !important;
        padding: 25px !important;
        backdrop-filter: blur(10px);
        animation: borderGlow 3s infinite;
        transition: all 0.3s ease;
    }}
    
    [data-testid="stFileUploader"] section:hover {{
        border-color: var(--neon-pink) !important;
        box-shadow: 0 0 25px rgba(255, 0, 255, 0.3);
        transform: scale(1.02);
    }}
    
    [data-testid="stFileUploaderDropzone"] div {{
        color: var(--text-light) !important;
        font-weight: 500;
    }}
    
    [data-testid="stFileUploader"] button {{
        background: linear-gradient(45deg, var(--neon-purple), var(--neon-pink)) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 10px 20px !important;
        border: none !important;
        transition: all 0.3s ease;
        box-shadow: 0 0 15px rgba(189, 0, 255, 0.4);
    }}
    
    [data-testid="stFileUploader"] button:hover {{
        background: linear-gradient(45deg, var(--neon-pink), var(--neon-cyan)) !important;
        box-shadow: 0 0 25px rgba(255, 0, 255, 0.6);
        transform: translateY(-2px);
    }}
    
    /* Neon buttons */
    .stButton > button {{
        background: linear-gradient(45deg, var(--neon-purple), var(--neon-pink)) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 14px 28px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 20px rgba(189, 0, 255, 0.4) !important;
        position: relative;
        overflow: hidden;
    }}
    
    .stButton > button:hover {{
        background: linear-gradient(45deg, var(--neon-cyan), var(--neon-green)) !important;
        box-shadow: 0 0 30px rgba(0, 255, 234, 0.6) !important;
        transform: translateY(-3px) scale(1.05) !important;
    }}
    
    .stButton > button:active {{
        transform: translateY(0) scale(1) !important;
    }}
    
    /* Neon progress bars */
    .progress {{
        height: 14px;
        background: rgba(0, 243, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid var(--neon-blue);
        box-shadow: inset 0 0 10px rgba(0, 243, 255, 0.2);
    }}
    
    .progress__fill {{
        height: 100%;
        background: linear-gradient(90deg, var(--neon-purple), var(--neon-pink), var(--neon-cyan));
        width: var(--target-width, 0%);
        animation: growBar 1.2s ease-out forwards, shimmer 2s infinite;
        box-shadow: 0 0 10px var(--neon-pink);
    }}
    
    @keyframes growBar {{
        from {{ width: 0%; opacity: 0.5; }}
        to {{ width: var(--target-width); opacity: 1; }}
    }}
    
    /* Neon badge */
    .badge {{
        padding: 6px 12px !important;  /* Smaller padding */
        font-size: 11px !important;    /* Smaller font */
        font-weight: 700 !important;
        position: absolute;
        right: 15px;
        top: 15px;
    }}
    
    /* Navigation links */
    .nav__link {{
        color: var(--neon-cyan) !important;
        text-decoration: none;
        font-weight: 600;
        padding: 10px 18px;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-shadow: 0 0 10px var(--neon-cyan);
        position: relative;
    }}
    
    .nav__link:hover {{
        color: var(--neon-pink) !important;
        background: rgba(255, 0, 255, 0.1);
        text-shadow: 0 0 15px var(--neon-pink);
        transform: translateY(-2px);
    }}
    
    /* Footer neon style */
    .footer {{
        padding: 30px 0;
        text-align: center;
        margin-top: 3rem;
    }}
    
    .footer-text {{
        max-width: 90%;
        white-space: normal !important;
        word-wrap: break-word !important;
        line-height: 1.5 !important;
        display: inline-block !important;
    }}
    
    .footer-text:hover {{
        background: linear-gradient(45deg, var(--neon-purple), var(--neon-pink));
        color: white;
        transform: translateY(-3px);
        box-shadow: 0 0 30px rgba(189, 0, 255, 0.4);
        text-shadow: 0 0 15px white;
    }}
    
    .team-member {{
        color: var(--neon-green) !important;
        text-shadow: 0 0 8px var(--neon-green) !important;
        transition: all 0.3s ease !important;
        position: relative !important;
        display: inline-block !important;
    }}

    .team-member:hover {{
        color: var(--neon-pink) !important;
        text-shadow: 0 0 12px var(--neon-pink) !important;
    }}
    
    /* Floating neon elements */
    .floating-element {{
        position: absolute;
        width: 120px;
        height: 120px;
        background: radial-gradient(circle, var(--neon-purple) 0%, transparent 70%);
        border-radius: 50%;
        filter: blur(25px);
        opacity: 0.3;
        z-index: -1;
        animation: float 8s ease-in-out infinite;
    }}
    
    .floating-element:nth-child(1) {{
        top: 20%;
        left: 10%;
        background: radial-gradient(circle, var(--neon-blue) 0%, transparent 70%);
        animation-delay: 0s;
    }}
    
    .floating-element:nth-child(2) {{
        top: 60%;
        right: 15%;
        background: radial-gradient(circle, var(--neon-pink) 0%, transparent 70%);
        animation-delay: 2s;
    }}
    
    .floating-element:nth-child(3) {{
        bottom: 20%;
        left: 20%;
        background: radial-gradient(circle, var(--neon-green) 0%, transparent 70%);
        animation-delay: 4s;
    }}
    
    /* Responsive design */
    @media (max-width: 768px) {{
        .navbar {{
            display: flex;
            width: 100%;
            align-items: center;
            justify-content: space-between;
    }}

        .navbar-center {{
            flex: 1;
            text-align: center;
            margin: 0 auto;
        }}
        
        .topbar {{
            padding: 15px 20px;
        }}
        
        .navbar-center h1 {{
            font-size: 22px !important;
        }}
    }}
    </style>
    
    <!-- Add floating neon background elements -->
    <div class="floating-element"></div>
    <div class="floating-element"></div>
    <div class="floating-element"></div>
    """, unsafe_allow_html=True)

# Load the trained model and class labels
@st.cache_resource
def load_my_model():
    try:
        model = load_model('my_image_classifier.h5')
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_labels():
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        class_labels = [class_name for class_name, index in sorted(class_indices.items(), key=lambda x: x[1])]
        return class_labels
    except Exception as e:
        st.error(f"❌ Error loading class labels: {e}")
        return []

# Function to make a prediction
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
        
        # Get all predictions for the bar chart
        all_predictions = []
        for i, prob in enumerate(prediction[0]):
            all_predictions.append({
                'breed': class_labels[i],
                'confidence': prob * 100
            })
        
        # Sort by confidence
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predicted_class, confidence, all_predictions, True
    except Exception as e:
        return f"Error: {str(e)}", 0, [], False

# Main application
def main():
    inject_custom_css()
    
    # Try to load logo as base64 if it exists
    logo_base64 = get_base64_encoded_image("assets/logo.png")
    logo_src = f"data:image/png;base64,{logo_base64}" if logo_base64 else "https://via.placeholder.com/78x78/1E40AF/FFFFFF?text=BP"
    
    # Header
    
    st.markdown(f"""
    <div class="topbar">
        <div class="navbar">
            <div class="navbar-left">
                <img src="{logo_src}" alt="Logo" style="width: 78px; height: 78px; object-fit: contain;" />
            </div>
            <div class="navbar-center">
                <h1>Bharat Pashudhan • Breed Assist</h1>
                <p>Image-based breed recognition (MVP)</p>
            </div>
            <div class="navbar-right">
                <a href="#" class="nav__link">Help</a>
                <a href="#" class="nav__link">About</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and class labels
    global model, class_labels
    model = load_my_model()
    class_labels = load_class_labels()
    
    if model is None:
        st.error("Model could not be loaded. Please check if 'my_image_classifier.h5' exists in the current directory.")
        return
    
    if not class_labels:
        st.error("Class labels could not be loaded. Please check if 'class_indices.json' exists in the current directory.")
        return
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="container">
            <h2 class="card__title">📸 Upload Cattle/Buffalo Image</h2>
            <p class="muted">JPEG/PNG up to 200MB. Good light • Full body or side profile • Avoid blur.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            # Display image preview
            image_display = Image.open(uploaded_file)
            st.image(image_display, caption="Selected image preview", use_container_width=True)
            
            # Prediction button
            if st.button("🔍 Predict Breed", use_container_width=True):
                with st.spinner('Analyzing the image...'):
                    predicted_class, confidence, all_predictions, success = predict_image(image_display)
                
                if success:
                    # Display results in the second column
                    with col2:
                        st.markdown(f"""
                        <div class="container" style="position: relative;">
                            <h2 class="card__title">✅ Prediction Result</h2>
                            <div class="result__top" style="position: relative;">
                                <div>
                                    <strong>{predicted_class}</strong>
                                    <div class="muted">with {confidence:.2f}% confidence</div>
                                </div>
                                <div class="badge">Top match</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display confidence bars for top predictions
                        # max_confidence = all_predictions[0]['confidence']
                        st.markdown("<div class='result__bars'>", unsafe_allow_html=True)
                        for i, pred in enumerate(all_predictions[:5]):  # Show top 5 predictions
                            # Create a progress bar for each breed
                            # bar_width = (pred['confidence'] / max_confidence) * 100 
                            st.markdown(f"""
                            <div class="bars__row">
                                <div>{pred['breed']}</div>
                                <div class="progress">
                                    <div class="progress__fill" style="--target-width: {pred['confidence']}%"></div>
                                </div>
                                <div class="percent">{pred['confidence']:.1f}%</div>
                            </div>
                            """, unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Additional breed information (placeholder)
                        st.markdown("""
                        <div class="breedinfo">
                            <h3>About this breed</h3>
                            <p class="muted">Breed information will be displayed here once available in our database.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.error(f"Prediction failed: {predicted_class}")
        
        # Tips section
        st.markdown("""
        <div class="container">
            <details class="tips">
                <summary>How to take a good photo (tips)</summary>
                <ul>
                    <li>Take in daylight; avoid strong backlight.</li>
                    <li>Try to capture the whole animal (side view).</li>
                    <li>Keep the camera steady to avoid blur.</li>
                </ul>
            </details>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if uploaded_file is None:
            st.markdown("""
            <div class="container">
                <h2 class="card__title">✅ Prediction Result</h2>
                <p class="muted">Upload an image to see prediction results here.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <span class="footer-text">
            <span>MVP for SIH • By </span>
            <span class="team-member">SYNTAX SERPENTS</span><span> • </span>
            <span class="team-member">MIDHILESH</span><span> • </span>
            <span class="team-member">ROHIT</span><span> • </span>
            <span class="team-member">MOHAN RAJ</span><span> • </span>
            <span class="team-member">KUMARAN</span><span> • </span>
            <span class="team-member">TANVI</span><span> • </span>
            <span class="team-member">SRILEKHA</span>
        </span>
    </div>
    """, unsafe_allow_html=True)
    

if __name__ == "__main__":
    main()