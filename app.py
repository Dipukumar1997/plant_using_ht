import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import wikipedia
import os
import google.generativeai as genai
import gdown

# ---------------- CONFIGURE GEMINI ---------------- #
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- LOAD MODEL ---------------- #
url = "https://drive.google.com/uc?id=1Ms1HkwFo7im2Yh6V9Hn90jE_qERJl96y"
output = "plant_disease_model.h5"
gdown.download(url, output, quiet=False)
# Load the model
model = tf.keras.models.load_model("plant_disease_model.h5")

# ---------------- CLASS NAMES ---------------- #
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
               'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
               'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
               'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
               'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
               'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
               'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
               'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
               'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
               'Tomato___healthy']

# ---------------- PREDICTION FUNCTION ---------------- #
def predict_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return class_names[np.argmax(prediction)]

# ---------------- GEMINI FALLBACK ---------------- #
def get_gemini_response(prompt):
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini error: {e}"

# ---------------- GET WIKIPEDIA INFO ---------------- #
def get_disease_info(disease_name):
    try:
        summary = wikipedia.summary(disease_name, sentences=5)
        page = wikipedia.page(disease_name)
        sections = page.sections
        keywords = ["control", "treatment", "management", "cure", "prevention", "solution", "therapy", "remedy"]
        cure_section = ""

        for section in sections:
            if any(k in section.lower() for k in keywords):
                cure_section = page.section(section)
                break

        return summary, cure_section

    except Exception:
        with st.spinner("Loading..."):
            fallback_summary = get_gemini_response(f"What is {disease_name}? Explain it in simple language.")
            fallback_cure = get_gemini_response(f"How to treat or manage {disease_name}?")
        return fallback_summary, fallback_cure

# ---------------- STREAMLIT UI ---------------- #
st.set_page_config(layout="wide")
st.title("🌿 Plant Disease Recognition System")

st.header("Upload an Image to Predict Disease")

col1, col2, col3 = st.columns([0.1, 1, 0.1])
with col2:
    st.markdown("""
        <style>
            .stFileUploader {
                width: 100% !important;
                max-width: 500px !important;
                padding: 20px !important;
                border: 2px dashed #ddd !important;
                text-align: center;
            }
            @media (max-width: 768px) {
                .stFileUploader {
                    width: 90% !important;
                    max-width: 100% !important;
                    padding: 15px !important;
                }
            }
        </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📂 Drag & Drop Your Image Here", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Predict"):
            st.write("🔬 Analyzing...")
            prediction = predict_image(image)
            st.success(f"🌱 Prediction: {prediction.replace('_', ' ')}")

            st.write("📚 Fetching more information about the disease...")
            summary, cure_info = get_disease_info(prediction.replace("_", " "))

            st.subheader("📝 Overview")
            st.write(summary)

            st.subheader("💊 Cure / Treatment")
            if cure_info:
                st.write(cure_info)
            else:
                st.info("We couldn't find a specific cure or treatment section. Please consult a plant pathologist.")

# ---------------- ABOUT PAGE BUTTON ---------------- #
st.markdown("""
    <style>
        .centered-button-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .green-box {
            background-color: #4CAF50;
            padding: 12px 24px;
            border-radius: 8px;
            text-align: center;
            display: inline-block;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="centered-button-container">', unsafe_allow_html=True)
st.link_button("📚 Know more about the project", "/About", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- ABOUT SECTION ---------------- #
st.markdown("<h2 id='about-section'>About the Project</h2>", unsafe_allow_html=True)
st.write("""
    This Plant Disease Recognition System uses AI to detect diseases in plants based on images.
    Upload a leaf image, and the system will classify it into categories such as healthy or diseased.
""")