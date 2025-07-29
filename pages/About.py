# pages/About.py
import streamlit as st

st.set_page_config(page_title="About", layout="wide")

st.title("🌿 About the Plant Disease Recognition System")

st.markdown("""
## 🔬 Project Overview

This AI-powered system is designed to **detect plant diseases** from leaf images using a **Deep Learning model** and then provide detailed, reliable information sourced from **Wikipedia** and **Google's Gemini AI**. 

The goal is to empower **farmers, agricultural researchers, and gardeners** with fast, accurate, and accessible disease diagnosis to improve crop health and yield. 🌾

---

## 🚀 Why This Project?

In a world where food security is increasingly critical, **early disease detection** in crops is essential. However, most farmers and cultivators **lack access to expert diagnostics**.

With the integration of AI, this project makes it possible to:
- **Detect disease instantly from an image**
- **Educate users with trustworthy, curated knowledge**
- **Suggest remedies or next steps for management**

---

## 🤖 AI + ML Magic Behind the Scenes

- **Image Classification**: A Convolutional Neural Network (CNN) model trained on the famous *PlantVillage* dataset.
- **Natural Language Generation**: Using **Gemini 1.5 Flash** by Google to provide fallback information when Wikipedia fails.
- **Knowledge Retrieval**: Smart scraping from **Wikipedia** for factual disease details, symptoms, and treatment methods.

---

## 🔧 Technologies Used

| Tool | Purpose |
|------|---------|
| 🧠 TensorFlow/Keras | Deep learning model for image classification |
| 🎨 Streamlit | Frontend UI & web deployment |
| 🧬 Gemini 1.5 Flash | LLM-based explanation and fallback info |
| 📚 Wikipedia API | Primary source for disease and cure info |
| 🖼️ Pillow (PIL) | Image handling |
| 🧮 NumPy | Array transformations |

---

## 📈 Key Features

- ✅ Upload leaf images for real-time disease prediction  
- ✅ View name, symptoms, and cure of predicted disease  
- ✅ Dual-source knowledge via Wikipedia and Gemini  
- ✅ Fully responsive and mobile-friendly  
- ✅ Minimal UI with custom styling  
- ✅ Offline and fast predictions  

---

## 🧠 Smart Fallback with AI

Whenever Wikipedia fails to provide info, the system triggers **Gemini** to generate an intelligent response using the prompt:

> "Explain the disease _[name]_ in simple language and suggest how to treat it."

This ensures that users always get a meaningful answer — no dead ends. ✨

---

## 📍 Real-World Applications

- 🚜 Agriculture & Crop Science  
- 🧪 Agricultural Research Labs  
- 📱 Mobile diagnosis for rural farmers  
- 🎓 AI + Biology Education Projects  
- 💡 Hackathons & Tech Exhibitions

---

## 👨‍💻 Developer’s Note

> “This project combines AI, computer vision, and accessible design to address a real-world problem. It’s more than just a tech demo – it’s a step toward empowering those who feed the world.”

Built with ❤️ for Hackathons and Innovation Showcases.

---

## 📬 Get In Touch

If you're interested in contributing, collaborating, or showcasing the project:

- GitHub: https://github.com/Dipukumar1997
- Email: dk95074450@gmail.com

---

## 📌 AI-Powered Prediction System

> This app uses **AI-generated insights** to enhance factual knowledge and offers **automated diagnosis** — a new step forward in smart agriculture.

---  

💚 *Let’s make agriculture smarter, one leaf at a time.*
""")
