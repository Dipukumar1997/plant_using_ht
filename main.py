# import os
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# from dotenv import load_dotenv
# import wikipedia
# import tensorflow as tf
# from PIL import Image
# import numpy as np
# import io
# import gdown
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import HTMLResponse, FileResponse
# from fastapi.staticfiles import StaticFiles
# import uvicorn
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import wikipedia
# import gdown
# from gemini import get_cure
# from gemini import get_gemini_response  # your Gemini fallback handler

# load_dotenv()
# app = FastAPI()

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# url = "https://drive.google.com/uc?id=1Ms1HkwFo7im2Yh6V9Hn90jE_qERJl96y"
# output = "plant_disease_model.h5"
# gdown.download(url, output, quiet=False)
# model = tf.keras.models.load_model("plant_disease_model.h5")
# class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
#                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
#                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
#                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
#                'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
#                'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
#                'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
#                'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
#                'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
#                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
#                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
#                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
#                'Tomato___healthy']  # your list of class names

# def predict_image(image_data):
#     image = Image.open(io.BytesIO(image_data)).resize((128, 128))
#     img_array = np.array(image) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     return class_names[np.argmax(prediction)]

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image_data = await file.read()
#     prediction = predict_image(image_data)
#     return {"prediction": prediction}


# @app.get("/info")
# async def get_disease_info(name: str):
#     try:
#         summary = wikipedia.summary(name, sentences=3)
#         cure_prompt = (
#         f"Explain in simple, clear, and human-readable language how to manage or cure '{name}'. "
#         "Include natural methods, chemical treatments (if any), and practical tips. "
#         "Use bullet points and sections if needed. Avoid technical jargon. Use markdown formatting."
#         )
#         cure = get_gemini_response(cure_prompt)
        
#         return {"summary": summary, "cure": cure}   
#     except:
#         pass
    


















import os
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import wikipedia
import gdown
from gemini import  get_gemini_response
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import gdown
from gemini import get_cure, get_gemini_response  # All good here

templates = Jinja2Templates(directory="templates")# load_dotenv()
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
url = "https://drive.google.com/uc?id=1Ms1HkwFo7im2Yh6V9Hn90jE_qERJl96y"
output = "plant_disease_model.h5"
gdown.download(url, output, quiet=False)
# Load model
model = tf.keras.models.load_model("plant_disease_model.h5")

class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
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
    'Tomato___healthy'
]


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((126, 126))  # use the correct size expected by your model
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 126, 126, 3)

        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class = class_names[predicted_class_index]
        confidence = float(round(100 * np.max(predictions[0]), 2))  # ✅ cast to native float
        treatment = get_cure(predicted_class)

        return JSONResponse(content={
            "prediction": predicted_class,
            "confidence": confidence,
            "cure": treatment
        })

    except Exception as e:
        return JSONResponse(content={
            "prediction": "Error",
            "confidence": 0,
            "cure": str(e)
        })

# Serve static files (your frontend)
# app.mount("/templates", StaticFiles(directory="templates"), name="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Class names




# # Endpoint: Predict
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     image_data = await file.read()
#     prediction = predict_image(image_data)
#     return {"prediction": prediction}

# # Endpoint: Info
# @app.get("/info")
# async def get_disease_info(name: str):
#     try:
#         summary = wikipedia.summary(name, sentences=3)
#         cure_prompt = (
#             f"Explain in simple, clear, and human-readable language how to manage or cure '{name}'. "
#             "Include natural methods, chemical treatments (if any), and practical tips. "
#             "Use bullet points and sections if needed. Avoid technical jargon. Use markdown formatting."
#         )
#         cure = get_gemini_response(cure_prompt)
#         return {"summary": summary, "cure": cure}
#     except Exception as e:
#         return {"summary": "No summary available.", "cure": "No cure information available."}



# Prediction function
# def predict_image(image_data):
#     image = Image.open(io.BytesIO(image_data)).resize((128, 128))
#     img_array = np.array(image) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     prediction = model.predict(img_array)
#     return class_names[np.argmax(prediction)]
