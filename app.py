from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import io

app = FastAPI()

# Mount static folder
#app.mount("/static", StaticFiles(directory="static"), name="static")

# Load HTML templates
templates = Jinja2Templates(directory=".")

# Load trained SVM model and scaler
with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load VGG16 model for feature extraction
vgg_model = VGG16(weights="imagenet", include_top=False, input_shape=(208, 176, 3), pooling="avg")

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).resize((176, 208)).convert("RGB")
    img_array = np.expand_dims(np.array(image), axis=0)
    img_array = preprocess_input(img_array)
    features = vgg_model.predict(img_array)
    scaled_features = scaler.transform(features)
    return scaled_features

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    image_bytes = await file.read()
    processed = preprocess_image(image_bytes)
    prediction = svm_model.predict(processed)[0]
    result = "🧠 Alzheimer's Detected" if prediction == 1 else "✅ No Alzheimer's Detected"
    return templates.TemplateResponse("index.html", {"request": request, "result": result})
