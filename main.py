from fastapi import FastAPI
from pydantic import BaseModel
from fastai.text.all import *
import os
import requests
import pathlib

app = FastAPI()

# Fix WindowsPath on Linux
pathlib.WindowsPath = pathlib.PosixPath

# Global model path and ID
model_path = 'ulmfit_airline_model.pkl'
gdrive_file_id = '1cdZScmtkKCT7c_1I9KaGXQWLr7qDAj0n'
learn = None  # Cache model after loading

# GDrive download with confirmation token support
def download_from_gdrive(file_id, dest_path):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "ULMFiT Airline Sentiment Classifier is running."}

@app.post("/predict/")
def predict(input: TextInput):
    global learn

    if not os.path.exists(model_path):
        print("Downloading model...")
        download_from_gdrive(gdrive_file_id, model_path)

    if learn is None:
        learn = load_learner(model_path)

    pred_class, pred_idx, probs = learn.predict(input.text)
    return {
        "input": input.text,
        "predicted_sentiment": pred_class,
        "confidence_scores": {
            "negative": round(float(probs[0]), 3),
            "neutral": round(float(probs[1]), 3),
            "positive": round(float(probs[2]), 3)
        }
    }