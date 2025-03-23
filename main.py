from fastapi import FastAPI
from pydantic import BaseModel
from fastai.text.all import *
import os
import requests
import pathlib

app = FastAPI()

# Compatibility fix for WindowsPath
pathlib.WindowsPath = pathlib.PosixPath

# Model filename and GDrive link
model_path = 'ulmfit_airline_model.pkl'
model_url = 'https://drive.google.com/uc?export=download&id=1cdZScmtkKCT7c_1I9KaGXQWLr7qDAj0n'

# Cache model instance after first load
learn = None

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "ULMFiT Airline Sentiment Classifier is running."}

@app.post("/predict/")
def predict(input: TextInput):
    global learn

    if not os.path.exists(model_path):
        print("Downloading model file from Google Drive...")
        with requests.get(model_url, stream=True) as r:
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

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