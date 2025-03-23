from fastapi import FastAPI
from pydantic import BaseModel
from fastai.text.all import *
import os
import requests
import pathlib

app = FastAPI()

# Download model from Google Drive if not present
model_path = 'ulmfit_airline_model.pkl'
if not os.path.exists(model_path):
    print("Downloading model file...")
    url = 'https://drive.google.com/uc?export=download&id=1cdZScmtkKCT7c_1I9KaGXQWLr7qDAj0n'
    r = requests.get(url)
    with open(model_path, 'wb') as f:
        f.write(r.content)

# Fix for WindowsPath incompatibility on Linux
pathlib.WindowsPath = pathlib.PosixPath

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "ULMFiT Airline Sentiment Classifier is running."}

@app.post("/predict/")
def predict(input: TextInput):
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