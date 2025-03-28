from fastapi import FastAPI
from pydantic import BaseModel
from fastai.text.all import *
import pathlib

app = FastAPI()
pathlib.WindowsPath = pathlib.PosixPath  # For Linux compatibility

model_path = 'ulmfit_airline_model.pkl'
learn = load_learner(model_path)  # Load once at startup

class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "ULMFiT Airline Sentiment Classifier is running."}

@app.post("/predict/")
def predict(input: TextInput):
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