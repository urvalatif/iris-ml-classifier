from typing import List

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris


app = FastAPI(
    title="Iris ML Classifier API",
    description="A simple FastAPI app for Iris flower classification",
    version="1.0.0"
)


class IrisInput(BaseModel):
    features: List[float]


class IrisPrediction(BaseModel):
    input_features: List[float]
    predicted_class: int
    predicted_flower: str


model = joblib.load("models/iris_model.joblib")
iris = load_iris()


@app.get("/")
def home():
    return {
        "message": "Welcome to the Iris ML Classifier API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True
    }


@app.post("/predict", response_model=IrisPrediction)
def predict_flower(data: IrisInput):
    if len(data.features) != 4:
        return {
            "input_features": data.features,
            "predicted_class": -1,
            "predicted_flower": "Invalid input. Please provide exactly 4 features."
        }

    input_array = np.array([data.features], dtype=np.float32)

    prediction = model.predict(input_array)[0]
    flower_name = iris.target_names[prediction]

    return {
        "input_features": data.features,
        "predicted_class": int(prediction),
        "predicted_flower": flower_name
    }