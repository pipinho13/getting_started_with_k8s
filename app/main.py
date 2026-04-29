"""FastAPI service that exposes the Iris classifier."""
import os
from pathlib import Path
from typing import List

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"

app = FastAPI(
    title="Iris Classifier",
    description="A minimal ML service used in the Kubernetes Getting Started tutorial.",
    version="1.0.0",
)


class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., example=5.1, description="Sepal length in cm")
    sepal_width: float = Field(..., example=3.5, description="Sepal width in cm")
    petal_length: float = Field(..., example=1.4, description="Petal length in cm")
    petal_width: float = Field(..., example=0.2, description="Petal width in cm")


class Prediction(BaseModel):
    predicted_class: str
    predicted_class_id: int
    probabilities: List[float]


@app.on_event("startup")
def load_model() -> None:
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found at {MODEL_PATH}. "
            "Run `python train.py` before starting the service."
        )
    bundle = joblib.load(MODEL_PATH)
    app.state.model = bundle["model"]
    app.state.target_names = bundle["target_names"]


@app.get("/")
def root() -> dict:
    return {
        "service": "iris-classifier",
        "pod": os.getenv("HOSTNAME", "unknown"),
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(features: IrisFeatures) -> Prediction:
    try:
        x = np.array(
            [[
                features.sepal_length,
                features.sepal_width,
                features.petal_length,
                features.petal_width,
            ]]
        )
        class_id = int(app.state.model.predict(x)[0])
        probs = app.state.model.predict_proba(x)[0].tolist()
        return Prediction(
            predicted_class=app.state.target_names[class_id],
            predicted_class_id=class_id,
            probabilities=probs,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
