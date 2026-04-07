from __future__ import annotations

import re
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.keras"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"
MAX_TITLE_LENGTH = 300
MAX_BATCH_SIZE = 50


def clean_title_api(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def load_artifacts() -> tuple[tf.keras.Model, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modèle introuvable: {MODEL_PATH}")
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(f"Vectoriseur introuvable: {VECTORIZER_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


class PredictRequest(BaseModel):
    title: str = Field(..., description="Titre d'article à classifier")

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        if value is None or not value.strip():
            raise ValueError("Le titre ne peut pas être vide.")
        if len(value) > MAX_TITLE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Le titre dépasse {MAX_TITLE_LENGTH} caractères.",
            )
        return value


class BatchPredictRequest(BaseModel):
    titles: list[str] = Field(..., description="Liste de titres")

    @field_validator("titles")
    @classmethod
    def validate_titles(cls, values: list[str]) -> list[str]:
        if not values:
            raise HTTPException(status_code=400, detail="La liste de titres est vide.")
        if len(values) > MAX_BATCH_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"La liste dépasse {MAX_BATCH_SIZE} titres.",
            )
        for idx, item in enumerate(values):
            if item is None or not item.strip():
                raise HTTPException(
                    status_code=422,
                    detail=f"Le titre à l'index {idx} est vide ou invalide.",
                )
            if len(item) > MAX_TITLE_LENGTH:
                raise HTTPException(
                    status_code=400,
                    detail=f"Le titre à l'index {idx} dépasse {MAX_TITLE_LENGTH} caractères.",
                )
        return values


app = FastAPI(
    title="Fake News Detector API",
    version="1.0.0",
    description="API de classification REAL/FAKE de titres de presse",
)


@app.on_event("startup")
def startup_event() -> None:
    model, vectorizer = load_artifacts()
    app.state.model = model
    app.state.vectorizer = vectorizer


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "fake_news_detector"}


def predict_scores(titles: list[str]) -> np.ndarray:
    cleaned_titles = [clean_title_api(t) for t in titles]
    vectors = app.state.vectorizer.transform(cleaned_titles)
    probs = app.state.model.predict(vectors, verbose=0).flatten()
    return probs


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, str | float]:
    title = clean_title_api(payload.title)
    probs = predict_scores([title])
    p_real = float(probs[0])
    label = "REAL" if p_real >= 0.5 else "FAKE"
    confidence = p_real if label == "REAL" else 1 - p_real
    return {"title": title, "label": label, "confidence": round(float(confidence), 4)}


@app.post("/predict/batch")
def predict_batch(payload: BatchPredictRequest) -> dict[str, list[dict[str, str | float]]]:
    titles = [clean_title_api(t) for t in payload.titles]
    probs = predict_scores(titles)
    predictions: list[dict[str, str | float]] = []
    for title, p_real in zip(titles, probs):
        p_real_f = float(p_real)
        label = "REAL" if p_real_f >= 0.5 else "FAKE"
        confidence = p_real_f if label == "REAL" else 1 - p_real_f
        predictions.append(
            {"title": title, "label": label, "confidence": round(float(confidence), 4)}
        )
    return {"predictions": predictions}
