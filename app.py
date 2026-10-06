"""Small CPU API for the financial-document triage baseline."""

import math
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

from decision_models.schemas import API_MODEL, LABELS, LabelName


class Document(BaseModel):
    body: str = Field(min_length=1)


class Prediction(BaseModel):
    label: LabelName
    probabilities: dict[LabelName, float]


@asynccontextmanager
async def lifespan(app):
    app.state.bundle = joblib.load(API_MODEL)
    yield


app = FastAPI(title="Financial document triage", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model": app.state.bundle["name"]}


@app.post("/predict", response_model=Prediction)
def predict(document: Document):
    bundle = app.state.bundle
    model = bundle["model"]
    probabilities = dict(zip(model.classes_, model.predict_proba([document.body])[0].tolist()))
    # The same validation-selected decision offsets as the balanced TF-IDF benchmark.
    label = max(LABELS, key=lambda name: math.log(max(probabilities[name], 1e-12)) + bundle["offsets"][name])
    return {"label": label, "probabilities": probabilities}
