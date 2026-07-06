"""FastAPI inference service: offender profile prediction for homicide cases.

Loads the two fitted pipelines once at startup (preprocessing is inside the
pipelines, so requests carry raw feature values). Run locally:

    uvicorn shr.api:app --port 8000

then POST a case to /predict, or open /docs for the interactive UI.
"""

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from shr import __version__
from shr.data import CATEGORICAL

MODELS_DIR = Path(os.environ.get("MODELS_DIR", "models"))

try:
    _clf = joblib.load(MODELS_DIR / "sex_classifier.joblib")
    _reg = joblib.load(MODELS_DIR / "age_regressor.joblib")
except FileNotFoundError as exc:
    raise RuntimeError(
        f"model artifacts not found in {MODELS_DIR.resolve()} — run `python -m shr.train` first"
    ) from exc
_metrics_path = MODELS_DIR / "metrics.json"
_metrics = json.loads(_metrics_path.read_text()) if _metrics_path.exists() else {}

# Category vocabulary seen at training time, straight from the fitted encoder.
# Unseen values still predict (they encode to -1) but are flagged in the reply.
_encoder = _clf.named_steps["prep"].named_transformers_["cat"]
_KNOWN = {col: set(cats) for col, cats in zip(CATEGORICAL, _encoder.categories_)}

app = FastAPI(
    title="SHR Offender Profile API",
    version=__version__,
    description="Predicts offender sex and age for a homicide case, trained on "
    "FBI Supplementary Homicide Reports 1976-2022 (solved cases). "
    "See /model-info for leak-free held-out test metrics.",
)


class Case(BaseModel):
    """One homicide case, field names as in the SHR dataset."""

    VicAge: int = Field(ge=0, le=120, description="victim age in years")
    VicSex: Literal["Male", "Female"]
    VicRace: str = Field(examples=["White", "Black", "Asian"])
    Weapon: str = Field(examples=["Handgun", "Knife", "Strangulation"])
    State: str = Field(examples=["California", "Texas"])
    Agentype: str = Field(default="Municipal police", examples=["Municipal police", "Sheriff"])
    Month: str = Field(examples=["January", "July"])
    Year: int = Field(ge=1900, le=2100)
    Homicide: str = Field(default="Murder and non-negligent manslaughter")
    ActionType: str = Field(default="Normal update")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict:
    """Training provenance and held-out test metrics of the loaded models."""
    return _metrics


@app.post("/predict")
def predict(case: Case) -> dict:
    row = pd.DataFrame([case.model_dump()])
    p_female = float(_clf.predict_proba(row)[0, 1])
    age = float(_reg.predict(row)[0])
    unknown = sorted(col for col in CATEGORICAL if getattr(case, col) not in _KNOWN[col])
    result = {
        "offender_sex": "Female" if p_female >= 0.5 else "Male",
        "p_female": round(p_female, 4),
        "offender_age_years": round(age, 1),
    }
    if unknown:
        result["warnings"] = [
            f"value not seen in training data for: {', '.join(unknown)} (prediction degrades gracefully)"
        ]
    return result
