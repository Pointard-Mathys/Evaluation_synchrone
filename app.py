import json
import logging
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field


load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
MODEL_PATH = Path("artifacts/model.pkl")
FEATURE_COLUMNS_PATH = Path("artifacts/feature_columns.json")
api_key_header = APIKeyHeader(name="X-API-Token", auto_error=False)

metrics = {
    "n_predictions": 0,
    "n_errors": 0,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

model = None
feature_columns = []

try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
except Exception as e:
    logger.error("Erreur au chargement des artefacts : %s", e)


class CustomerInput(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    contract: Literal["Month-to-month", "One year", "Two year"]


app = FastAPI(title="Churn Prediction API", version="1.0")


def verify_api_token(api_token: str | None = Depends(api_key_header)):
    if not API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API token is not configured",
        )
    if api_token != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API token",
        )


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/metrics", dependencies=[Depends(verify_api_token)])
def get_metrics():
    return metrics


@app.post("/predict", dependencies=[Depends(verify_api_token)])
def predict(payload: CustomerInput):
    if model is None:
        metrics["n_errors"] += 1
        raise HTTPException(status_code=500, detail="Modèle indisponible")

    try:
        df = pd.DataFrame([payload.model_dump()])
        df_encoded = pd.get_dummies(df, drop_first=True)
        df_aligned = df_encoded.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(df_aligned)[0]
        confidence = model.predict_proba(df_aligned)[0].max()

        metrics["n_predictions"] += 1
        logger.info("prediction=%s", prediction)

        return {
            "prediction": int(prediction),
            "label": "churn" if prediction == 1 else "no_churn",
            "confidence": float(confidence),
        }
    except Exception as e:
        metrics["n_errors"] += 1
        logger.error("Erreur pendant la prédiction : %s", e)
        raise HTTPException(status_code=500, detail=str(e))
