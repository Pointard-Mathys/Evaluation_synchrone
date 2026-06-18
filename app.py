import json
import logging
import os
from pathlib import Path
from typing import Literal
import hashlib

import joblib
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field


load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
MODEL_PATH = Path("artifacts/model.pkl")
CHECKSUM_PATH = Path("artifacts/model.pkl.sha256")
FEATURE_COLUMNS_PATH = Path("artifacts/feature_columns.json")
api_key_header = APIKeyHeader(name="X-API-Token", auto_error=False)



def _verify_model_integrity(model_path: Path, checksum_path: Path) -> None:
    """Vérifie l'intégrité du fichier modèle via son checksum SHA-256.

    Lève une RuntimeError si le fichier de checksum est absent ou si le
    hash calculé ne correspond pas au hash attendu.
    """
    if not checksum_path.exists():
        raise RuntimeError(
            f"Fichier de checksum introuvable : {checksum_path}. "
            "Le modèle ne peut pas être chargé sans vérification d'intégrité."
        )

    expected = checksum_path.read_text(encoding="utf-8").strip()

    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    actual = h.hexdigest()

    if actual != expected:
        raise RuntimeError(
            f"Échec de la vérification d'intégrité du modèle : "
            f"hash attendu={expected}, hash calculé={actual}. "
            "L'artefact a peut-être été altéré ou remplacé."
        )





metrics = {
    "n_predictions": 0,
    "n_errors": 0,
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn-api")

model = None
feature_columns = []
artifact_load_error = None

try:
    _verify_model_integrity(MODEL_PATH, CHECKSUM_PATH)
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
except Exception as e:
    artifact_load_error = str(e)
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
    if model is None or not feature_columns:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "model_loaded": model is not None,
                "feature_columns_loaded": bool(feature_columns),
                "reason": "Artifacts unavailable",
            },
        )

    return {
        "status": "healthy",
        "model_loaded": True,
        "feature_columns_loaded": True,
    }


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
