#src/serve/api.py

"""
FastAPI Application for Bank Marketing Prediction
Production-ready REST API with monitoring, validation, and full preprocessing.

Assumptions:
- Trained model saved at: models/best_model.pkl
- Metadata saved at: models/best_model_metadata.yaml
- Preprocessing artifacts:
    - data/features/scaler.pkl
    - data/features/feature_columns.pkl
- Training pipeline:
    1) bank_raw_clean.csv
    2) FeatureEngineer().engineer_features(df)
    3) One-hot encoding (pd.get_dummies, drop_first=True)
    4) Scaling with StandardScaler
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

# Import your feature engineering logic
from src.data.feature_engineering import FeatureEngineer

# -------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bank_marketing_api")

# -------------------------------------------------------------------
# Prometheus metrics
# -------------------------------------------------------------------
PREDICTIONS_COUNTER = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["model_version", "prediction"],
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# -------------------------------------------------------------------
# Global state
# -------------------------------------------------------------------
model = None
scaler = None
feature_columns: List[str] = []
model_metadata: Dict = {}

start_time = time.time()


# -------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------
def load_model_and_artifacts() -> None:
    """
    Load trained model and preprocessing artifacts into global variables.
    """
    global model, scaler, feature_columns, model_metadata

    logger.info("Loading model and preprocessing artifacts...")

    # ---- Load model ----
    model_path = Path("models/best_model.pkl")
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run `python -m src.models.train` first."
        )
    model = joblib.load(model_path)
    logger.info(f"✅ Loaded model from {model_path}")

    # ---- Load metadata ----
    meta_path = Path("models/best_model_metadata.yaml")
    if meta_path.exists():
        with open(meta_path, "r") as f:
            model_metadata = yaml.safe_load(f) or {}
        logger.info("✅ Loaded model metadata")
    else:
        logger.warning("⚠️ Model metadata file not found. Using defaults.")
        model_metadata = {}

    # ---- Load scaler + feature columns ----
    features_dir = Path("data/features")
    scaler_path = features_dir / "scaler.pkl"
    feat_cols_path = features_dir / "feature_columns.pkl"

    if not scaler_path.exists():
        raise FileNotFoundError(
            f"{scaler_path} not found. Run `python -m src.data.preprocessing` first."
        )
    if not feat_cols_path.exists():
        raise FileNotFoundError(
            f"{feat_cols_path} not found. Run `python -m src.data.preprocessing` first."
        )

    scaler_obj = joblib.load(scaler_path)
    feat_cols = joblib.load(feat_cols_path)

    # basic sanity checks
    if not isinstance(feat_cols, list):
        raise ValueError("feature_columns.pkl must contain a list of column names.")

    global scaler, feature_columns
    scaler = scaler_obj
    feature_columns = feat_cols

    logger.info(f"✅ Loaded scaler from {scaler_path}")
    logger.info(f"✅ Loaded {len(feature_columns)} feature columns")

    logger.info("🚀 Model + artifacts loading complete.")


# -------------------------------------------------------------------
# FastAPI lifespan (startup / shutdown)
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting Bank Marketing Prediction API...")
    load_model_and_artifacts()
    yield
    logger.info("🛑 Shutting down API...")


app = FastAPI(
    title="Bank Marketing Prediction API",
    description="ML API for predicting term deposit subscriptions (UCI bank marketing dataset)",
    version="1.0.0",
    lifespan=lifespan,
)

# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus instrumentation for FastAPI
Instrumentator().instrument(app).expose(app)


# -------------------------------------------------------------------
# Pydantic models
# -------------------------------------------------------------------
class CustomerData(BaseModel):
    """
    Single customer data for prediction.
    Uses snake_case for convenience but will be mapped to original training column names.
    """

    age: int = Field(..., ge=18, le=100, description="Customer age")
    job: str = Field(..., description="Job type")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    default: str = Field(..., description="Has credit in default? (yes/no/unknown)")
    housing: str = Field(..., description="Has housing loan? (yes/no/unknown)")
    loan: str = Field(..., description="Has personal loan? (yes/no/unknown)")
    contact: str = Field(..., description="Contact communication type")
    month: str = Field(..., description="Last contact month (e.g., may, jun)")
    day_of_week: str = Field(..., description="Last contact day of week (mon–fri)")
    duration: int = Field(
        ..., ge=0, description="Last contact duration in seconds (same as training)"
    )
    campaign: int = Field(
        ..., ge=1, description="Number of contacts performed during this campaign"
    )
    pdays: int = Field(
        ...,
        ge=0,
        description="Days since last contact from previous campaign (999 = never)",
    )
    previous: int = Field(
        ..., ge=0, description="Number of contacts performed before this campaign"
    )
    poutcome: str = Field(..., description="Outcome of previous marketing campaign")
    emp_var_rate: float = Field(..., description="Employment variation rate")
    cons_price_idx: float = Field(..., description="Consumer price index")
    cons_conf_idx: float = Field(..., description="Consumer confidence index")
    euribor3m: float = Field(..., description="Euribor 3 month rate")
    nr_employed: float = Field(..., description="Number of employees")

    class Config:
        schema_extra = {
            "example": {
                "age": 35,
                "job": "management",
                "marital": "married",
                "education": "university.degree",
                "default": "no",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "duration": 261,
                "campaign": 2,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp_var_rate": 1.1,
                "cons_price_idx": 93.994,
                "cons_conf_idx": -36.4,
                "euribor3m": 4.857,
                "nr_employed": 5191.0,
            }
        }


class BatchPredictionRequest(BaseModel):
    instances: List[CustomerData] = Field(..., max_items=256)


class PredictionResponse(BaseModel):
    prediction: str
    probability: float
    confidence: str
    model_version: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str
    model_type: str
    version: str
    metrics: Dict
    trained_at: str
    feature_count: int


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------
def get_model_version() -> str:
    # Fallback-friendly "version" for metrics/logs.
    if not model_metadata:
        return "1.0.0"
    # try some likely keys, fall back to 1.0.0
    return (
        model_metadata.get("version")
        or model_metadata.get("saved_at", "")[:10]
        or "1.0.0"
    )


def get_confidence_level(prob: float) -> str:
    if prob >= 0.8:
        return "high"
    if prob >= 0.6:
        return "medium"
    return "low"


def preprocess_single_instance(data: CustomerData) -> np.ndarray:
    """
    Full preprocessing for a single incoming instance:
      1. Convert to DataFrame with original training column names.
      2. Apply FeatureEngineer.engineer_features().
      3. One-hot encode (get_dummies) and align with training feature_columns.
      4. Scale using loaded StandardScaler.
    Returns: np.ndarray shape (1, n_features)
    """
    if scaler is None or not feature_columns:
        raise RuntimeError("Preprocessing artifacts not loaded.")

    # ---- 1. Raw DataFrame in training schema ----
    raw_dict = data.dict()

    # Map snake_case to dotted names where needed
    column_mapping = {
        "emp_var_rate": "emp.var.rate",
        "cons_price_idx": "cons.price.idx",
        "cons_conf_idx": "cons.conf.idx",
        "nr_employed": "nr.employed",
    }

    # Build initial DataFrame with snake_case
    df = pd.DataFrame([raw_dict])

    # Rename to training names
    df = df.rename(columns=column_mapping)

    # Ensure columns order / presence is not critical here; engineer will handle.

    # ---- 2. Feature engineering ----
    fe = FeatureEngineer()
    df_fe = fe.engineer_features(df)

    # We don't have target 'y' here; training pipeline had it only in original dataset.
    # Just ensure it's not present by any chance.
    df_fe = df_fe.drop(columns=["y"], errors="ignore")

    # ---- 3. One-hot encoding ----
    # Use the simplest consistent approach: get_dummies on full DF,
    # then align to feature_columns (from training).
    df_encoded = pd.get_dummies(df_fe, drop_first=True)

    # Add missing columns with 0
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    # Keep only training columns, in correct order
    df_encoded = df_encoded[feature_columns]

    # ---- 4. Scaling ----
    X_scaled = scaler.transform(df_encoded.values)

    return X_scaled


# -------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "Bank Marketing Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_version": get_model_version(),
        "uptime_seconds": time.time() - start_time,
    }


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    # Graceful defaults for metadata
    m = model_metadata or {}
    return {
        "model_name": m.get("model_name", "best_model"),
        "model_type": m.get("model_type", "xgboost"),
        "version": get_model_version(),
        "metrics": {
            "best_f1": m.get("best_f1")
            or m.get("best_f1_score")
            or None
        },
        "trained_at": m.get("saved_at") or m.get("trained_at", "unknown"),
        "feature_count": len(feature_columns),
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_single(data: CustomerData, request: Request):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    start = time.time()
    try:
        X = preprocess_single_instance(data)

        proba = model.predict_proba(X)[0]
        pred_class = model.predict(X)[0]

        p_positive = float(proba[1])
        label = "yes" if int(pred_class) == 1 else "no"

        latency = time.time() - start
        PREDICTION_LATENCY.observe(latency)
        PREDICTIONS_COUNTER.labels(
            model_version=get_model_version(), prediction=label
        ).inc()

        logger.info(
            "Prediction: %s (p=%.4f, latency=%.4fs)", label, p_positive, latency
        )

        return {
            "prediction": label,
            "probability": p_positive,
            "confidence": get_confidence_level(p_positive),
            "model_version": get_model_version(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {e}",
        )


@app.post(
    "/api/v1/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
)
async def predict_batch(request: BatchPredictionRequest):
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    predictions: List[PredictionResponse] = []

    for instance in request.instances:
        try:
            X = preprocess_single_instance(instance)
            proba = model.predict_proba(X)[0]
            pred_class = model.predict(X)[0]

            p_positive = float(proba[1])
            label = "yes" if int(pred_class) == 1 else "no"

            PREDICTIONS_COUNTER.labels(
                model_version=get_model_version(), prediction=label
            ).inc()

            predictions.append(
                PredictionResponse(
                    prediction=label,
                    probability=p_positive,
                    confidence=get_confidence_level(p_positive),
                    model_version=get_model_version(),
                    timestamp=datetime.utcnow().isoformat(),
                )
            )
        except Exception as e:
            logger.error(f"Batch prediction error for instance: {e}")
            # we still return something for this row
            predictions.append(
                PredictionResponse(
                    prediction="error",
                    probability=0.0,
                    confidence="unknown",
                    model_version=get_model_version(),
                    timestamp=datetime.utcnow().isoformat(),
                )
            )

    return BatchPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """
    Prometheus metrics endpoint (custom).
    Note: Instrumentator also exposes /metrics; this endpoint returns the same.
    """
    return JSONResponse(
        content=generate_latest().decode("utf-8"),
        media_type="text/plain",
    )


# -------------------------------------------------------------------
# Error handlers
# -------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.serve.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )