"""
TerraGuard NER — Backend API
=============================
A standalone FastAPI service around the same core/ ML + RAG + DB modules
used by the Streamlit dashboard. This is what a mobile app, a future
React frontend, or SIH judges' Postman collection would call directly —
the dashboard and this API share one brain, they never duplicate logic.

Run:
    uvicorn backend.main:app --reload --port 8000

Docs (auto-generated):
    http://127.0.0.1:8000/docs
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.ml import load_artifacts, predict, risk_level, feature_importance
from core.rag import CaseRAG, grounded_explanation, build_from_dataframe
from core.data import make_case_text
from core.db import init_db, table, insert_predictions

MODEL_FILE = ROOT / "models" / "risk_models.joblib"

app = FastAPI(
    title="TerraGuard NER API",
    description="AI-based early warning & landslide risk monitoring — backend service (SIH 26001 prototype).",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """Arbitrary feature:value pairs. Any keys that aren't model features are ignored."""
    features: dict = Field(..., example={"rainfall_mm": 180.0, "slope_deg": 34.0, "soil_moisture": 0.42})
    save: bool = Field(default=False, description="If true, persist this prediction to the database.")


class PredictResponse(BaseModel):
    risk_score: float
    risk_percent: float
    risk_level: str
    model: str
    task: str
    explanation: str


@app.on_event("startup")
def _startup():
    init_db()


def _load_or_404():
    if not MODEL_FILE.exists():
        raise HTTPException(
            status_code=503,
            detail="No trained model yet. Run scripts/train_models.py (or click 'Train / Refresh' in the dashboard) first.",
        )
    return load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_FILE.exists(), "time": datetime.utcnow().isoformat()}


@app.get("/model/info")
def model_info():
    artifacts = _load_or_404()
    return {
        "task": artifacts["task"],
        "target": artifacts.get("target"),
        "features": artifacts["features"],
        "best_model": artifacts.get("best_model"),
        "class_mapping": artifacts.get("class_mapping"),
    }


@app.post("/predict", response_model=PredictResponse)
def predict_one(req: PredictRequest):
    artifacts = _load_or_404()
    features = artifacts["features"]
    row = {f: req.features.get(f) for f in features}
    missing = [f for f in features if row[f] is None]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required features: {missing}")

    df_row = pd.DataFrame([row])
    score = float(predict(df_row, artifacts)[0])
    level = risk_level(score)

    explanation = f"{level} risk ({score * 100:.1f}%). RAG index not built yet — train from the dashboard for grounded explanations."
    try:
        rag = CaseRAG.load()
        retrieved = rag.retrieve(make_case_text(df_row.iloc[0]), k=5)
        explanation = grounded_explanation(score, df_row.iloc[0], retrieved, feature_importance(artifacts))
    except Exception:
        pass

    if req.save:
        insert_predictions([{
            "observation_id": None,
            "risk_score": score,
            "risk_level": level,
            "model_name": artifacts.get("best_model") or "isolation_forest",
            "model_version": "prototype-v1",
            "explanation": explanation,
            "created_at": datetime.utcnow().isoformat(),
        }])

    return PredictResponse(
        risk_score=score,
        risk_percent=round(score * 100, 1),
        risk_level=level,
        model=artifacts.get("best_model") or "isolation_forest",
        task=artifacts["task"],
        explanation=explanation,
    )


@app.get("/stats")
def stats():
    """Row counts across the operational tables — same numbers the dashboard shows."""
    init_db()
    out = {}
    for name in ["observations", "predictions", "historical_cases", "alerts", "model_runs"]:
        out[name] = int(len(table(name)))
    return out


@app.get("/alerts")
def alerts(limit: int = 50):
    init_db()
    df = table("alerts")
    if len(df):
        df = df.sort_values("created_at", ascending=False).head(limit)
    return df.to_dict(orient="records")


@app.get("/predictions/recent")
def recent_predictions(limit: int = 50):
    init_db()
    df = table("predictions")
    if len(df):
        df = df.sort_values("created_at", ascending=False).head(limit)
    return df.to_dict(orient="records")
