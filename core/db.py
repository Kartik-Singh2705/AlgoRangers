from __future__ import annotations
from pathlib import Path
from sqlalchemy import create_engine, text
import pandas as pd
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "terraguard.db"
engine = create_engine(f"sqlite:///{DB_PATH}", future=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    observed_at TEXT,
    latitude REAL,
    longitude REAL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    observation_id INTEGER,
    risk_score REAL NOT NULL,
    risk_level TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    explanation TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS historical_cases (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    case_text TEXT NOT NULL,
    outcome TEXT,
    latitude REAL,
    longitude REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER,
    severity TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'PENDING',
    message TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    dataset_source TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with engine.begin() as conn:
        for statement in SCHEMA.strip().split(";\n"):
            if statement.strip():
                conn.execute(text(statement))

def insert_observations(df: pd.DataFrame, source: str, lat_col=None, lon_col=None):
    import json
    init_db()
    now = datetime.utcnow().isoformat()
    rows=[]
    for _, r in df.iterrows():
        lat = float(r[lat_col]) if lat_col and pd.notna(r[lat_col]) else None
        lon = float(r[lon_col]) if lon_col and pd.notna(r[lon_col]) else None
        payload = {str(k): (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k,v in r.items()}
        rows.append({"source":source, "observed_at": None, "latitude":lat, "longitude":lon,
                     "payload_json":json.dumps(payload, default=str), "created_at":now})
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO observations
        (source, observed_at, latitude, longitude, payload_json, created_at)
        VALUES (:source,:observed_at,:latitude,:longitude,:payload_json,:created_at)"""), rows)

def insert_predictions(items: list[dict]):
    init_db()
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO predictions
        (observation_id,risk_score,risk_level,model_name,model_version,explanation,created_at)
        VALUES (:observation_id,:risk_score,:risk_level,:model_name,:model_version,:explanation,:created_at)"""), items)

def insert_cases(df: pd.DataFrame, source: str, case_texts: list[str], target=None, lat_col=None, lon_col=None):
    init_db()
    now=datetime.utcnow().isoformat()
    rows=[]
    for i, txt in enumerate(case_texts):
        r=df.iloc[i]
        rows.append({
            "source":source, "case_text":txt,
            "outcome":None if target is None or pd.isna(r[target]) else str(r[target]),
            "latitude":None if lat_col is None or pd.isna(r[lat_col]) else float(r[lat_col]),
            "longitude":None if lon_col is None or pd.isna(r[lon_col]) else float(r[lon_col]),
            "created_at":now
        })
    with engine.begin() as conn:
        conn.execute(text("""INSERT INTO historical_cases
        (source,case_text,outcome,latitude,longitude,created_at)
        VALUES (:source,:case_text,:outcome,:latitude,:longitude,:created_at)"""), rows)

def table(name: str) -> pd.DataFrame:
    init_db()
    return pd.read_sql(text(f"SELECT * FROM {name}"), engine)

def clear_dynamic():
    init_db()
    with engine.begin() as conn:
        for t in ("observations","predictions","historical_cases","alerts","model_runs"):
            conn.execute(text(f"DELETE FROM {t}"))
