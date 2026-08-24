# TerraGuard NER — Prototype

Prototype for SIH 26001: AI-Based Early Warning & Landslide Risk Monitoring System for the North Eastern Region.

## Architecture at a glance

```
Kaggle CSV / demo CSV
        │
        ▼
  core/data.py  (adapter: infers target, lat/lon, numeric features)
        │
        ├──► core/db.py    SQLite (observations, predictions, historical_cases, alerts, model_runs)
        ├──► core/ml.py    Random Forest + HistGradientBoosting (or Isolation Forest if unsupervised)
        └──► core/rag.py   TF-IDF case retrieval + grounded explanation
        │
        ├──► app.py            Streamlit dashboard (GIS map, KPIs, predictor, RAG explorer)
        └──► backend/main.py   FastAPI service (/predict, /model/info, /stats, /alerts)
```

Both the dashboard and the API import the **same** `core/` modules — there is one ML/RAG brain, not two implementations to keep in sync.

## Important prototype rule

**The model should be trained on the Kaggle dataset your team selected, placed in `data/raw/`.**
No synthetic data is used for your actual submission model. A synthetic CSV generator
(`scripts/generate_demo_data.py`) is included purely so you can rehearse the full demo
flow before your real dataset is ready — see "Quick demo" below.

The application is deliberately built with a **data-source adapter**:
- Today: Kaggle CSV -> SQLite -> ML -> RAG -> Streamlit dashboard / FastAPI
- Later: IMD/GSI/ISRO/other government connectors -> same normalized tables -> same ML/RAG/dashboard

This keeps the prototype replaceable without rewriting the whole system.

## Quick demo (no Kaggle account needed)

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/generate_demo_data.py
python scripts/train_models.py --data data/raw/demo_synthetic_ner.csv --target landslide

streamlit run app.py
```

This rehearses the full flow end-to-end (train → dashboard → map → predictor → RAG) with clearly-labeled synthetic data. Swap in your real Kaggle CSV for the actual submission.

## Recommended Kaggle dataset

For a first prototype, a Kaggle landslide dataset with environmental predictors is appropriate. One public example is the "Landslide dataset" by Raju Mavinmar, which contains rainfall, slope, soil saturation, vegetation cover, earthquake activity, proximity to water and a binary `Landslide` target.

Another richer option is Kaggle's "Wireless Sensor Network Landslide Dataset", which includes rainfall windows, slope, soil saturation, NDVI, elevation, soil composition, pore-water pressure, strain and acoustic/seismic indicators.

**Do not mix datasets blindly.** Put the exact Kaggle CSV your team selected into `data/raw/` and let the app inspect its columns before training.

## Run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

# Put your Kaggle CSV in data/raw/
python scripts/train_models.py --data data/raw/YOUR_KAGGLE_FILE.csv

streamlit run app.py
```

If the CSV has a binary target, the trainer builds:
1. Random Forest
2. HistGradientBoosting

If no binary target is detected, the trainer falls back to Isolation Forest for anomaly/risk scoring and clearly labels the result as **unsupervised**, not a landslide-event probability.

## Database

The prototype uses SQLite for zero-setup presentation deployment. The schema mirrors the planned PostgreSQL/PostGIS design:
- observations
- predictions
- historical_cases
- alerts
- model_runs

When you move to Supabase/PostgreSQL/PostGIS, the application layer can keep the same logical tables and API contracts.

## Backend API

A standalone FastAPI service exposes the same ML/RAG core over HTTP, so a mobile app, a future
React frontend, or the judging panel's Postman collection can call it directly instead of going
through the Streamlit UI:

```bash
uvicorn backend.main:app --reload --port 8000
```

Interactive docs: `http://127.0.0.1:8000/docs`

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Liveness + whether a trained model exists |
| `/model/info` | GET | Active model, target, feature list |
| `/predict` | POST | Score one location: `{"features": {...}, "save": true}` |
| `/stats` | GET | Row counts across all database tables |
| `/alerts` | GET | Recent alerts |
| `/predictions/recent` | GET | Recent predictions |

Train a model first (via `scripts/train_models.py` or the dashboard's "Train / Refresh" button) — `/predict` returns `503` until a model exists.

## RAG

The RAG module:
1. turns historical rows into case documents,
2. builds an embedding index (FAISS when installed; TF-IDF fallback),
3. retrieves the most similar historical cases for a current prediction,
4. produces a grounded explanation using retrieved evidence + model feature importance.

This avoids inventing explanations from unrelated information. An LLM generator can be added later behind the same `generate_explanation()` interface.

## Presentation story

**Kaggle data → ingestion → database → ML risk score → RAG evidence → GIS/analytics dashboard → future government APIs**

The dashboard labels Kaggle data as **Prototype / Historical Dataset** so the team does not accidentally claim it is live government data.

### Windows import error fix

Run the training command from the `TerraGuard_NER_Prototype` project root:

```powershell
cd C:\path\to\TerraGuard_NER_Prototype
python scripts\train_models.py --data data\raw\YOUR_KAGGLE_FILE.csv
```

The current version also adds the project root to Python's import path automatically, so `ModuleNotFoundError: No module named 'core'` is fixed even when the script is launched directly.
