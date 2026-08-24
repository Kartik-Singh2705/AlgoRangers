from __future__ import annotations
from pathlib import Path
import json, glob
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from core.data import load_csv, infer_target, numeric_features, map_columns, make_case_text
from core.ml import train_models, predict, risk_level, feature_importance, load_artifacts
from core.rag import CaseRAG, grounded_explanation, build_from_dataframe
from core.db import init_db, table

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data/raw"
MODEL_FILE = ROOT / "models/risk_models.joblib"

st.set_page_config(page_title="TerraGuard NER", page_icon="⛰️", layout="wide", initial_sidebar_state="expanded")
init_db()

# ----------------------------------------------------------------------------
# Visual identity
#
# Palette is drawn from the NER landscape itself: basalt/slate for high-risk
# terrain, ridge-green for the forested hills, warm stone for card surfaces.
# The hero banner uses a topographic contour-line motif -- the same visual
# language a slope-stability map uses -- rather than a generic gradient.
# ----------------------------------------------------------------------------
RISK_COLORS = {"LOW": "#3F8F5C", "MODERATE": "#D9A23B", "HIGH": "#DB7B3F", "CRITICAL": "#C1443C"}

CONTOUR_SVG = """data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='600' height='200' viewBox='0 0 600 200'><g fill='none' stroke='%23ffffff' stroke-opacity='0.08' stroke-width='2'><path d='M-20,40 C 80,10 160,70 260,40 S 440,10 640,50'/><path d='M-20,80 C 80,50 160,110 260,80 S 440,50 640,90'/><path d='M-20,120 C 80,90 160,150 260,120 S 440,90 640,130'/><path d='M-20,160 C 80,130 160,190 260,160 S 440,130 640,170'/></g></svg>"""

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1rem; max-width: 1500px; }}
h1, h2, h3, .hero-title {{ font-family: 'Space Grotesk', sans-serif; }}

/* Hero banner */
.tg-hero {{
    background: linear-gradient(120deg, #16211F 0%, #223B33 55%, #2F6F5E 100%), url("{CONTOUR_SVG}");
    background-blend-mode: overlay;
    border-radius: 18px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.3rem;
    color: #F4F1EA;
    box-shadow: 0 8px 24px rgba(20,30,28,0.18);
}}
.tg-hero-title {{ font-size: 1.9rem; font-weight: 700; margin: 0; letter-spacing: -0.01em; }}
.tg-hero-sub {{ font-size: 0.95rem; color: #CFE3D8; margin-top: 0.35rem; }}
.tg-badge {{
    display:inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.03em; background: rgba(255,255,255,0.14); color: #EAF3EE;
    margin-right: 6px; text-transform: uppercase;
}}

/* KPI cards */
.tg-kpi {{
    background: #FBFAF6; border: 1px solid #E7E2D6; border-radius: 14px;
    padding: 0.9rem 1.1rem; border-left: 5px solid #2F6F5E;
}}
.tg-kpi .label {{ font-size: 0.78rem; color: #6B6656; text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }}
.tg-kpi .value {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 700; color: #1E2624; margin-top: 2px; }}

/* Risk pills */
.risk-pill {{
    display:inline-flex; align-items:center; gap:6px; padding: 5px 12px; border-radius: 999px;
    font-weight: 700; font-size: 0.85rem; color: white;
}}
.risk-pill .dot {{ width:8px; height:8px; border-radius:50%; background: rgba(255,255,255,0.85); }}

/* Section caption */
.tg-flow {{
    background: #F1EEE4; border-radius: 12px; padding: 0.7rem 1rem; font-size: 0.92rem;
    color: #3A4340; border: 1px dashed #C9C2AD;
}}

.stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


def risk_pill(level: str, percent: float | None = None) -> str:
    color = RISK_COLORS.get(level, "#888")
    label = f"{level}" + (f" · {percent:.1f}%" if percent is not None else "")
    return f'<span class="risk-pill" style="background:{color};"><span class="dot"></span>{label}</span>'


def kpi_card(label: str, value: str) -> str:
    return f'<div class="tg-kpi"><div class="label">{label}</div><div class="value">{value}</div></div>'


st.markdown(f"""
<div class="tg-hero">
  <div class="tg-badge">SIH 26001</div>
  <div class="tg-badge">MDoNER · North Eastern Region</div>
  <p class="tg-hero-title">⛰️ TerraGuard NER</p>
  <p class="tg-hero-sub">AI-based early warning &amp; real-time landslide risk monitoring — prototype dashboard</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Sidebar — data source & training controls
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🛰️ Control panel")
    st.caption("Data source: Kaggle / historical prototype dataset")
    st.divider()

    csvs = sorted(glob.glob(str(RAW / "*.csv")))
    uploaded = st.file_uploader("Upload Kaggle CSV", type=["csv"])
    if uploaded is not None:
        dest = RAW / "kaggle_uploaded.csv"
        dest.write_bytes(uploaded.getvalue())
        st.success("Saved to data/raw/.")
        csvs = sorted(glob.glob(str(RAW / "*.csv")))

    if not csvs:
        st.warning(
            "No CSV found yet.\n\n"
            "• Run `python scripts/generate_demo_data.py` for a quick synthetic demo, or\n"
            "• Add your selected Kaggle CSV to `data/raw/`."
        )
        st.stop()

    selected = st.selectbox("Dataset", csvs, format_func=lambda x: Path(x).name)
    df = load_csv(selected)
    target_guess = infer_target(df)
    target = st.selectbox(
        "Target column", ["AUTO"] + list(df.columns),
        index=(["AUTO"] + list(df.columns)).index(target_guess) if target_guess in df.columns else 0,
    )
    target = None if target == "AUTO" else target

    if st.button("🔁 Train / Refresh ML + RAG", type="primary", use_container_width=True):
        try:
            with st.spinner("Training models and rebuilding RAG index..."):
                features = numeric_features(df, target)
                artifacts, metrics = train_models(df, features, target)
                lat, lon = map_columns(df)
                texts = [make_case_text(r, target) for _, r in df.iterrows()]
                build_from_dataframe(df, texts, target, lat, lon)
            st.success("Models and RAG index rebuilt.")
            st.rerun()
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.caption("Backend API (optional): `uvicorn backend.main:app --reload` → docs at `/docs`")

if not MODEL_FILE.exists():
    st.info("👈 Select a dataset and click **Train / Refresh ML + RAG** in the sidebar to get started.")
    st.stop()

artifacts = load_artifacts()
features = artifacts["features"]
scores = predict(df, artifacts)
view = df.copy()
view["risk_score"] = scores
view["risk_percent"] = (scores * 100).round(1)
view["risk_level"] = [risk_level(float(x)) for x in scores]

# ----------------------------------------------------------------------------
# KPI row
# ----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(kpi_card("Records monitored", f"{len(view):,}"), unsafe_allow_html=True)
with k2:
    st.markdown(kpi_card("Critical zones", f"{int((view.risk_level == 'CRITICAL').sum()):,}"), unsafe_allow_html=True)
with k3:
    st.markdown(kpi_card("High-risk zones", f"{int((view.risk_level == 'HIGH').sum()):,}"), unsafe_allow_html=True)
with k4:
    model_label = "Anomaly (unsupervised)" if artifacts["task"] != "classification" else artifacts["best_model"].replace("_", " ").title()
    st.markdown(kpi_card("Active model", model_label), unsafe_allow_html=True)

st.write("")
tabs = st.tabs(["📊 Overview", "🗺️ Risk Map", "🎯 Predict Risk", "🧠 RAG Explanation", "📁 Data & Model", "🗄️ Database"])

# ----------------------------------------------------------------------------
with tabs[0]:
    st.markdown(
        '<div class="tg-flow">Kaggle data → ingestion → database → ML risk score → RAG evidence → GIS/analytics dashboard → future government APIs</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if artifacts["task"] == "unsupervised_anomaly":
        st.warning("No binary landslide target was detected. Scores are anomaly/risk scores, not calibrated landslide probabilities.")

    counts = view["risk_level"].value_counts().reindex(["LOW", "MODERATE", "HIGH", "CRITICAL"], fill_value=0)
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values,
        marker_color=[RISK_COLORS[l] for l in counts.index],
        text=counts.values, textposition="outside",
    ))
    fig.update_layout(title="Risk distribution across monitored locations", xaxis_title="Risk level",
                       yaxis_title="Locations", height=380, margin=dict(t=50, b=10),
                       plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Highest-risk observations")
    cols = [c for c in view.columns if c in features or c in ("risk_percent", "risk_level")]
    top = view.sort_values("risk_score", ascending=False)[cols].head(20).reset_index(drop=True)
    st.dataframe(
        top, use_container_width=True,
        column_config={
            "risk_percent": st.column_config.ProgressColumn("Risk %", min_value=0, max_value=100, format="%.1f%%"),
            "risk_level": st.column_config.TextColumn("Level"),
        },
    )

# ----------------------------------------------------------------------------
with tabs[1]:
    st.subheader("GIS risk dashboard")
    lat, lon = map_columns(view)
    if lat and lon:
        m = view.dropna(subset=[lat, lon]).copy()
        m["latitude"] = pd.to_numeric(m[lat], errors="coerce")
        m["longitude"] = pd.to_numeric(m[lon], errors="coerce")
        m = m.dropna(subset=["latitude", "longitude"])
        if len(m):
            fig = px.scatter_map(
                m, lat="latitude", lon="longitude", color="risk_percent",
                size="risk_percent", hover_name="risk_level",
                hover_data={c: True for c in features[:6]},
                zoom=5, height=650, color_continuous_scale=["#3F8F5C", "#D9A23B", "#DB7B3F", "#C1443C"])
            fig.update_layout(map_style="open-street-map", margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Latitude/longitude columns were detected but contain no usable coordinates.")
    else:
        st.info("This dataset has no latitude/longitude fields. The GIS layer stays ready — once government/GIS data is connected, points plot without any change to the ML layer.")

    st.subheader("All monitored locations")
    st.dataframe(view.sort_values("risk_score", ascending=False).head(100), use_container_width=True)

# ----------------------------------------------------------------------------
with tabs[2]:
    st.subheader("Single-location risk prediction")
    st.caption("Enter environmental readings for one point to get an on-demand risk score — this mirrors what the `/predict` API endpoint returns.")
    values = {}
    cols = st.columns(3)
    for i, c in enumerate(features):
        with cols[i % 3]:
            s = pd.to_numeric(df[c], errors="coerce")
            default = float(s.median()) if s.notna().any() else 0.0
            lo = float(s.min()) if s.notna().any() else 0.0
            hi = float(s.max()) if s.notna().any() else max(1.0, default + 1.0)
            if lo == hi:
                hi = lo + 1.0
            values[c] = st.number_input(c, value=default, min_value=lo, max_value=hi)
    row = pd.DataFrame([values])

    if st.button("Predict this location", type="primary"):
        p = float(predict(row, artifacts)[0])
        level = risk_level(p)
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(risk_pill(level, p * 100), unsafe_allow_html=True)
            st.write("")
            if level == "CRITICAL":
                st.error("Prototype threshold — operational validation required before real-world use.")
            elif level == "HIGH":
                st.warning("Prioritize field inspection.")
            elif level == "MODERATE":
                st.warning("Continue monitoring.")
            else:
                st.success("Routine monitoring.")
        try:
            rag = CaseRAG.load()
            retrieved = rag.retrieve(make_case_text(row.iloc[0]), k=5)
            explanation = grounded_explanation(p, row.iloc[0], retrieved, feature_importance(artifacts))
            with c2:
                st.markdown(explanation)
        except Exception as e:
            with c2:
                st.info(f"RAG index unavailable: {e}")

# ----------------------------------------------------------------------------
with tabs[3]:
    st.subheader("RAG explanation engine")
    st.caption("Retrieval is grounded in rows from the selected dataset — nothing is invented outside retrieved evidence + model feature importance.")
    query = st.text_area("Ask about a risk pattern", value="Why is this area at higher landslide risk?")
    if st.button("Retrieve similar cases"):
        try:
            rag = CaseRAG.load()
            results = rag.retrieve(query, k=8)
            if not results:
                st.info("No cases retrieved.")
            for i, r in enumerate(results, 1):
                st.markdown(f"**Case {i}** · similarity `{r['score']:.2f}`")
                st.write(r["text"][:800])
                st.divider()
        except Exception as e:
            st.error(f"RAG index unavailable: {e}")

# ----------------------------------------------------------------------------
with tabs[4]:
    st.subheader("Dataset & model transparency")
    st.write({
        "source": "Kaggle / historical prototype", "rows": len(df), "columns": len(df.columns),
        "target": artifacts.get("target"), "task": artifacts["task"], "features": features,
    })
    st.dataframe(df.head(100), use_container_width=True)
    imp = feature_importance(artifacts)
    if len(imp):
        fig = px.bar(imp.head(15).sort_values("importance"), x="importance", y="feature", orientation="h",
                     title="Top model feature importance", color_discrete_sequence=["#2F6F5E"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
    metrics_path = ROOT / "models/metrics.json"
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text()))
    st.markdown("**Prototype disclaimer:** this data source is not live government data, and model metrics do not establish operational safety.")

# ----------------------------------------------------------------------------
with tabs[5]:
    st.subheader("Central database")
    st.caption("SQLite for the presentation prototype; table structure migrates to PostgreSQL/PostGIS without changing the app logic.")
    for name in ["observations", "predictions", "historical_cases", "alerts", "model_runs"]:
        t = table(name)
        st.markdown(f"**{name}** — {len(t):,} rows")
        if len(t):
            st.dataframe(t.head(20), use_container_width=True)

st.divider()
st.caption("Future source adapter: IMD rainfall + GSI landslide records + ISRO/Bhuvan terrain/satellite + field/sensor feeds → same normalized database → same ML/RAG/dashboard.")
