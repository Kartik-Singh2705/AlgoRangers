"""
============================================================
TERRAGUARD NER
AI-Based Early Warning & Landslide Risk Monitoring System
============================================================

Run:
    streamlit run dashboard.py
============================================================
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TerraGuard NER",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS ONLY
# No HTML components are used anywhere else.
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        linear-gradient(
            135deg,
            #07111f 0%,
            #0b1728 50%,
            #06101d 100%
        );
    }

    section[data-testid="stSidebar"] {
        background-color: #08111f;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
    }

    div[data-testid="metric-container"] {
        background-color: #111f33;
        border: 1px solid #24364d;
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="metric-container"] label {
        color: #8fa6bd !important;
    }

    div[data-testid="metric-container"] div {
        color: white;
    }

    .main-title {
        font-size: 38px;
        font-weight: 800;
    }

    .subtitle {
        color: #8fa6bd;
        font-size: 15px;
        margin-bottom: 20px;
    }

    .online {
        color: #35e69a;
        font-weight: 700;
    }

    .critical {
        color: #ff5c70;
        font-weight: 800;
    }

    .high {
        color: #ff9f43;
        font-weight: 800;
    }

    .warning {
        color: #ffd166;
        font-weight: 800;
    }

    .low {
        color: #35e69a;
        font-weight: 800;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE CONFIG
# ============================================================

DB_HOST = os.getenv(
    "DB_HOST",
    "localhost",
)

DB_PORT = os.getenv(
    "DB_PORT",
    "5432",
)

DB_NAME = os.getenv(
    "DB_NAME",
    "terraguard",
)

DB_USER = os.getenv(
    "DB_USER",
    "terraguard",
)

DB_PASSWORD = os.getenv(
    "DB_PASSWORD",
    "",
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    import psycopg2

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ============================================================
# DATABASE QUERY
# ============================================================

def query_database(
    query: str,
) -> pd.DataFrame:

    connection = None

    try:

        connection = get_connection()

        return pd.read_sql_query(
            query,
            connection,
        )

    except Exception as error:

        st.error(
            f"Database error: {error}"
        )

        return pd.DataFrame()

    finally:

        if connection:

            connection.close()


# ============================================================
# LOAD PREDICTIONS
# ============================================================

@st.cache_data(ttl=5)
def load_predictions():

    query = """
        SELECT
            id,
            timestamp,
            latitude,
            longitude,
            risk_score,
            risk_level,
            model_version
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT 500;
    """

    return query_database(query)


# ============================================================
# LOAD ALERTS
# ============================================================

@st.cache_data(ttl=5)
def load_alerts():

    query = """
        SELECT
            id,
            latitude,
            longitude,
            place_name,
            district,
            state,
            risk_score,
            risk_level,
            primary_cause,
            recipients,
            channel,
            delivery_status,
            sent_at
        FROM alerts
        ORDER BY sent_at DESC
        LIMIT 500;
    """

    return query_database(query)


# ============================================================
# PREPARE PREDICTIONS
# ============================================================

def prepare_predictions(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:

        return df

    df = df.copy()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    df["risk_score"] = pd.to_numeric(
        df["risk_score"],
        errors="coerce",
    ).fillna(0)

    # Convert percentage-style score to 0-1.
    if df["risk_score"].max() > 1:

        df["risk_score"] /= 100

    df["risk_score"] = (
        df["risk_score"]
        .clip(0, 1)
    )

    df["risk_level"] = (
        df["risk_level"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )

    return df


# ============================================================
# PREPARE ALERTS
# ============================================================

def prepare_alerts(
    df: pd.DataFrame,
) -> pd.DataFrame:

    if df.empty:

        return df

    df = df.copy()

    df["risk_score"] = pd.to_numeric(
        df["risk_score"],
        errors="coerce",
    ).fillna(0)

    if df["risk_score"].max() > 1:

        df["risk_score"] /= 100

    df["risk_score"] = (
        df["risk_score"]
        .clip(0, 1)
    )

    df["risk_level"] = (
        df["risk_level"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

predictions = prepare_predictions(
    load_predictions()
)

alerts = prepare_alerts(
    load_alerts()
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌍 TerraGuard")

    st.caption(
        "North Eastern Region"
    )

    st.divider()

    st.subheader(
        "🎛️ Risk Filters"
    )

    # --------------------------------------------------------
    # STATE FILTER
    # --------------------------------------------------------

    states = ["All"]

    if not alerts.empty:

        if "state" in alerts.columns:

            state_values = (
                alerts["state"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            states += sorted(
                state_values
            )

    selected_state = st.selectbox(
        "State",
        states,
    )

    # --------------------------------------------------------
    # RISK FILTER
    # --------------------------------------------------------

    selected_risk = st.selectbox(
        "Risk Level",
        [
            "All",
            "LOW",
            "MEDIUM",
            "WARNING",
            "HIGH",
            "CRITICAL",
        ],
    )

    st.divider()

    # --------------------------------------------------------
    # REFRESH
    # --------------------------------------------------------

    if st.button(
        "🔄 Refresh Live Data",
        use_container_width=True,
    ):

        st.cache_data.clear()

        st.rerun()

    # --------------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------------

    if st.button(
        "🧠 Run AI Prediction Now",
        use_container_width=True,
    ):

        with st.spinner(
            "Running TerraGuard AI..."
        ):

            try:

                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "realtime.pipeline",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode == 0:

                    st.success(
                        "Prediction generated."
                    )

                    st.cache_data.clear()

                    st.rerun()

                else:

                    st.error(
                        "Pipeline failed."
                    )

                    st.code(
                        result.stderr
                    )

            except Exception as error:

                st.error(
                    f"Pipeline error: {error}"
                )

    st.divider()

    st.subheader(
        "🟢 System Status"
    )

    st.write(
        "🟢 AI Risk Engine"
    )

    st.write(
        "🟢 PostgreSQL"
    )

    st.write(
        "🟢 Prediction Pipeline"
    )

    st.write(
        "🟢 Alert Engine"
    )

    st.write(
        "🟢 Telegram"
    )

    st.divider()

    st.caption(
        "TerraGuard NER"
    )

    st.caption(
        "SIH 2026 • PS 26001"
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_predictions = predictions.copy()

filtered_alerts = alerts.copy()


# ============================================================
# RISK FILTER
# ============================================================

if (
    selected_risk != "All"
    and not filtered_predictions.empty
):

    filtered_predictions = (
        filtered_predictions[
            filtered_predictions[
                "risk_level"
            ]
            == selected_risk
        ]
    )


if (
    selected_risk != "All"
    and not filtered_alerts.empty
):

    filtered_alerts = (
        filtered_alerts[
            filtered_alerts[
                "risk_level"
            ]
            == selected_risk
        ]
    )


# ============================================================
# STATE FILTER
# ============================================================

if (
    selected_state != "All"
    and not filtered_alerts.empty
):

    filtered_alerts = (
        filtered_alerts[
            filtered_alerts[
                "state"
            ]
            == selected_state
        ]
    )


# ============================================================
# HEADER
# ============================================================

title_col, status_col = st.columns(
    [4, 1]
)

with title_col:

    st.title(
        "🌍 TerraGuard NER"
    )

    st.caption(
        "AI-Based Early Warning & Landslide "
        "Risk Monitoring System"
    )

with status_col:

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    st.success(
        "● SYSTEM ONLINE"
    )


st.divider()


# ============================================================
# TOP METRICS
# ============================================================

live_predictions = len(
    filtered_predictions
)

active_alerts = len(
    filtered_alerts
)

critical_count = 0

highest_risk = 0.0

locations = 0


if not filtered_predictions.empty:

    critical_count = int(
        (
            filtered_predictions[
                "risk_level"
            ]
            == "CRITICAL"
        ).sum()
    )

    highest_risk = (
        filtered_predictions[
            "risk_score"
        ]
        .max()
        * 100
    )

    locations = (
        filtered_predictions[
            [
                "latitude",
                "longitude",
            ]
        ]
        .dropna()
        .drop_duplicates()
        .shape[0]
    )


m1, m2, m3, m4 = st.columns(4)


with m1:

    st.metric(
        "🛰️ Live Predictions",
        live_predictions,
        help="Latest ML prediction records.",
    )


with m2:

    st.metric(
        "🚨 Active Alerts",
        active_alerts,
        help="Stored alert events.",
    )


with m3:

    st.metric(
        "🔴 Critical Locations",
        critical_count,
    )


with m4:

    st.metric(
        "📍 Monitored Locations",
        locations,
    )


# ============================================================
# LATEST RISK
# ============================================================

st.subheader(
    "⚡ Current AI Risk Assessment"
)


if not filtered_predictions.empty:

    latest = (
        filtered_predictions
        .sort_values(
            "timestamp",
            ascending=False,
        )
        .iloc[0]
    )

    risk = (
        float(
            latest["risk_score"]
        )
        * 100
    )

    level = str(
        latest["risk_level"]
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Risk Score",
            f"{risk:.1f}%",
        )

    with c2:

        st.metric(
            "Risk Level",
            level,
        )

    with c3:

        st.metric(
            "Latitude",
            f"{float(latest['latitude']):.4f}",
        )

    with c4:

        st.metric(
            "Longitude",
            f"{float(latest['longitude']):.4f}",
        )

    st.progress(
        min(
            max(
                float(
                    latest["risk_score"]
                ),
                0,
            ),
            1,
        )
    )

    if level == "CRITICAL":

        st.error(
            "🔴 CRITICAL RISK — "
            "Immediate attention required."
        )

    elif level == "HIGH":

        st.warning(
            "🟠 HIGH RISK — "
            "Monitoring and preparedness recommended."
        )

    elif level in [
        "WARNING",
        "MEDIUM",
        "MODERATE",
    ]:

        st.warning(
            "🟡 WARNING — "
            "Continue monitoring conditions."
        )

    else:

        st.success(
            "🟢 LOW RISK — "
            "No immediate threat detected."
        )

else:

    st.info(
        "No prediction is available yet. "
        "Click '🧠 Run AI Prediction Now'."
    )


# ============================================================
# MAP
# ============================================================

st.subheader(
    "🗺️ Live Regional Risk Map"
)


if not filtered_predictions.empty:

    map_df = (
        filtered_predictions[
            [
                "latitude",
                "longitude",
                "risk_score",
                "risk_level",
            ]
        ]
        .dropna(
            subset=[
                "latitude",
                "longitude",
            ]
        )
        .copy()
    )

    if not map_df.empty:

        map_df["risk_percent"] = (
            map_df["risk_score"]
            * 100
        )

        # --------------------------------------------
        # Color by risk
        # --------------------------------------------

        def get_color(level):

            level = str(
                level
            ).upper()

            if level == "CRITICAL":

                return [
                    255,
                    60,
                    60,
                    220,
                ]

            if level == "HIGH":

                return [
                    255,
                    130,
                    30,
                    220,
                ]

            if level in [
                "WARNING",
                "MEDIUM",
                "MODERATE",
            ]:

                return [
                    255,
                    210,
                    50,
                    220,
                ]

            return [
                40,
                220,
                150,
                220,
            ]

        map_df["color"] = (
            map_df["risk_level"]
            .apply(get_color)
        )

        map_df["radius"] = (
            map_df["risk_score"]
            * 30000
            + 5000
        )

        layer = pdk.Layer(

            "ScatterplotLayer",

            data=map_df,

            get_position=[
                "longitude",
                "latitude",
            ],

            get_fill_color=[
                "color"
            ],

            get_radius=[
                "radius"
            ],

            pickable=True,

            auto_highlight=True,

        )

        view_state = pdk.ViewState(

            latitude=float(
                map_df[
                    "latitude"
                ].mean()
            ),

            longitude=float(
                map_df[
                    "longitude"
                ].mean()
            ),

            zoom=5.5,

            pitch=20,

        )

        deck = pdk.Deck(

            layers=[
                layer
            ],

            initial_view_state=
                view_state,

            tooltip={
                "html":
                """
                <b>Risk:</b>
                {risk_percent}%<br>
                <b>Level:</b>
                {risk_level}<br>
                <b>Latitude:</b>
                {latitude}<br>
                <b>Longitude:</b>
                {longitude}
                """,

                "style": {
                    "backgroundColor":
                        "#0b1728",
                    "color":
                        "white",
                },
            },

        )

        st.pydeck_chart(
            deck,
            use_container_width=True,
        )

    else:

        st.warning(
            "Predictions exist but coordinates are missing."
        )

else:

    st.info(
        "No prediction coordinates available."
    )


# ============================================================
# ANALYTICS
# ============================================================

analytics1, analytics2 = st.columns(
    2
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

with analytics1:

    st.subheader(
        "📊 Risk Distribution"
    )

    if not filtered_predictions.empty:

        distribution = (
            filtered_predictions[
                "risk_level"
            ]
            .value_counts()
            .rename_axis(
                "Risk Level"
            )
            .reset_index(
                name="Locations"
            )
        )

        fig = px.bar(
            distribution,
            x="Risk Level",
            y="Locations",
            text="Locations",
        )

        fig.update_layout(
            height=350,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#ffffff",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.info(
            "No risk data."
        )


# ============================================================
# RISK TREND
# ============================================================

with analytics2:

    st.subheader(
        "📈 Risk Trend"
    )

    if not filtered_predictions.empty:

        trend = (
            filtered_predictions
            .dropna(
                subset=[
                    "timestamp"
                ]
            )
            .sort_values(
                "timestamp"
            )
            .copy()
        )

        trend["Risk %"] = (
            trend[
                "risk_score"
            ]
            * 100
        )

        if not trend.empty:

            fig = px.line(
                trend,
                x="timestamp",
                y="Risk %",
                markers=True,
            )

            fig.update_layout(
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#ffffff",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No timestamp data."
            )

    else:

        st.info(
            "No prediction data."
        )


# ============================================================
# LATEST PREDICTIONS
# ============================================================

st.subheader(
    "🧠 Latest AI Predictions"
)


if not filtered_predictions.empty:

    display_df = (
        filtered_predictions
        .sort_values(
            "timestamp",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    display_df["Risk"] = (
        display_df[
            "risk_score"
        ]
        * 100
    ).round(2).astype(str) + "%"

    display_df = display_df[
        [
            "timestamp",
            "latitude",
            "longitude",
            "Risk",
            "risk_level",
            "model_version",
        ]
    ]

    display_df.columns = [
        "Time",
        "Latitude",
        "Longitude",
        "Risk",
        "Risk Level",
        "Model",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No predictions stored in PostgreSQL."
    )


# ============================================================
# ALERT CENTER
# ============================================================

st.subheader(
    "🚨 Alert Center"
)


if not filtered_alerts.empty:

    alert_df = (
        filtered_alerts
        .head(20)
        .copy()
    )

    alert_df["Risk"] = (
        alert_df[
            "risk_score"
        ]
        * 100
    ).round(2).astype(str) + "%"

    columns = [
        "place_name",
        "district",
        "state",
        "Risk",
        "risk_level",
        "primary_cause",
        "channel",
        "delivery_status",
        "sent_at",
    ]

    columns = [
        column
        for column in columns
        if column in alert_df.columns
    ]

    alert_df = alert_df[
        columns
    ]

    rename = {
        "place_name":
            "Location",

        "district":
            "District",

        "state":
            "State",

        "risk_level":
            "Risk Level",

        "primary_cause":
            "Primary Cause",

        "channel":
            "Channel",

        "delivery_status":
            "Delivery",

        "sent_at":
            "Sent At",
    }

    alert_df = alert_df.rename(
        columns=rename
    )

    st.dataframe(
        alert_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.success(
        "🟢 No alerts found for the selected filters."
    )


# ============================================================
# PIPELINE STATUS
# ============================================================

st.subheader(
    "⚙️ TerraGuard Pipeline"
)


p1, p2, p3, p4, p5 = st.columns(5)


with p1:

    st.info(
        "📡\n\n"
        "**DATA**\n\n"
        "Incoming observations"
    )


with p2:

    st.info(
        "🧹\n\n"
        "**VALIDATION**\n\n"
        "Feature validation"
    )


with p3:

    st.info(
        "🧠\n\n"
        "**AI MODEL**\n\n"
        "Risk prediction"
    )


with p4:

    st.info(
        "🚨\n\n"
        "**ALERT ENGINE**\n\n"
        "Emergency notification"
    )


with p5:

    st.success(
        "🗺️\n\n"
        "**DASHBOARD**\n\n"
        "Live monitoring"
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

with st.expander(
    "🔧 System Diagnostics"
):

    st.write(
        "Database:",
        DB_NAME,
    )

    st.write(
        "Host:",
        DB_HOST,
    )

    st.write(
        "User:",
        DB_USER,
    )

    st.write(
        "Prediction records:",
        len(predictions),
    )

    st.write(
        "Alert records:",
        len(alerts),
    )

    if not predictions.empty:

        st.write(
            "Latest prediction:",
            predictions.iloc[0].to_dict(),
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌍 TerraGuard NER • "
    "AI-Based Early Warning & Landslide Risk Monitoring "
    "• SIH 2026"
)