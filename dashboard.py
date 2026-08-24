"""
============================================================
TerraGuard NER
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
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TerraGuard NER",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "terraguard")
DB_USER = os.getenv("DB_USER", "terraguard")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        linear-gradient(
            135deg,
            #06111f 0%,
            #0a192c 50%,
            #06101c 100%
        );
    }

    section[data-testid="stSidebar"] {
        background-color: #081321;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    div[data-testid="metric-container"] {
        background-color: #101f32;
        border: 1px solid #263c55;
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="metric-container"] label {
        color: #91a6bb !important;
    }

    div[data-testid="metric-container"] div {
        color: #ffffff;
    }

    .small-text {
        color: #91a6bb;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    """
    Create PostgreSQL connection using values from .env.
    """

    import psycopg2

    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ============================================================
# DATABASE QUERY FUNCTION
# ============================================================

def query_database(query: str) -> pd.DataFrame:
    """
    Execute SQL query and return a pandas DataFrame.

    Uses cursor directly instead of pandas' DBAPI connection
    to avoid pandas SQLAlchemy warnings.
    """

    connection = None
    cursor = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute(query)

        rows = cursor.fetchall()

        columns = [
            description[0]
            for description in cursor.description
        ]

        return pd.DataFrame(
            rows,
            columns=columns,
        )

    except Exception as error:

        st.error(
            f"Database error: {error}"
        )

        return pd.DataFrame()

    finally:

        if cursor is not None:
            cursor.close()

        if connection is not None:
            connection.close()


# ============================================================
# LOAD PREDICTIONS
# ============================================================

@st.cache_data(ttl=5)
def load_predictions() -> pd.DataFrame:

    query = """
        SELECT

            p.id AS prediction_id,

            p.observation_id,

            o.observed_at AS timestamp,

            l.id AS location_id,

            l.latitude,

            l.longitude,

            l.place_name,

            l.district,

            l.state,

            l.country,

            p.risk_score,

            p.risk_percentage,

            p.risk_level,

            p.primary_cause AS cause,
            p.explanation,

            p.model_name,

            p.model_version,

            o.rainfall_1h,

            o.rainfall_3h,

            o.rainfall_6h,

            o.rainfall_24h,

            o.rainfall_72h,

            o.slope_angle,

            o.soil_saturation,

            o.soil_moisture,

            o.vegetation_cover,

            o.ndvi,

            o.elevation,

            o.earthquake_activity,

            o.proximity_to_water,

            o.distance_to_road_m,

            o.prior_events_5y,

            o.soil_type,

            p.created_at AS prediction_created_at

        FROM predictions AS p

        INNER JOIN observations AS o
            ON p.observation_id = o.id

        INNER JOIN locations AS l
            ON o.location_id = l.id

        ORDER BY p.created_at DESC

        LIMIT 500;
    """

    return query_database(query)


# ============================================================
# LOAD ALERTS
# ============================================================

@st.cache_data(ttl=5)
def load_alerts() -> pd.DataFrame:

    query = """
        SELECT

            id,

            prediction_id,

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

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "timestamp" in df.columns:

        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            errors="coerce",
        )

    if "prediction_created_at" in df.columns:

        df["prediction_created_at"] = pd.to_datetime(
            df["prediction_created_at"],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    for column in [
        "latitude",
        "longitude",
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    df["risk_score"] = pd.to_numeric(
        df["risk_score"],
        errors="coerce",
    ).fillna(0)

    # The database stores risk_score as 0-1.
    df["risk_score"] = (
        df["risk_score"]
        .clip(0, 1)
    )

    # --------------------------------------------------------
    # Risk percentage
    # --------------------------------------------------------

    if "risk_percentage" in df.columns:

        df["risk_percentage"] = pd.to_numeric(
            df["risk_percentage"],
            errors="coerce",
        )

        df["risk_percentage"] = (
            df["risk_percentage"]
            .fillna(
                df["risk_score"] * 100
            )
        )

    else:

        df["risk_percentage"] = (
            df["risk_score"] * 100
        )

    df["risk_percentage"] = (
        df["risk_percentage"]
        .clip(0, 100)
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    df["risk_level"] = (
        df["risk_level"]
        .fillna("UNKNOWN")
        .astype(str)
        .str.upper()
    )

    # --------------------------------------------------------
    # Text columns
    # --------------------------------------------------------

    text_columns = [
        "place_name",
        "district",
        "state",
        "country",
        "cause",
        "explanation",
        "model_name",
        "model_version",
        "soil_type",
    ]

    for column in text_columns:

        if column in df.columns:

            df[column] = (
                df[column]
                .fillna("Unknown")
                .astype(str)
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

    # --------------------------------------------------------
    # Numeric fields
    # --------------------------------------------------------

    for column in [
        "latitude",
        "longitude",
        "risk_score",
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Risk score
    # --------------------------------------------------------

    if "risk_score" in df.columns:

        df["risk_score"] = (
            df["risk_score"]
            .fillna(0)
            .clip(0, 1)
        )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if "risk_level" in df.columns:

        df["risk_level"] = (
            df["risk_level"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.upper()
        )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    if "sent_at" in df.columns:

        df["sent_at"] = pd.to_datetime(
            df["sent_at"],
            errors="coerce",
        )

    return df


# ============================================================
# LOAD DATABASE DATA
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

    if not alerts.empty and "state" in alerts.columns:

        state_values = (
            alerts["state"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        states.extend(
            sorted(state_values)
        )

    if (
        not predictions.empty
        and "state" in predictions.columns
    ):

        state_values = (
            predictions["state"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        for state in sorted(state_values):

            if state not in states:

                states.append(state)

    selected_state = st.selectbox(
        "State",
        states,
    )

    # --------------------------------------------------------
    # RISK FILTER
    # --------------------------------------------------------

    risk_options = [
        "All",
        "LOW",
        "MEDIUM",
        "MODERATE",
        "WARNING",
        "HIGH",
        "CRITICAL",
    ]

    selected_risk = st.selectbox(
        "Risk Level",
        risk_options,
    )

    st.divider()

    # --------------------------------------------------------
    # REFRESH BUTTON
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
            "Running TerraGuard AI pipeline..."
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
                        "AI prediction generated."
                    )

                    if result.stdout:

                        with st.expander(
                            "Pipeline Output"
                        ):

                            st.code(
                                result.stdout
                            )

                    st.cache_data.clear()

                    st.rerun()

                else:

                    st.error(
                        "Pipeline execution failed."
                    )

                    if result.stderr:

                        st.code(
                            result.stderr
                        )

            except Exception as error:

                st.error(
                    f"Pipeline error: {error}"
                )

    st.divider()

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.subheader(
        "🟢 System Status"
    )

    st.write(
        "🟢 PostgreSQL"
    )

    st.write(
        "🟢 AI Risk Engine"
    )

    st.write(
        "🟢 Observation Pipeline"
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
        "SIH 2026"
    )


# ============================================================
# APPLY FILTERS
# ============================================================

filtered_predictions = predictions.copy()

filtered_alerts = alerts.copy()


# ------------------------------------------------------------
# Risk filter
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# State filter
# ------------------------------------------------------------

if (
    selected_state != "All"
    and not filtered_predictions.empty
):

    filtered_predictions = (
        filtered_predictions[
            filtered_predictions[
                "state"
            ]
            == selected_state
        ]
    )


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

title_column, status_column = st.columns(
    [4, 1]
)

with title_column:

    st.title(
        "🌍 TerraGuard NER"
    )

    st.caption(
        "AI-Based Early Warning & Landslide "
        "Risk Monitoring System"
    )

with status_column:

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

critical_locations = 0

monitored_locations = 0

highest_risk = 0.0


if not filtered_predictions.empty:

    critical_locations = int(
        (
            filtered_predictions[
                "risk_level"
            ]
            == "CRITICAL"
        ).sum()
    )

    monitored_locations = (
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

    highest_risk = float(
        filtered_predictions[
            "risk_percentage"
        ].max()
    )


metric1, metric2, metric3, metric4 = st.columns(
    4
)


with metric1:

    st.metric(
        "🛰️ Live Predictions",
        live_predictions,
    )


with metric2:

    st.metric(
        "🚨 Active Alerts",
        active_alerts,
    )


with metric3:

    st.metric(
        "🔴 Critical Locations",
        critical_locations,
    )


with metric4:

    st.metric(
        "📍 Monitored Locations",
        monitored_locations,
    )


# ============================================================
# CURRENT AI RISK ASSESSMENT
# ============================================================

st.subheader(
    "⚡ Current AI Risk Assessment"
)


if not filtered_predictions.empty:

    latest = (
        filtered_predictions
        .sort_values(
            "prediction_created_at",
            ascending=False,
        )
        .iloc[0]
    )

    risk = float(
        latest["risk_percentage"]
    )

    level = str(
        latest["risk_level"]
    )

    location_name = str(
        latest.get(
            "place_name",
            "Unknown",
        )
    )

    district = str(
        latest.get(
            "district",
            "Unknown",
        )
    )

    state = str(
        latest.get(
            "state",
            "Unknown",
        )
    )

    latitude = float(
        latest["latitude"]
    )

    longitude = float(
        latest["longitude"]
    )

    cause = str(
        latest.get(
            "cause",
            "Unknown",
        )
    )

    explanation = str(
        latest.get(
            "explanation",
            "No explanation available.",
        )
    )

    # --------------------------------------------------------
    # Main risk metrics
    # --------------------------------------------------------

    r1, r2, r3, r4, r5 = st.columns(
        5
    )

    with r1:

        st.metric(
            "Risk Score",
            f"{risk:.1f}%",
        )

    with r2:

        st.metric(
            "Risk Level",
            level,
        )

    with r3:

        st.metric(
            "📍 Location",
            location_name,
        )

    with r4:

        st.metric(
            "District",
            district,
        )

    with r5:

        st.metric(
            "State",
            state,
        )

    # --------------------------------------------------------
    # Progress bar
    # --------------------------------------------------------

    st.progress(
        min(
            max(
                risk / 100,
                0,
            ),
            1,
        )
    )

    # --------------------------------------------------------
    # Risk warning
    # --------------------------------------------------------

    if level == "CRITICAL":

        st.error(
            "🔴 CRITICAL RISK — Immediate "
            "disaster-management attention required."
        )

    elif level == "HIGH":

        st.warning(
            "🟠 HIGH RISK — Authorities should "
            "increase monitoring and preparedness."
        )

    elif level in [
        "WARNING",
        "MEDIUM",
        "MODERATE",
    ]:

        st.warning(
            "🟡 WARNING — Continue monitoring "
            "environmental conditions."
        )

    else:

        st.success(
            "🟢 LOW RISK — No immediate threat "
            "detected by the AI model."
        )

    # --------------------------------------------------------
    # Coordinates
    # --------------------------------------------------------

    st.markdown(
        "### 📌 Geographic Information"
    )

    g1, g2, g3, g4 = st.columns(
        4
    )

    with g1:

        st.metric(
            "Latitude",
            f"{latitude:.6f}",
        )

    with g2:

        st.metric(
            "Longitude",
            f"{longitude:.6f}",
        )

    with g3:

        st.metric(
            "District",
            district,
        )

    with g4:

        st.metric(
            "State",
            state,
        )

    # --------------------------------------------------------
    # Cause and explanation
    # --------------------------------------------------------

    st.markdown(
        "### 🔎 Why Is This Location At Risk?"
    )

    cause_column, explanation_column = st.columns(
        [1, 2]
    )

    with cause_column:

        st.warning(
            f"Primary Cause\n\n{cause}"
        )

    with explanation_column:

        st.info(
            explanation
        )

    # --------------------------------------------------------
    # Environmental conditions
    # --------------------------------------------------------

    st.markdown(
        "### 🌧️ Environmental Conditions"
    )

    e1, e2, e3, e4 = st.columns(
        4
    )

    rainfall_24h = float(
        latest.get(
            "rainfall_24h",
            0,
        )
        or 0
    )

    slope_angle = float(
        latest.get(
            "slope_angle",
            0,
        )
        or 0
    )

    soil_saturation = float(
        latest.get(
            "soil_saturation",
            0,
        )
        or 0
    )

    earthquake = float(
        latest.get(
            "earthquake_activity",
            0,
        )
        or 0
    )

    with e1:

        st.metric(
            "24h Rainfall",
            f"{rainfall_24h:.1f} mm",
        )

    with e2:

        st.metric(
            "Slope Angle",
            f"{slope_angle:.1f}°",
        )

    with e3:

        st.metric(
            "Soil Saturation",
            f"{soil_saturation:.1f}%",
        )

    with e4:

        st.metric(
            "Earthquake Activity",
            f"{earthquake:.2f}",
        )

    # --------------------------------------------------------
    # Model information
    # --------------------------------------------------------

    st.markdown(
        "### 🧠 Model Information"
    )

    m1, m2, m3 = st.columns(
        3
    )

    with m1:

        st.write(
            "**Model:**",
            latest.get(
                "model_name",
                "Unknown",
            ),
        )

    with m2:

        st.write(
            "**Version:**",
            latest.get(
                "model_version",
                "Unknown",
            ),
        )

    with m3:

        st.write(
            "**Prediction Time:**",
            latest.get(
                "timestamp",
                "Unknown",
            ),
        )

else:

    st.info(
        "No prediction is available. "
        "Click **Run AI Prediction Now** "
        "from the sidebar."
    )


# ============================================================
# LIVE REGIONAL MAP
# ============================================================

st.subheader(
    "🗺️ Live Regional Risk Map"
)


map_columns = [
    "latitude",
    "longitude",
    "risk_percentage",
    "risk_level",
    "place_name",
    "district",
    "state",
]


if not filtered_predictions.empty:

    map_df = (
        filtered_predictions[
            [
                column
                for column in map_columns
                if column in filtered_predictions.columns
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

else:

    map_df = pd.DataFrame()


if not map_df.empty:

    # --------------------------------------------------------
    # Map colors
    # --------------------------------------------------------

    def risk_color(level):

        level = str(
            level
        ).upper()

        if level == "CRITICAL":

            return [
                255,
                50,
                50,
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
                40,
                220,
            ]

        return [
            40,
            220,
            150,
            220,
        ]

    map_df["color"] = (
        map_df[
            "risk_level"
        ]
        .apply(risk_color)
    )

    map_df["radius"] = (
        map_df[
            "risk_percentage"
        ]
        .clip(0, 100)
        * 200
        + 5000
    )

    # --------------------------------------------------------
    # PyDeck layer
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Map center
    # --------------------------------------------------------

    center_latitude = float(
        map_df[
            "latitude"
        ].mean()
    )

    center_longitude = float(
        map_df[
            "longitude"
        ].mean()
    )

    view_state = pdk.ViewState(

        latitude=center_latitude,

        longitude=center_longitude,

        zoom=5.5,

        pitch=20,

    )

    # --------------------------------------------------------
    # Tooltip
    # --------------------------------------------------------

    tooltip = {

        "html":
        """
        <b>Location:</b> {place_name}<br/>
        <b>District:</b> {district}<br/>
        <b>State:</b> {state}<br/>
        <b>Risk:</b> {risk_percentage}%<br/>
        <b>Level:</b> {risk_level}<br/>
        <b>Latitude:</b> {latitude}<br/>
        <b>Longitude:</b> {longitude}
        """,

        "style": {
            "backgroundColor": "#0b1728",
            "color": "white",
        },

    }

    deck = pdk.Deck(

        layers=[
            layer
        ],

        initial_view_state=view_state,

        tooltip=tooltip,

    )

    st.pydeck_chart(
        deck,
        use_container_width=True,
    )

else:

    st.info(
        "No geographic prediction data is available."
    )


# ============================================================
# RISK ANALYTICS
# ============================================================

st.subheader(
    "📊 Risk Analytics"
)


analytics_left, analytics_right = st.columns(
    2
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

with analytics_left:

    st.markdown(
        "#### Risk Distribution"
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
            font_color="white",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    else:

        st.info(
            "No risk distribution data."
        )


# ============================================================
# RISK TREND
# ============================================================

with analytics_right:

    st.markdown(
        "#### Risk Trend"
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

        if not trend.empty:

            trend["Risk %"] = (
                trend[
                    "risk_percentage"
                ]
            )

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
                font_color="white",
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:

            st.info(
                "No timestamp data available."
            )

    else:

        st.info(
            "No prediction data."
        )


# ============================================================
# LATEST PREDICTIONS TABLE
# ============================================================

st.subheader(
    "🧠 Latest AI Predictions"
)


if not filtered_predictions.empty:

    table = (
        filtered_predictions
        .sort_values(
            "prediction_created_at",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    table["Risk"] = (
        table[
            "risk_percentage"
        ]
        .round(2)
        .astype(str)
        + "%"
    )

    display_columns = [
        "timestamp",
        "place_name",
        "district",
        "state",
        "latitude",
        "longitude",
        "Risk",
        "risk_level",
        "cause",
        "model_name",
    ]

    display_columns = [
        column
        for column in display_columns
        if column in table.columns
    ]

    table = table[
        display_columns
    ]

    rename_columns = {

        "timestamp":
            "Time",

        "place_name":
            "Location",

        "district":
            "District",

        "state":
            "State",

        "latitude":
            "Latitude",

        "longitude":
            "Longitude",

        "risk_level":
            "Risk Level",

        "cause":
            "Primary Cause",

        "model_name":
            "Model",

    }

    table = table.rename(
        columns=rename_columns
    )

    st.dataframe(
        table,
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

    alert_table = (
        filtered_alerts
        .head(20)
        .copy()
    )

    alert_table["Risk"] = (
        alert_table[
            "risk_score"
        ]
        .fillna(0)
        * 100
    ).round(2).astype(str) + "%"

    alert_columns = [

        "sent_at",

        "place_name",

        "district",

        "state",

        "latitude",

        "longitude",

        "Risk",

        "risk_level",

        "primary_cause",

        "channel",

        "delivery_status",

    ]

    alert_columns = [
        column
        for column in alert_columns
        if column in alert_table.columns
    ]

    alert_table = alert_table[
        alert_columns
    ]

    alert_rename = {

        "sent_at":
            "Sent At",

        "place_name":
            "Location",

        "district":
            "District",

        "state":
            "State",

        "latitude":
            "Latitude",

        "longitude":
            "Longitude",

        "risk_level":
            "Risk Level",

        "primary_cause":
            "Primary Cause",

        "channel":
            "Channel",

        "delivery_status":
            "Delivery",

    }

    alert_table = alert_table.rename(
        columns=alert_rename
    )

    st.dataframe(
        alert_table,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.success(
        "🟢 No alerts found for the selected filters."
    )


# ============================================================
# SELECTED LOCATION DETAILS
# ============================================================

st.subheader(
    "📍 Location Intelligence"
)


if not filtered_predictions.empty:

    location_records = (
        filtered_predictions[
            [
                "location_id",
                "place_name",
                "district",
                "state",
                "latitude",
                "longitude",
            ]
        ]
        .drop_duplicates(
            subset=[
                "location_id"
            ]
        )
        .sort_values(
            "place_name"
        )
    )

    if not location_records.empty:

        selected_location = st.selectbox(
            "Select a monitored location",
            location_records[
                "place_name"
            ].tolist(),
        )

        selected_record = (
            location_records[
                location_records[
                    "place_name"
                ]
                == selected_location
            ]
            .iloc[0]
        )

        l1, l2, l3, l4 = st.columns(
            4
        )

        with l1:

            st.metric(
                "Location",
                selected_record[
                    "place_name"
                ],
            )

        with l2:

            st.metric(
                "District",
                selected_record[
                    "district"
                ],
            )

        with l3:

            st.metric(
                "Latitude",
                f"{float(selected_record['latitude']):.6f}",
            )

        with l4:

            st.metric(
                "Longitude",
                f"{float(selected_record['longitude']):.6f}",
            )


# ============================================================
# SYSTEM PIPELINE
# ============================================================

st.subheader(
    "⚙️ TerraGuard AI Pipeline"
)


pipeline1, pipeline2, pipeline3, pipeline4, pipeline5 = st.columns(
    5
)


with pipeline1:

    st.info(
        "📡\n\n"
        "**REAL-TIME DATA**\n\n"
        "Environmental observations"
    )


with pipeline2:

    st.info(
        "🧹\n\n"
        "**VALIDATION**\n\n"
        "Data quality checks"
    )


with pipeline3:

    st.info(
        "🧠\n\n"
        "**AI MODEL**\n\n"
        "Landslide risk prediction"
    )


with pipeline4:

    st.warning(
        "🚨\n\n"
        "**ALERT ENGINE**\n\n"
        "Emergency notification"
    )


with pipeline5:

    st.success(
        "🗺️\n\n"
        "**MONITORING**\n\n"
        "Live dashboard"
    )


# ============================================================
# DATABASE INFORMATION
# ============================================================

with st.expander(
    "🔧 System Diagnostics"
):

    st.write(
        "**Database:**",
        DB_NAME,
    )

    st.write(
        "**Host:**",
        DB_HOST,
    )

    st.write(
        "**User:**",
        DB_USER,
    )

    st.write(
        "**Prediction records loaded:**",
        len(predictions),
    )

    st.write(
        "**Alert records loaded:**",
        len(alerts),
    )

    if not predictions.empty:

        st.write(
            "**Latest prediction:**"
        )

        st.json(
            predictions.iloc[0].to_dict()
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌍 TerraGuard NER • "
    "AI-Based Early Warning & Landslide Risk Monitoring "
    "System • SIH 2026"
)
