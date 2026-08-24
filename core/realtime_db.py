"""
TerraGuard NER - Real-Time PostgreSQL Database Layer
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg


# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DB_HOST = os.getenv(
    "TERRAGUARD_DB_HOST",
    "localhost",
)

DB_PORT = os.getenv(
    "TERRAGUARD_DB_PORT",
    "5432",
)

DB_NAME = os.getenv(
    "TERRAGUARD_DB_NAME",
    "terraguard",
)

DB_USER = os.getenv(
    "TERRAGUARD_DB_USER",
    "terraguard",
)

DB_PASSWORD = os.getenv(
    "TERRAGUARD_DB_PASSWORD",
    "terraguard_dev",
)


def get_connection():
    """
    Create a PostgreSQL connection.
    """

    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ============================================================
# LOCATION
# ============================================================

def save_location(
    observation: dict,
) -> int:
    """
    Insert or retrieve a location.

    Returns:
        PostgreSQL location ID.
    """

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO locations (
                    latitude,
                    longitude,
                    place_name,
                    district,
                    state,
                    country
                )

                VALUES (
                    %s, %s, %s, %s, %s, %s
                )

                ON CONFLICT (
                    latitude,
                    longitude
                )

                DO UPDATE SET
                    place_name = EXCLUDED.place_name,
                    district = EXCLUDED.district,
                    state = EXCLUDED.state,
                    country = EXCLUDED.country

                RETURNING id;
                """,

                (
                    observation["latitude"],
                    observation["longitude"],
                    observation.get(
                        "place_name"
                    ),
                    observation.get(
                        "district"
                    ),
                    observation.get(
                        "state"
                    ),
                    observation.get(
                        "country",
                        "India",
                    ),
                ),
            )

            location_id = cur.fetchone()[0]

    return location_id


# ============================================================
# OBSERVATION
# ============================================================

def save_observation(
    observation: dict,
) -> int:
    """
    Save one real-time observation.

    Returns:
        Observation ID.
    """

    location_id = save_location(
        observation
    )

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO observations (

                    location_id,
                    observed_at,
                    source,

                    rainfall_1h,
                    rainfall_3h,
                    rainfall_6h,
                    rainfall_24h,
                    rainfall_72h,

                    slope_angle,
                    soil_saturation,
                    soil_moisture,

                    vegetation_cover,
                    ndvi,

                    elevation,

                    earthquake_activity,
                    proximity_to_water,

                    distance_to_road_m,
                    prior_events_5y,

                    soil_type,
                    raw_data

                )

                VALUES (

                    %s, %s, %s,

                    %s, %s, %s, %s, %s,

                    %s, %s, %s,

                    %s, %s,

                    %s,

                    %s, %s,

                    %s, %s,

                    %s,
                    %s

                )

                RETURNING id;
                """,

                (
                    location_id,

                    observation["timestamp"],

                    observation.get(
                        "source",
                        "unknown",
                    ),

                    observation.get(
                        "rainfall_1h"
                    ),

                    observation.get(
                        "rainfall_3h"
                    ),

                    observation.get(
                        "rainfall_6h"
                    ),

                    observation.get(
                        "rainfall_24h"
                    ),

                    observation.get(
                        "rainfall_72h"
                    ),

                    observation.get(
                        "slope_angle"
                    ),

                    observation.get(
                        "soil_saturation"
                    ),

                    observation.get(
                        "soil_moisture"
                    ),

                    observation.get(
                        "vegetation_cover"
                    ),

                    observation.get(
                        "ndvi"
                    ),

                    observation.get(
                        "elevation"
                    ),

                    observation.get(
                        "earthquake_activity"
                    ),

                    observation.get(
                        "proximity_to_water"
                    ),

                    observation.get(
                        "distance_to_road_m"
                    ),

                    observation.get(
                        "prior_events_5y"
                    ),

                    observation.get(
                        "soil_type"
                    ),

                    json.dumps(
                        observation
                    ),
                ),
            )

            observation_id = cur.fetchone()[0]

    return observation_id


# ============================================================
# PREDICTION
# ============================================================

def save_prediction(
    observation_id: int,
    prediction: dict,
) -> int:
    """
    Save an ML prediction associated with an observation.
    """

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                INSERT INTO predictions (

                    observation_id,

                    risk_score,
                    risk_percentage,
                    risk_level,

                    model_name,
                    model_version,

                    primary_cause,
                    explanation

                )

                VALUES (

                    %s,

                    %s,
                    %s,
                    %s,

                    %s,
                    %s,

                    %s,
                    %s

                )

                RETURNING id;
                """,

                (
                    observation_id,

                    prediction[
                        "risk_score"
                    ],

                    prediction[
                        "risk_percentage"
                    ],

                    prediction[
                        "risk_level"
                    ],

                    prediction.get(
                        "model",
                        "unknown",
                    ),

                    prediction.get(
                        "model_version"
                    ),

                    prediction.get(
                        "primary_cause"
                    ),

                    prediction.get(
                        "explanation"
                    ),
                ),
            )

            prediction_id = cur.fetchone()[0]

    return prediction_id


# ============================================================
# LATEST PREDICTIONS
# ============================================================

def get_latest_predictions(
    limit: int = 20,
) -> list[dict]:
    """
    Retrieve the latest predictions.
    """

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                """
                SELECT

                    p.id,
                    p.risk_score,
                    p.risk_percentage,
                    p.risk_level,
                    p.model_name,
                    p.created_at,

                    o.observed_at,

                    l.latitude,
                    l.longitude,
                    l.place_name,
                    l.district,
                    l.state

                FROM predictions p

                JOIN observations o
                    ON p.observation_id = o.id

                JOIN locations l
                    ON o.location_id = l.id

                ORDER BY p.created_at DESC

                LIMIT %s;
                """,
                (limit,),
            )

            columns = [
                description.name
                for description in cur.description
            ]

            rows = cur.fetchall()

    return [
        dict(
            zip(
                columns,
                row,
            )
        )
        for row in rows
    ]