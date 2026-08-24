
from __future__ import annotations
from core.location import resolve_location
from pathlib import Path
import re
import pandas as pd
import numpy as np

TARGET_CANDIDATES = [
    "landslide", "landslide_event", "landslide_occurrence", "landslide_occurred",
    "label", "target", "risk", "risk_level", "hazard"
]
LAT_CANDIDATES = ["latitude", "lat", "y"]
LON_CANDIDATES = ["longitude", "lon", "lng", "long", "x"]
TIME_CANDIDATES = ["timestamp", "datetime", "date", "event_date", "time"]

def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in norm:
            return norm[key]
    return None

def infer_target(df: pd.DataFrame, explicit: str | None = None) -> str | None:
    if explicit and explicit in df.columns:
        return explicit
    c = find_column(df, TARGET_CANDIDATES)
    if c is not None:
        nunique = df[c].dropna().nunique()
        if 2 <= nunique <= 10:
            return c
    # A final heuristic: small-cardinality numeric/bool column with target-like name.
    for c in df.columns:
        name = re.sub(r"[^a-z0-9]", "", c.lower())
        if any(k in name for k in ("landslide", "occurrence", "event", "target", "label")):
            if df[c].dropna().nunique() <= 10:
                return c
    return None

def numeric_features(df: pd.DataFrame, target: str | None) -> list[str]:
    cols = df.select_dtypes(include=np.number).columns.tolist()
    excluded = {target} if target else set()
    # Coordinates and obvious IDs are retained for maps/database but not trained by default.
    excluded |= {c for c in cols if re.search(r"(^id$|_id$|^index$|latitude|longitude|^lat$|^lon$|^lng$)", c, re.I)}
    return [c for c in cols if c not in excluded and df[c].nunique(dropna=True) > 1]

def map_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    return find_column(df, LAT_CANDIDATES), find_column(df, LON_CANDIDATES)

def time_column(df: pd.DataFrame) -> str | None:
    return find_column(df, TIME_CANDIDATES)

def clean_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].replace([np.inf, -np.inf], np.nan)
        X[c] = X[c].fillna(X[c].median())
    return X

def make_case_text(row: pd.Series, target: str | None = None) -> str:
    parts = []
    for k, v in row.items():
        if target and k == target:
            continue
        if pd.isna(v):
            continue
        parts.append(f"{k}={v}")
    if target and target in row.index:
        parts.append(f"outcome={row[target]}")
    return "; ".join(parts)
# ============================================================
# REAL-TIME OBSERVATION DATA CONTRACT
# ============================================================

from typing import Any


# These are the fields expected by the real-time pipeline.
REALTIME_FIELDS = [
    "timestamp",
    "latitude",
    "longitude",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_24h",
    "rainfall_72h",
    "slope_angle",
    "soil_saturation",
    "soil_moisture",
    "vegetation_cover",
    "ndvi",
    "elevation",
    "earthquake_activity",
    "proximity_to_water",
    "distance_to_road_m",
    "prior_events_5y",
    "soil_type",
    "source",
]


# Numeric fields used by the ML pipeline.
REALTIME_NUMERIC_FIELDS = [
    "latitude",
    "longitude",
    "rainfall_1h",
    "rainfall_3h",
    "rainfall_6h",
    "rainfall_24h",
    "rainfall_72h",
    "slope_angle",
    "soil_saturation",
    "soil_moisture",
    "vegetation_cover",
    "ndvi",
    "elevation",
    "earthquake_activity",
    "proximity_to_water",
    "distance_to_road_m",
    "prior_events_5y",
]


def create_realtime_observation(
    *,
    timestamp: Any,
    latitude: float,
    longitude: float,
    rainfall_1h: float = 0.0,
    rainfall_3h: float = 0.0,
    rainfall_6h: float = 0.0,
    rainfall_24h: float = 0.0,
    rainfall_72h: float = 0.0,
    slope_angle: float = 0.0,
    soil_saturation: float = 0.0,
    soil_moisture: float = 0.0,
    vegetation_cover: float = 0.0,
    ndvi: float = 0.0,
    elevation: float = 0.0,
    earthquake_activity: float = 0.0,
    proximity_to_water: float = 0.0,
    distance_to_road_m: float = 500.0,
    prior_events_5y: float = 0.0,
    soil_type: str = "unknown",
    source: str = "simulation",
) -> dict:
    """
    Create one standardized real-time observation.

    Every future data source should eventually be converted
    into this format before entering the ML pipeline.
    """

    observation = {
        "timestamp": timestamp,
        "latitude": latitude,
        "longitude": longitude,

        "rainfall_1h": rainfall_1h,
        "rainfall_3h": rainfall_3h,
        "rainfall_6h": rainfall_6h,
        "rainfall_24h": rainfall_24h,
        "rainfall_72h": rainfall_72h,

        "slope_angle": slope_angle,
        "soil_saturation": soil_saturation,
        "soil_moisture": soil_moisture,

        "vegetation_cover": vegetation_cover,
        "ndvi": ndvi,

        "elevation": elevation,

        "earthquake_activity": earthquake_activity,
        "proximity_to_water": proximity_to_water,

        "distance_to_road_m": distance_to_road_m,
        "prior_events_5y": prior_events_5y,

        "soil_type": soil_type,
        "source": source,
    }
        # Resolve human-readable location from coordinates
    location = resolve_location(
        latitude,
        longitude,
    )

    observation.update(
        {
            "place_name": location["place_name"],
            "district": location["district"],
            "state": location["state"],
            "country": location["country"],
            "location_distance_km": location["distance_km"],
        }
    )

    return observation


def validate_realtime_observation(
    observation: dict,
) -> tuple[bool, list[str]]:
    """
    Validate a real-time observation before it enters
    the ML/database pipeline.

    Returns:
        (True, []) if valid
        (False, [errors]) if invalid
    """

    errors = []

    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required_fields = [
        "timestamp",
        "latitude",
        "longitude",
    ]

    for field in required_fields:
        if field not in observation:
            errors.append(f"Missing required field: {field}")

    if errors:
        return False, errors

    # --------------------------------------------------------
    # Coordinate validation
    # --------------------------------------------------------

    try:
        latitude = float(observation["latitude"])
        longitude = float(observation["longitude"])
    except (TypeError, ValueError):
        errors.append("Latitude and longitude must be numeric.")
        return False, errors

    if not -90 <= latitude <= 90:
        errors.append(
            f"Invalid latitude: {latitude}. "
            "Latitude must be between -90 and 90."
        )

    if not -180 <= longitude <= 180:
        errors.append(
            f"Invalid longitude: {longitude}. "
            "Longitude must be between -180 and 180."
        )

    # --------------------------------------------------------
    # Numeric feature validation
    # --------------------------------------------------------

    for field in REALTIME_NUMERIC_FIELDS:
        if field not in observation:
            continue

        try:
            value = float(observation[field])

            if not np.isfinite(value):
                errors.append(f"{field} must be finite.")

        except (TypeError, ValueError):
            errors.append(
                f"{field} must be numeric. "
                f"Received: {observation[field]!r}"
            )

    # --------------------------------------------------------
    # Range checks
    # --------------------------------------------------------

    range_checks = {
        "rainfall_1h": (0, 1000),
        "rainfall_3h": (0, 2000),
        "rainfall_6h": (0, 3000),
        "rainfall_24h": (0, 5000),
        "rainfall_72h": (0, 10000),

        "slope_angle": (0, 90),

        "soil_saturation": (0, 100),
        "soil_moisture": (0, 100),

        "vegetation_cover": (0, 100),

        "ndvi": (-1, 1),

        "elevation": (-500, 10000),

        "earthquake_activity": (0, 20),

        "proximity_to_water": (0, 100000),

        "distance_to_road_m": (0, 100000),
        "prior_events_5y": (0, 100),
    }

    for field, (minimum, maximum) in range_checks.items():

        if field not in observation:
            continue

        try:
            value = float(observation[field])

            if value < minimum or value > maximum:
                errors.append(
                    f"{field}={value} is outside the "
                    f"allowed range [{minimum}, {maximum}]."
                )

        except (TypeError, ValueError):
            pass

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    return len(errors) == 0, errors


def observation_to_dataframe(observation: dict) -> pd.DataFrame:
    """
    Convert one validated observation into a one-row
    pandas DataFrame.

    This will be useful when passing real-time data
    into the existing ML code.
    """

    return pd.DataFrame([observation])


def get_realtime_feature_columns() -> list[str]:
    """
    Return the numeric fields that can be used as
    real-time model inputs.
    """

    return REALTIME_NUMERIC_FIELDS.copy()