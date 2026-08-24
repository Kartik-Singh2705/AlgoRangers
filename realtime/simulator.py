"""
TerraGuard NER - Real-Time Data Simulator

This module simulates continuous environmental observations
that would eventually come from real sensors/APIs.

The simulator does NOT make predictions.
It only generates observations.
"""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone, timedelta

from core.data import (
    create_realtime_observation,
    validate_realtime_observation,
)


# ============================================================
# SIMULATION LOCATIONS
# ============================================================

SIMULATION_LOCATIONS = [
    {
        "name": "Dhemaji",
        "latitude": 27.4728,
        "longitude": 94.9120,
        "base_slope": 37.4,
        "base_elevation": 180.0,
    },
    {
        "name": "Guwahati",
        "latitude": 26.1445,
        "longitude": 91.7362,
        "base_slope": 28.5,
        "base_elevation": 55.0,
    },
    {
        "name": "Itanagar",
        "latitude": 27.0844,
        "longitude": 93.6053,
        "base_slope": 42.0,
        "base_elevation": 320.0,
    },
    {
        "name": "Gangtok",
        "latitude": 27.3389,
        "longitude": 88.6065,
        "base_slope": 48.0,
        "base_elevation": 1650.0,
    },
    {
        "name": "Shillong",
        "latitude": 25.5788,
        "longitude": 91.8933,
        "base_slope": 35.0,
        "base_elevation": 1496.0,
    },
]


# ============================================================
# SIMULATION STATE
# ============================================================

_simulation_step = 0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    """Keep a value inside a specified range."""

    return max(minimum, min(value, maximum))


def get_current_timestamp() -> str:
    """Return current timestamp in ISO format."""

    return datetime.now(timezone.utc).isoformat()


def choose_location() -> dict:
    """Choose one monitored location."""

    return random.choice(SIMULATION_LOCATIONS)


# ============================================================
# GENERATE ONE OBSERVATION
# ============================================================

def generate_observation(
    location: dict | None = None,
) -> dict:
    """
    Generate one simulated real-time environmental observation.

    The values are intentionally varied so that the dashboard
    can demonstrate changing environmental conditions.
    """

    global _simulation_step

    _simulation_step += 1

    if location is None:
        location = choose_location()

    # --------------------------------------------------------
    # Rainfall
    # --------------------------------------------------------
    #
    # Every ~20 observations we simulate a stronger rainfall
    # period so that the system can eventually demonstrate
    # increasing landslide risk.
    # --------------------------------------------------------

    storm_cycle = (_simulation_step % 20) / 20

    if storm_cycle > 0.65:

        rainfall_multiplier = 1.5 + (
            (storm_cycle - 0.65) * 3
        )

    else:

        rainfall_multiplier = 0.7 + random.random() * 0.5

    rainfall_1h = (
        random.uniform(5, 35)
        * rainfall_multiplier
    )

    rainfall_3h = (
        rainfall_1h * random.uniform(1.8, 2.8)
    )

    rainfall_6h = (
        rainfall_3h * random.uniform(1.5, 2.2)
    )

    rainfall_24h = (
        rainfall_6h * random.uniform(2.0, 4.0)
    )

    rainfall_72h = (
        rainfall_24h * random.uniform(2.0, 3.5)
    )

    # --------------------------------------------------------
    # Soil conditions
    # --------------------------------------------------------

    base_saturation = 45 + (
        rainfall_24h / 20
    )

    soil_saturation = (
        base_saturation
        + random.uniform(-5, 5)
    )

    soil_saturation = clamp(
        soil_saturation,
        10,
        98,
    )

    soil_moisture = (
        soil_saturation
        + random.uniform(-5, 5)
    )

    soil_moisture = clamp(
        soil_moisture,
        5,
        98,
    )

    # --------------------------------------------------------
    # Terrain
    # --------------------------------------------------------

    slope_angle = (
        location["base_slope"]
        + random.uniform(-3, 3)
    )

    elevation = (
        location["base_elevation"]
        + random.uniform(-20, 20)
    )

    # --------------------------------------------------------
    # Vegetation
    # --------------------------------------------------------

    vegetation_cover = random.uniform(
        20,
        85,
    )

    ndvi = (
        vegetation_cover / 100
        + random.uniform(-0.08, 0.08)
    )

    ndvi = clamp(
        ndvi,
        -1,
        1,
    )

    # --------------------------------------------------------
    # Earthquake activity
    # --------------------------------------------------------

    earthquake_activity = random.uniform(
        0,
        2,
    )

    # Occasionally simulate seismic activity.
    if random.random() < 0.05:
        earthquake_activity = random.uniform(
            3,
            5,
        )

    # --------------------------------------------------------
    # Distance from water
    # --------------------------------------------------------

    proximity_to_water = random.uniform(
        50,
        5000,
    )
    # --------------------------------------------------------
# Historical / infrastructure features
# --------------------------------------------------------

    distance_to_road_m = random.uniform(
    50,
    3000,
)

    prior_events_5y = random.randint(
    0,
    5,
)   
    

    # --------------------------------------------------------
    # Create standardized observation
    # --------------------------------------------------------

    observation = create_realtime_observation(
        timestamp=get_current_timestamp(),

        latitude=location["latitude"],
        longitude=location["longitude"],

        rainfall_1h=round(rainfall_1h, 2),
        rainfall_3h=round(rainfall_3h, 2),
        rainfall_6h=round(rainfall_6h, 2),
        rainfall_24h=round(rainfall_24h, 2),
        rainfall_72h=round(rainfall_72h, 2),

        slope_angle=round(
            slope_angle,
            2,
        ),

        soil_saturation=round(
            soil_saturation,
            2,
        ),

        soil_moisture=round(
            soil_moisture,
            2,
        ),

        vegetation_cover=round(
            vegetation_cover,
            2,
        ),

        ndvi=round(
            ndvi,
            3,
        ),

        elevation=round(
            elevation,
            2,
        ),

        earthquake_activity=round(
            earthquake_activity,
            2,
        ),

        proximity_to_water=round(
            proximity_to_water,
            2,
        ),

        distance_to_road_m=round(
            distance_to_road_m,
            2,
        ),

        prior_events_5y=round(
            prior_events_5y,
            2,
        ),

        soil_type="mixed",

        source="simulation",
    )

    return observation


# ============================================================
# VALIDATED OBSERVATION
# ============================================================

def generate_validated_observation(
    location: dict | None = None,
) -> dict:
    """
    Generate and validate one observation.

    Raises ValueError if generated data is invalid.
    """

    observation = generate_observation(
        location=location
    )

    valid, errors = validate_realtime_observation(
        observation
    )

    if not valid:

        raise ValueError(
            "Generated observation failed validation: "
            + "; ".join(errors)
        )

    return observation


# ============================================================
# CONTINUOUS STREAM
# ============================================================

def stream_observations(
    interval_seconds: int = 10,
    location: dict | None = None,
):
    """
    Continuously generate observations.

    This function will later be replaced by actual
    real-time data ingestion.
    """

    print("=" * 60)
    print("TerraGuard NER - Real-Time Data Simulator")
    print("=" * 60)

    print(
        f"Generating a new observation every "
        f"{interval_seconds} seconds."
    )

    print("Press Ctrl+C to stop.")
    print("=" * 60)

    try:

        while True:

            observation = generate_validated_observation(
                location=location
            )

            yield observation

            time.sleep(interval_seconds)

    except KeyboardInterrupt:

        print("\nSimulation stopped.")


# ============================================================
# TERMINAL DEMO
# ============================================================

def print_observation(
    observation: dict,
) -> None:
    """Print an observation in a human-readable format."""

    print("\n" + "-" * 60)

    print(
        f"Time       : {observation['timestamp']}"
    )

    print(
        f"Location   : "
        f"{observation.get('place_name', 'Unknown')}, "
        f"{observation.get('state', 'Unknown')}"
    )

    print(
        f"Coordinates: "
        f"{observation['latitude']:.4f}, "
        f"{observation['longitude']:.4f}"
    )

    print(
        f"Rainfall   : "
        f"{observation['rainfall_24h']:.2f} mm / 24h"
    )

    print(
        f"Saturation : "
        f"{observation['soil_saturation']:.2f}%"
    )

    print(
        f"Slope      : "
        f"{observation['slope_angle']:.2f}°"
    )

    print(
        f"Vegetation : "
        f"{observation['vegetation_cover']:.2f}%"
    )

    print(
        f"Earthquake : "
        f"{observation['earthquake_activity']:.2f}"
    )

    print(
        f"Source     : "
        f"{observation['source']}"
    )

    print("-" * 60)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Use a fixed location for easier testing.
    location = SIMULATION_LOCATIONS[0]

    for observation in stream_observations(
        interval_seconds=5,
        location=location,
    ):

        print_observation(observation)