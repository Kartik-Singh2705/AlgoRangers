"""
TerraGuard NER - Real-Time ML Pipeline

Complete pipeline:

Real-time observation
        ↓
Validation
        ↓
Location
        ↓
Feature adaptation
        ↓
ML prediction
        ↓
Risk score
        ↓
Risk level
        ↓
Explainable AI
        ↓
Historical evidence
        ↓
Alert engine
        ↓
Final result
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# TERRAGUARD CORE IMPORTS
# ============================================================

from core.data import (
    validate_realtime_observation,
)

from core.ml import (
    load_artifacts,
    predict,
    risk_level,
)

from core.explain import (
    explain_prediction,
)

from core.evidence import (
    get_historical_evidence,
)

from core.alert_engine import (
    process_alert,
)


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]


# ============================================================
# MODEL FEATURE ALIASES
# ============================================================

FEATURE_ALIASES = {

    "rainfall_mm": [
        "rainfall_24h",
    ],

    "rainfall_1h": [
        "rainfall_1h",
    ],

    "rainfall_3h": [
        "rainfall_3h",
    ],

    "rainfall_6h": [
        "rainfall_6h",
    ],

    "rainfall_24h": [
        "rainfall_24h",
    ],

    "rainfall_72h": [
        "rainfall_72h",
    ],

    "slope_deg": [
        "slope_angle",
    ],

    "slope_angle": [
        "slope_angle",
    ],

    "elevation_m": [
        "elevation",
    ],

    "elevation": [
        "elevation",
    ],

    "seismic_index": [
        "earthquake_activity",
    ],

    "earthquake_activity": [
        "earthquake_activity",
    ],

    "soil_moisture": [
        "soil_moisture",
    ],

    "soil_saturation": [
        "soil_saturation",
    ],

    "vegetation_cover": [
        "vegetation_cover",
    ],

    "ndvi": [
        "ndvi",
    ],

    "proximity_to_water": [
        "proximity_to_water",
    ],

    "distance_to_road_m": [
        "distance_to_road_m",
    ],

    "prior_events_5y": [
        "prior_events_5y",
    ],
}


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    """
    Load the trained TerraGuard model artifacts.
    """

    try:

        artifacts = load_artifacts()

    except FileNotFoundError as error:

        raise RuntimeError(
            "\nNo trained model was found.\n\n"
            "Please make sure your trained model exists "
            "inside the models directory.\n"
        ) from error

    return artifacts


# ============================================================
# VALIDATE OBSERVATION
# ============================================================

def validate_observation(
    observation: dict,
) -> None:
    """
    Validate one real-time observation.
    """

    result = validate_realtime_observation(
        observation
    )

    # --------------------------------------------------------
    # Some versions return:
    #
    #     True, []
    #
    # Others may return:
    #
    #     True
    #
    # Handle both.
    # --------------------------------------------------------

    if isinstance(
        result,
        tuple,
    ):

        valid = result[0]

        errors = (
            result[1]
            if len(result) > 1
            else []
        )

    else:

        valid = result
        errors = []

    if not valid:

        if errors:

            error_text = "\n".join(
                f"  - {error}"
                for error in errors
            )

        else:

            error_text = (
                "Unknown validation error."
            )

        raise ValueError(
            "Invalid real-time observation:\n"
            f"{error_text}"
        )


# ============================================================
# BUILD MODEL INPUT
# ============================================================

def build_model_input(
    observation: dict,
    artifacts: dict,
):
    """
    Convert the real-time observation into the exact
    feature structure expected by the trained model.
    """

    expected_features = artifacts.get(
        "features",
        [],
    )

    if not expected_features:

        raise RuntimeError(
            "No model feature list found in artifacts."
        )

    row = {}

    missing_features = []

    # --------------------------------------------------------
    # Process every model feature
    # --------------------------------------------------------

    for feature in expected_features:

        # ----------------------------------------------------
        # Exact match
        # ----------------------------------------------------

        if feature in observation:

            row[feature] = observation[
                feature
            ]

            continue

        # ----------------------------------------------------
        # Try aliases
        # ----------------------------------------------------

        aliases = FEATURE_ALIASES.get(
            feature,
            [],
        )

        found = False

        for alias in aliases:

            if alias not in observation:
                continue

            value = observation[
                alias
            ]

            # ------------------------------------------------
            # Convert percentage soil moisture to fraction
            # only if the model expects the fraction.
            # ------------------------------------------------

            if (
                feature == "soil_moisture"
                and isinstance(
                    value,
                    (int, float),
                )
                and value > 1
            ):

                value = value / 100.0

            row[feature] = value

            found = True

            break

        if found:

            continue

        # ----------------------------------------------------
        # Missing
        # ----------------------------------------------------

        missing_features.append(
            feature
        )

        row[feature] = np.nan

    X = pd.DataFrame(
        [row],
        columns=expected_features,
    )

    return X, missing_features


# ============================================================
# NORMALIZE PREDICTION
# ============================================================

def normalize_prediction(
    prediction,
) -> float:
    """
    Convert model output into a value between 0 and 1.
    """

    try:

        value = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

    except (
        TypeError,
        ValueError,
        IndexError,
    ) as error:

        raise RuntimeError(
            "Unable to convert model prediction "
            "into a risk score."
        ) from error

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


# ============================================================
# GENERATE EXPLANATION
# ============================================================

def generate_explanation(
    observation: dict,
    artifacts: dict,
    risk_probability: float,
) -> dict:
    """
    Generate explainable-AI information.

    If explanation fails, prediction continues.
    """

    try:

        explanation = explain_prediction(
            observation=observation,
            artifacts=artifacts,
            risk_score=risk_probability,
        )

        if isinstance(
            explanation,
            dict,
        ):

            return explanation

    except Exception as error:

        print()
        print(
            "WARNING: Explanation failed:"
        )

        print(
            f"  {error}"
        )

    return {

        "primary_cause":
            "Explanation unavailable",

        "factors":
            [],

        "explanation":
            "Risk prediction succeeded, "
            "but feature-level explanation "
            "is unavailable.",
    }


# ============================================================
# GET HISTORICAL EVIDENCE
# ============================================================

def generate_historical_evidence(
    observation: dict,
) -> dict:
    """
    Search historical landslide events.

    If historical search fails, the ML prediction
    is still allowed to continue.
    """

    try:

        evidence = (
            get_historical_evidence(
                observation,
                limit=5,
            )
        )

        if isinstance(
            evidence,
            dict,
        ):

            return evidence

    except Exception as error:

        print()
        print(
            "WARNING: Historical evidence failed:"
        )

        print(
            f"  {error}"
        )

    return {

        "historical_events_found":
            0,

        "nearest_event_km":
            None,

        "historical_causes":
            [],

        "evidence_text":
            "Historical evidence is "
            "currently unavailable.",

        "events":
            [],
    }


# ============================================================
# PREDICT ONE OBSERVATION
# ============================================================

def predict_observation(
    observation: dict,
    artifacts: dict | None = None,
) -> dict:
    """
    Run the complete TerraGuard prediction pipeline.
    """

    # ========================================================
    # STEP 1 — VALIDATION
    # ========================================================

    validate_observation(
        observation
    )

    # ========================================================
    # STEP 2 — LOAD MODEL
    # ========================================================

    if artifacts is None:

        artifacts = load_model()

    # ========================================================
    # STEP 3 — BUILD MODEL INPUT
    # ========================================================

    X, missing_features = (
        build_model_input(
            observation,
            artifacts,
        )
    )

    # ========================================================
    # STEP 4 — ML PREDICTION
    # ========================================================

    raw_prediction = predict(
        X,
        artifacts,
    )

    risk_probability = (
        normalize_prediction(
            raw_prediction
        )
    )

    # ========================================================
    # STEP 5 — RISK LEVEL
    # ========================================================

    level = risk_level(
        risk_probability
    )

    # ========================================================
    # STEP 6 — EXPLAINABLE AI
    # ========================================================

    explanation = (
        generate_explanation(
            observation=observation,
            artifacts=artifacts,
            risk_probability=risk_probability,
        )
    )

    # ========================================================
    # STEP 7 — HISTORICAL EVIDENCE
    # ========================================================

    historical_evidence = (
        generate_historical_evidence(
            observation
        )
    )

    # ========================================================
    # STEP 8 — MODEL NAME
    # ========================================================

    model_name = artifacts.get(
        "best_model",
        "unknown",
    )

    # ========================================================
    # STEP 9 — COMPLETE RESULT
    # ========================================================

    result = {

        # ----------------------------------------------------
        # TIME
        # ----------------------------------------------------

        "timestamp":
            observation.get(
                "timestamp"
            ),

        # ----------------------------------------------------
        # LOCATION
        # ----------------------------------------------------

        "latitude":
            observation.get(
                "latitude"
            ),

        "longitude":
            observation.get(
                "longitude"
            ),

        "place_name":
            observation.get(
                "place_name",
                "Unknown",
            ),

        "district":
            observation.get(
                "district",
                "Unknown",
            ),

        "state":
            observation.get(
                "state",
                "Unknown",
            ),

        "country":
            observation.get(
                "country",
                "India",
            ),

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        "risk_score":
            round(
                risk_probability,
                4,
            ),

        "risk_percentage":
            round(
                risk_probability * 100,
                2,
            ),

        "risk_level":
            level,

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        "model":
            model_name,

        "model_version":
            artifacts.get(
                "model_version"
            ),

        # ----------------------------------------------------
        # EXPLAINABILITY
        # ----------------------------------------------------

        "primary_cause":
            explanation.get(
                "primary_cause",
                "Unavailable",
            ),

        "explanation":
            explanation.get(
                "explanation",
                "",
            ),

        "cause_factors":
            explanation.get(
                "factors",
                [],
            ),

        # ----------------------------------------------------
        # HISTORICAL EVIDENCE
        # ----------------------------------------------------

        "historical_events_found":
            historical_evidence.get(
                "historical_events_found",
                0,
            ),

        "nearest_historical_event_km":
            historical_evidence.get(
                "nearest_event_km"
            ),

        "historical_causes":
            historical_evidence.get(
                "historical_causes",
                [],
            ),

        "historical_evidence":
            historical_evidence.get(
                "evidence_text",
                "",
            ),

        "historical_events":
            historical_evidence.get(
                "events",
                [],
            ),

        # ----------------------------------------------------
        # DATA QUALITY
        # ----------------------------------------------------

        "missing_features":
            missing_features,

        # ----------------------------------------------------
        # SOURCE
        # ----------------------------------------------------

        "source":
            observation.get(
                "source",
                "unknown",
            ),
    }

    # ========================================================
    # STEP 10 — ALERT ENGINE
    # ========================================================

    try:

        process_alert(
            result
        )

    except Exception as error:

        # ----------------------------------------------------
        # IMPORTANT:
        # An alert failure must NEVER stop the ML pipeline.
        # ----------------------------------------------------

        print()
        print(
            "WARNING: Alert engine failed:"
        )

        print(
            f"  {error}"
        )

    return result


# ============================================================
# PRINT PREDICTION
# ============================================================

def print_prediction(
    result: dict,
) -> None:
    """
    Print the complete TerraGuard result.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "TERRAGUARD NER - "
        "REAL-TIME RISK PREDICTION"
    )

    print(
        "=" * 70
    )

    # ========================================================
    # TIME
    # ========================================================

    print(
        f"Time       : "
        f"{result.get('timestamp', 'Unknown')}"
    )

    # ========================================================
    # LOCATION
    # ========================================================

    print(
        f"Location   : "
        f"{result.get('place_name', 'Unknown')}, "
        f"{result.get('state', 'Unknown')}"
    )

    print(
        f"District   : "
        f"{result.get('district', 'Unknown')}"
    )

    latitude = result.get(
        "latitude"
    )

    longitude = result.get(
        "longitude"
    )

    if (
        isinstance(
            latitude,
            (int, float),
        )
        and
        isinstance(
            longitude,
            (int, float),
        )
    ):

        print(
            f"Coordinates: "
            f"{latitude:.6f}, "
            f"{longitude:.6f}"
        )

    else:

        print(
            f"Coordinates: "
            f"{latitude}, {longitude}"
        )

    # ========================================================
    # RISK
    # ========================================================

    print()

    print(
        f"Risk Score : "
        f"{result.get('risk_percentage', 0):.2f}%"
    )

    print(
        f"Risk Level : "
        f"{result.get('risk_level', 'UNKNOWN')}"
    )

    # ========================================================
    # MODEL
    # ========================================================

    print(
        f"Model      : "
        f"{result.get('model', 'unknown')}"
    )

    if result.get(
        "model_version"
    ):

        print(
            f"Version    : "
            f"{result['model_version']}"
        )

    # ========================================================
    # EXPLANATION
    # ========================================================

    print()
    print(
        "WHY IS THIS LOCATION AT RISK?"
    )

    print(
        "-" * 70
    )

    print(
        f"Primary factor: "
        f"{result.get('primary_cause', 'Unavailable')}"
    )

    factors = result.get(
        "cause_factors",
        [],
    )

    if factors:

        print()

        print(
            "Top contributing factors:"
        )

        for index, factor in enumerate(
            factors,
            start=1,
        ):

            label = factor.get(
                "label",
                factor.get(
                    "feature",
                    "Unknown",
                ),
            )

            importance = factor.get(
                "importance_percentage",
                0,
            )

            value = factor.get(
                "value",
                "Unknown",
            )

            condition = factor.get(
                "condition",
                "unknown",
            )

            effect = factor.get(
                "effect",
                "",
            )

            print(
                f"  {index}. "
                f"{label}: "
                f"{importance:.2f}%"
            )

            print(
                f"      Current value: "
                f"{value}"
            )

            print(
                f"      Condition: "
                f"{condition}"
            )

            if effect:

                print(
                    f"      Effect: "
                    f"{effect}"
                )

    else:

        print(
            "  Feature-level explanation "
            "is unavailable."
        )

    print()

    print(
        "Explanation:"
    )

    print(
        f"  {result.get('explanation', 'Unavailable')}"
    )

    # ========================================================
    # HISTORICAL EVIDENCE
    # ========================================================

    print()

    print(
        "HISTORICAL EVIDENCE"
    )

    print(
        "-" * 70
    )

    event_count = result.get(
        "historical_events_found",
        0,
    )

    print(
        f"Similar historical events: "
        f"{event_count}"
    )

    nearest = result.get(
        "nearest_historical_event_km"
    )

    if nearest is not None:

        try:

            print(
                f"Nearest historical event: "
                f"{float(nearest):.2f} km"
            )

        except (
            TypeError,
            ValueError,
        ):

            print(
                f"Nearest historical event: "
                f"{nearest}"
            )

    causes = result.get(
        "historical_causes",
        [],
    )

    if causes:

        print()

        print(
            "Historical causes:"
        )

        for cause in causes[:3]:

            print(
                f"  • {cause}"
            )

    print()

    print(
        result.get(
            "historical_evidence",
            "No historical evidence available.",
        )
    )

    # ========================================================
    # MISSING FEATURES
    # ========================================================

    missing = result.get(
        "missing_features",
        [],
    )

    if missing:

        print()

        print(
            "WARNING: Missing model features:"
        )

        for feature in missing:

            print(
                f"  - {feature}"
            )

    # ========================================================
    # SOURCE
    # ========================================================

    print()

    print(
        f"Data Source: "
        f"{result.get('source', 'unknown')}"
    )

    # ========================================================
    # END
    # ========================================================

    print(
        "=" * 70
    )


# ============================================================
# SINGLE TEST
# ============================================================

def run_single_test():
    """
    Generate one simulated observation and run
    the complete TerraGuard pipeline.
    """

    from realtime.simulator import (
        generate_validated_observation,
    )

    print()
    print(
        "Starting TerraGuard "
        "real-time ML test..."
    )

    # --------------------------------------------------------
    # Generate observation
    # --------------------------------------------------------

    observation = (
        generate_validated_observation()
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    artifacts = load_model()

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    result = predict_observation(
        observation=observation,
        artifacts=artifacts,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print_prediction(
        result
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_single_test()