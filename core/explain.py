"""
TerraGuard NER - Explainable AI Engine

This module explains why the current ML model produced
a particular landslide-risk prediction.

For models such as HistGradientBoosting that do not expose
feature_importances_, this module uses local sensitivity:

    change one feature
            ↓
    run the model again
            ↓
    measure change in risk
            ↓
    rank features by influence

IMPORTANT:
These are model-influence indicators, not proof that a
physical factor caused a landslide.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.ml import predict


# ============================================================
# HUMAN-READABLE FEATURE NAMES
# ============================================================

FEATURE_LABELS = {

    "rainfall_mm":
        "24-hour rainfall",

    "rainfall_1h":
        "1-hour rainfall",

    "rainfall_3h":
        "3-hour rainfall",

    "rainfall_6h":
        "6-hour rainfall",

    "rainfall_24h":
        "24-hour rainfall",

    "rainfall_72h":
        "72-hour rainfall",

    "slope_deg":
        "Slope angle",

    "slope_angle":
        "Slope angle",

    "soil_moisture":
        "Soil moisture",

    "soil_saturation":
        "Soil saturation",

    "vegetation_cover":
        "Vegetation cover",

    "ndvi":
        "Vegetation index (NDVI)",

    "elevation_m":
        "Elevation",

    "elevation":
        "Elevation",

    "seismic_index":
        "Seismic activity",

    "earthquake_activity":
        "Earthquake activity",

    "proximity_to_water":
        "Proximity to water",

    "distance_to_road_m":
        "Distance to road",

    "prior_events_5y":
        "Previous landslide events",
}


# ============================================================
# FEATURE ALIASES
# ============================================================

FEATURE_ALIASES = {

    "rainfall_mm":
        "rainfall_24h",

    "slope_deg":
        "slope_angle",

    "elevation_m":
        "elevation",

    "seismic_index":
        "earthquake_activity",
}


# ============================================================
# GET HUMAN-READABLE LABEL
# ============================================================

def get_feature_label(
    feature: str,
) -> str:
    """
    Convert a technical feature name into a
    human-readable name.
    """

    return FEATURE_LABELS.get(
        feature,
        feature.replace(
            "_",
            " ",
        ).title(),
    )


# ============================================================
# GET OBSERVATION VALUE
# ============================================================

def get_feature_value(
    observation: dict,
    feature: str,
) -> Any:
    """
    Retrieve a feature value from the real-time observation.

    Supports aliases between model feature names and
    real-time observation names.
    """

    # --------------------------------------------------------
    # Exact match
    # --------------------------------------------------------

    if feature in observation:

        return observation[feature]

    # --------------------------------------------------------
    # Alias
    # --------------------------------------------------------

    alias = FEATURE_ALIASES.get(
        feature
    )

    if alias and alias in observation:

        return observation[alias]

    return None


# ============================================================
# CONDITION DESCRIPTION
# ============================================================

def describe_feature_condition(
    feature: str,
    value: Any,
) -> str:
    """
    Convert a feature value into a human-readable
    environmental condition.
    """

    if value is None:

        return "data unavailable"

    try:

        value = float(value)

    except (
        TypeError,
        ValueError,
    ):

        return str(value)

    # --------------------------------------------------------
    # Rainfall
    # --------------------------------------------------------

    if feature in {
        "rainfall_mm",
        "rainfall_24h",
    }:

        if value >= 200:
            return "extremely high"

        if value >= 100:
            return "high"

        if value >= 50:
            return "moderate"

        return "low"

    # --------------------------------------------------------
    # 1-hour rainfall
    # --------------------------------------------------------

    if feature == "rainfall_1h":

        if value >= 50:
            return "extremely high"

        if value >= 25:
            return "high"

        if value >= 10:
            return "moderate"

        return "low"

    # --------------------------------------------------------
    # Slope
    # --------------------------------------------------------

    if feature in {
        "slope_deg",
        "slope_angle",
    }:

        if value >= 40:
            return "very steep"

        if value >= 30:
            return "steep"

        if value >= 15:
            return "moderate"

        return "gentle"

    # --------------------------------------------------------
    # Soil saturation
    # --------------------------------------------------------

    if feature == "soil_saturation":

        if value >= 85:
            return "very high"

        if value >= 70:
            return "high"

        if value >= 50:
            return "moderate"

        return "low"

    # --------------------------------------------------------
    # Soil moisture
    # --------------------------------------------------------

    if feature == "soil_moisture":

        if value > 1:

            if value >= 85:
                return "very high"

            if value >= 70:
                return "high"

            if value >= 50:
                return "moderate"

            return "low"

        else:

            if value >= 0.85:
                return "very high"

            if value >= 0.70:
                return "high"

            if value >= 0.50:
                return "moderate"

            return "low"

    # --------------------------------------------------------
    # Vegetation
    # --------------------------------------------------------

    if feature == "vegetation_cover":

        if value < 20:
            return "very low"

        if value < 40:
            return "low"

        if value < 70:
            return "moderate"

        return "high"

    # --------------------------------------------------------
    # NDVI
    # --------------------------------------------------------

    if feature == "ndvi":

        if value < 0.2:
            return "very low vegetation"

        if value < 0.4:
            return "low vegetation"

        if value < 0.6:
            return "moderate vegetation"

        return "high vegetation"

    # --------------------------------------------------------
    # Earthquake
    # --------------------------------------------------------

    if feature in {
        "earthquake_activity",
        "seismic_index",
    }:

        if value >= 4:
            return "high"

        if value >= 2:
            return "moderate"

        return "low"

    # --------------------------------------------------------
    # Historical events
    # --------------------------------------------------------

    if feature == "prior_events_5y":

        if value >= 4:
            return "many previous events"

        if value >= 2:
            return "previous events detected"

        return "few or no previous events"

    # --------------------------------------------------------
    # Distance from road
    # --------------------------------------------------------

    if feature == "distance_to_road_m":

        if value <= 100:
            return "very close to road"

        if value <= 500:
            return "close to road"

        if value <= 2000:
            return "moderate distance"

        return "far from road"

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return f"{value:.2f}"


# ============================================================
# EXTRACT MODEL FEATURES
# ============================================================

def get_model_features(
    artifacts: dict,
) -> list[str]:
    """
    Retrieve the exact feature list used by the model.
    """

    features = artifacts.get(
        "features",
        [],
    )

    if not features:

        raise RuntimeError(
            "No model feature list found "
            "inside the artifacts."
        )

    return list(features)


# ============================================================
# GET CURRENT MODEL
# ============================================================

def get_model(
    artifacts: dict,
):
    """
    Retrieve the trained model from the artifact structure
    used by TerraGuard.
    """

    models = artifacts.get(
        "models"
    )

    best_model_name = artifacts.get(
        "best_model"
    )

    if models is None:

        raise RuntimeError(
            "Model dictionary not found "
            "in artifacts."
        )

    if best_model_name is None:

        raise RuntimeError(
            "best_model not found "
            "in artifacts."
        )

    if best_model_name not in models:

        raise RuntimeError(
            f"Model '{best_model_name}' "
            "not found in artifacts."
        )

    return models[
        best_model_name
    ]


# ============================================================
# BUILD BASE MODEL INPUT
# ============================================================

def build_base_input(
    observation: dict,
    artifacts: dict,
) -> pd.DataFrame:
    """
    Construct a DataFrame containing exactly the model
    features.

    Missing values are represented as NaN and are handled
    by the existing TerraGuard ML preprocessing pipeline.
    """

    features = get_model_features(
        artifacts
    )

    row = {}

    for feature in features:

        value = get_feature_value(
            observation,
            feature,
        )

        row[feature] = value

    return pd.DataFrame(
        [row],
        columns=features,
    )


# ============================================================
# GET BASELINE PREDICTION
# ============================================================

def get_base_prediction(
    observation: dict,
    artifacts: dict,
) -> float:
    """
    Get the model's current risk prediction.

    We intentionally use the existing core.ml.predict()
    function so that the exact same preprocessing used by
    the production pipeline is applied.
    """

    X = build_base_input(
        observation,
        artifacts,
    )

    prediction = predict(
        X,
        artifacts,
    )

    value = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


# ============================================================
# PERTURB ONE FEATURE
# ============================================================

def perturb_value(
    value: float,
    feature: str,
    direction: float,
) -> float:
    """
    Generate a slightly changed value.

    direction:
        +1 = increase
        -1 = decrease
    """

    value = float(value)

    # --------------------------------------------------------
    # Feature-specific perturbation size
    # --------------------------------------------------------

    if feature in {
        "rainfall_mm",
        "rainfall_1h",
        "rainfall_3h",
        "rainfall_6h",
        "rainfall_24h",
        "rainfall_72h",
    }:

        delta = max(
            abs(value) * 0.10,
            1.0,
        )

    elif feature in {
        "slope_deg",
        "slope_angle",
    }:

        delta = 2.0

    elif feature in {
        "soil_saturation",
        "vegetation_cover",
    }:

        delta = 5.0

    elif feature == "soil_moisture":

        if value > 1:

            delta = 5.0

        else:

            delta = 0.05

    elif feature == "ndvi":

        delta = 0.05

    elif feature in {
        "earthquake_activity",
        "seismic_index",
    }:

        delta = 0.5

    elif feature in {
        "elevation",
        "elevation_m",
    }:

        delta = max(
            abs(value) * 0.05,
            10.0,
        )

    elif feature == "proximity_to_water":

        delta = max(
            abs(value) * 0.10,
            10.0,
        )

    elif feature == "distance_to_road_m":

        delta = max(
            abs(value) * 0.10,
            10.0,
        )

    elif feature == "prior_events_5y":

        delta = 1.0

    else:

        delta = max(
            abs(value) * 0.10,
            0.01,
        )

    new_value = (
        value
        + direction * delta
    )

    # --------------------------------------------------------
    # Prevent physically impossible values
    # --------------------------------------------------------

    if feature in {
        "soil_saturation",
        "vegetation_cover",
    }:

        new_value = max(
            0.0,
            min(
                100.0,
                new_value,
            ),
        )

    elif feature == "soil_moisture":

        if value > 1:

            new_value = max(
                0.0,
                min(
                    100.0,
                    new_value,
                ),
            )

        else:

            new_value = max(
                0.0,
                min(
                    1.0,
                    new_value,
                ),
            )

    elif feature == "ndvi":

        new_value = max(
            -1.0,
            min(
                1.0,
                new_value,
            ),
        )

    elif feature in {
        "slope_deg",
        "slope_angle",
    }:

        new_value = max(
            0.0,
            min(
                90.0,
                new_value,
            ),
        )

    elif feature in {
        "earthquake_activity",
        "seismic_index",
    }:

        new_value = max(
            0.0,
            new_value,
        )

    elif feature in {
        "rainfall_mm",
        "rainfall_1h",
        "rainfall_3h",
        "rainfall_6h",
        "rainfall_24h",
        "rainfall_72h",
        "elevation",
        "elevation_m",
        "proximity_to_water",
        "distance_to_road_m",
        "prior_events_5y",
    }:

        new_value = max(
            0.0,
            new_value,
        )

    return new_value


# ============================================================
# LOCAL SENSITIVITY
# ============================================================

def calculate_local_influence(
    observation: dict,
    artifacts: dict,
    baseline_prediction: float,
) -> dict[str, dict]:
    """
    Calculate how sensitive the current prediction is
    to each numerical feature.

    For each feature:

        prediction(feature + delta)
        prediction(feature - delta)

    The difference is used as a local influence score.

    This is NOT a causal claim.
    """

    features = get_model_features(
        artifacts
    )

    influence = {}

    for feature in features:

        current_value = get_feature_value(
            observation,
            feature,
        )

        # ----------------------------------------------------
        # Only numerical features can be perturbed.
        # ----------------------------------------------------

        if current_value is None:

            continue

        try:

            numeric_value = float(
                current_value
            )

        except (
            TypeError,
            ValueError,
        ):

            continue

        # ----------------------------------------------------
        # Create HIGH and LOW versions
        # ----------------------------------------------------

        high_value = perturb_value(
            numeric_value,
            feature,
            +1,
        )

        low_value = perturb_value(
            numeric_value,
            feature,
            -1,
        )

        # ----------------------------------------------------
        # Create modified observations
        # ----------------------------------------------------

        high_observation = (
            observation.copy()
        )

        low_observation = (
            observation.copy()
        )

        # ----------------------------------------------------
        # Important:
        # update the actual observation field rather than
        # necessarily the model feature alias.
        # ----------------------------------------------------

        if feature in high_observation:

            high_observation[
                feature
            ] = high_value

            low_observation[
                feature
            ] = low_value

        else:

            alias = FEATURE_ALIASES.get(
                feature
            )

            if alias is None:
                continue

            if alias not in high_observation:
                continue

            high_observation[
                alias
            ] = high_value

            low_observation[
                alias
            ] = low_value

        # ----------------------------------------------------
        # Run predictions
        # ----------------------------------------------------

        try:

            high_prediction = (
                get_base_prediction(
                    high_observation,
                    artifacts,
                )
            )

            low_prediction = (
                get_base_prediction(
                    low_observation,
                    artifacts,
                )
            )

        except Exception:

            continue

        # ----------------------------------------------------
        # Local sensitivity
        # ----------------------------------------------------

        sensitivity = abs(
            high_prediction
            - low_prediction
        )

        # Direction of influence
        #
        # Positive means increasing the feature increased
        # the model's risk prediction.
        #
        # Negative means increasing the feature decreased
        # the model's risk prediction.

        direction = (
            high_prediction
            - low_prediction
        )

        influence[
            feature
        ] = {

            "sensitivity": float(
                sensitivity
            ),

            "direction": float(
                direction
            ),

            "low_prediction": float(
                low_prediction
            ),

            "high_prediction": float(
                high_prediction
            ),

            "current_value": numeric_value,
        }

    return influence


# ============================================================
# NORMALIZE INFLUENCE
# ============================================================

def normalize_influence(
    influence: dict[str, dict],
) -> dict[str, float]:
    """
    Convert local sensitivity into percentages.
    """

    if not influence:

        return {}

    total = sum(
        max(
            0.0,
            item["sensitivity"],
        )
        for item in influence.values()
    )

    if total <= 0:

        return {}

    return {

        feature: (
            max(
                0.0,
                item["sensitivity"],
            )
            / total
        ) * 100

        for feature, item
        in influence.items()
    }


# ============================================================
# MAIN EXPLANATION FUNCTION
# ============================================================

def explain_prediction(
    observation: dict,
    artifacts: dict,
    risk_score: float,
    top_n: int = 5,
) -> dict:
    """
    Explain a single risk prediction.

    Returns:

        primary_cause
        factors
        explanation
        risk_score
    """

    # --------------------------------------------------------
    # Calculate local influence
    # --------------------------------------------------------

    influence = (
        calculate_local_influence(
            observation=observation,
            artifacts=artifacts,
            baseline_prediction=risk_score,
        )
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    percentages = (
        normalize_influence(
            influence
        )
    )

    # --------------------------------------------------------
    # Rank
    # --------------------------------------------------------

    ranked = sorted(
        percentages.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    factors = []

    for feature, percentage in ranked[
        :top_n
    ]:

        current_value = get_feature_value(
            observation,
            feature,
        )

        condition = (
            describe_feature_condition(
                feature,
                current_value,
            )
        )

        information = influence[
            feature
        ]

        direction = (
            information["direction"]
        )

        if direction > 0:

            effect = (
                "increasing this factor "
                "increases predicted risk"
            )

        elif direction < 0:

            effect = (
                "increasing this factor "
                "decreases predicted risk"
            )

        else:

            effect = (
                "has little local effect "
                "on predicted risk"
            )

        factors.append(
            {
                "feature": feature,

                "label": get_feature_label(
                    feature
                ),

                "value": current_value,

                "importance_percentage":
                    round(
                        percentage,
                        2,
                    ),

                "condition": condition,

                "effect": effect,

                "direction":
                    round(
                        direction,
                        5,
                    ),
            }
        )

    # --------------------------------------------------------
    # Primary cause / factor
    # --------------------------------------------------------

    if factors:

        primary_cause = factors[
            0
        ]["label"]

    else:

        primary_cause = (
            "Explanation unavailable"
        )

    # --------------------------------------------------------
    # Generate explanation
    # --------------------------------------------------------

    if factors:

        first = factors[0]

        explanation = (
            f"The strongest model-influencing factor "
            f"for this observation is "
            f"{first['label']}. "
            f"Its current condition is "
            f"{first['condition']}, and "
            f"{first['effect']}."
        )

        if len(factors) > 1:

            second = factors[1]

            explanation += (
                f" Another important factor is "
                f"{second['label']}, "
                f"currently described as "
                f"{second['condition']}."
            )

    else:

        explanation = (
            "The current model could not produce "
            "a feature-level explanation for this "
            "observation."
        )

    return {

        "primary_cause":
            primary_cause,

        "factors":
            factors,

        "explanation":
            explanation,

        "risk_score":
            round(
                float(risk_score),
                4,
            ),
    }


# ============================================================
# TERMINAL DISPLAY
# ============================================================

def print_explanation(
    explanation: dict,
) -> None:
    """
    Print an explanation in the terminal.
    """

    print()
    print(
        "=" * 70
    )

    print(
        "MODEL-INFLUENCING FACTORS"
    )

    print(
        "=" * 70
    )

    print(
        f"Primary factor: "
        f"{explanation['primary_cause']}"
    )

    print()

    factors = explanation.get(
        "factors",
        [],
    )

    for index, factor in enumerate(
        factors,
        start=1,
    ):

        print(
            f"{index}. "
            f"{factor['label']}"
        )

        print(
            f"   Current value : "
            f"{factor['value']}"
        )

        print(
            f"   Condition     : "
            f"{factor['condition']}"
        )

        print(
            f"   Influence     : "
            f"{factor['importance_percentage']:.2f}%"
        )

        print(
            f"   Effect        : "
            f"{factor['effect']}"
        )

        print()

    print(
        "Explanation:"
    )

    print(
        explanation["explanation"]
    )

    print(
        "=" * 70
    )


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    from realtime.simulator import (
        generate_validated_observation,
    )

    from core.ml import (
        load_artifacts,
    )

    print(
        "Testing TerraGuard explainability..."
    )

    observation = (
        generate_validated_observation()
    )

    artifacts = load_artifacts()

    risk = get_base_prediction(
        observation,
        artifacts,
    )

    explanation = explain_prediction(
        observation=observation,
        artifacts=artifacts,
        risk_score=risk,
    )

    print_explanation(
        explanation
    )