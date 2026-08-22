import joblib
import pandas as pd


MODEL_PATH = "backend/models/landslide_xgb.pkl"


FEATURES = [
    "rainfall_24h",
    "rainfall_3d",
    "rainfall_7d",
    "soil_moisture",
    "elevation",
    "slope",
    "ndvi",
    "historical_landslides"
]


model = joblib.load(MODEL_PATH)


def predict_risk(data):

    df = pd.DataFrame([data], columns=FEATURES)

    probability = float(model.predict_proba(df)[0][1])

    if probability < 0.30:
        risk = "LOW"
    elif probability < 0.60:
        risk = "MODERATE"
    elif probability < 0.80:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return {
        "probability": round(probability, 4),
        "risk": risk
    }