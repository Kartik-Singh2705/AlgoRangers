"""
Generate a small SYNTHETIC demo CSV so judges/teammates can run the full
pipeline in under a minute without first downloading a Kaggle dataset.

This is a convenience for rehearsing the demo flow only. It is NOT the
dataset your model should be trained on for the actual submission —
keep using your selected Kaggle CSV in data/raw/ for that (see README).

Usage:
    python scripts/generate_demo_data.py
    python scripts/train_models.py --data data/raw/demo_synthetic_ner.csv --target landslide
    streamlit run app.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "raw" / "demo_synthetic_ner.csv"

# Rough bounding box covering the North Eastern Region states, used only
# to scatter demo points so the GIS map tab has something to plot.
NER_LAT_RANGE = (23.8, 29.5)
NER_LON_RANGE = (89.7, 97.4)


def main(n: int = 600, seed: int = 42):
    rng = np.random.default_rng(seed)

    rainfall_mm = rng.gamma(shape=2.2, scale=45, size=n)              # 24h rainfall
    slope_deg = rng.uniform(5, 60, size=n)                             # terrain slope
    soil_moisture = np.clip(rng.normal(0.4, 0.15, size=n), 0, 1)       # fraction saturated
    ndvi = np.clip(rng.normal(0.55, 0.2, size=n), -0.1, 1)             # vegetation cover
    elevation_m = rng.uniform(50, 2200, size=n)
    distance_to_road_m = rng.exponential(300, size=n)
    seismic_index = np.clip(rng.normal(2.0, 1.0, size=n), 0, 8)
    prior_events_5y = rng.poisson(0.6, size=n)

    # A hand-tuned latent risk score so the synthetic labels are at least
    # directionally sensible (higher rain/slope/moisture -> more risk;
    # more vegetation -> less risk). This is illustrative only.
    latent = (
        0.032 * rainfall_mm
        + 0.045 * slope_deg
        + 3.2 * soil_moisture
        - 1.6 * ndvi
        + 0.35 * seismic_index
        + 0.5 * prior_events_5y
        - 0.0009 * distance_to_road_m
        + rng.normal(0, 1.1, size=n)
    )
    threshold = np.quantile(latent, 0.74)  # ~26% positive class, imbalanced like real events
    landslide = (latent > threshold).astype(int)

    lat = rng.uniform(*NER_LAT_RANGE, size=n)
    lon = rng.uniform(*NER_LON_RANGE, size=n)

    df = pd.DataFrame({
        "latitude": lat.round(4),
        "longitude": lon.round(4),
        "rainfall_mm": rainfall_mm.round(1),
        "slope_deg": slope_deg.round(1),
        "soil_moisture": soil_moisture.round(3),
        "ndvi": ndvi.round(3),
        "elevation_m": elevation_m.round(1),
        "distance_to_road_m": distance_to_road_m.round(1),
        "seismic_index": seismic_index.round(2),
        "prior_events_5y": prior_events_5y,
        "landslide": landslide,
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} SYNTHETIC demo rows -> {OUT}")
    print(f"Positive rate: {df['landslide'].mean():.1%}")
    print("Reminder: this file is for rehearsing the demo flow only, not for your real submission model.")


if __name__ == "__main__":
    main()
