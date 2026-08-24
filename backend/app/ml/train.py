import os
import numpy as np
import pandas as pd


def generate_dataset(n_samples=5000, random_state=42):
    np.random.seed(random_state)

    rainfall_24h = np.random.gamma(2.0, 30.0, n_samples)
    rainfall_3d = rainfall_24h + np.random.gamma(2.5, 45.0, n_samples)
    rainfall_7d = rainfall_3d + np.random.gamma(3.0, 60.0, n_samples)

    soil_moisture = np.random.uniform(0.2, 1.0, n_samples)
    elevation = np.random.uniform(50, 3000, n_samples)
    slope = np.random.uniform(2, 50, n_samples)
    ndvi = np.random.uniform(0.1, 0.9, n_samples)
    historical_landslides = np.random.poisson(2, n_samples)

    # Synthetic relationship for development only.
    risk_score = (
        0.025 * rainfall_24h
        + 0.012 * rainfall_3d
        + 0.006 * rainfall_7d
        + 2.0 * soil_moisture
        + 0.08 * slope
        + 0.35 * historical_landslides
        - 1.5 * ndvi
    )

    probability = 1 / (1 + np.exp(-(risk_score - 8)))

    landslide = np.random.binomial(1, probability)

    df = pd.DataFrame({
        "rainfall_24h": rainfall_24h,
        "rainfall_3d": rainfall_3d,
        "rainfall_7d": rainfall_7d,
        "soil_moisture": soil_moisture,
        "elevation": elevation,
        "slope": slope,
        "ndvi": ndvi,
        "historical_landslides": historical_landslides,
        "landslide": landslide
    })

    return df


if __name__ == "__main__":
    df = generate_dataset()

    output_dir = "backend/data/mock"
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "landslide_mock.csv")

    df.to_csv(output_file, index=False)

    print(f"Dataset created: {output_file}")
    print(f"Rows: {len(df)}")
    print(df.head())