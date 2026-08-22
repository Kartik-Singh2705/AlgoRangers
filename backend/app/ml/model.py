import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from xgboost import XGBClassifier


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

TARGET = "landslide"


def train_model():

    data_path = "backend/data/mock/landslide_mock.csv"

    df = pd.read_csv(data_path)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nModel Evaluation")
    print("----------------")

    print("Accuracy :", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall   :", recall_score(y_test, predictions))
    print("F1 Score :", f1_score(y_test, predictions))
    print("ROC-AUC  :", roc_auc_score(y_test, probabilities))

    os.makedirs("backend/models", exist_ok=True)

    model_path = "backend/models/landslide_xgb.pkl"

    joblib.dump(model, model_path)

    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    train_model()