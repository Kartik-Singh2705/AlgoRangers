from __future__ import annotations
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from .data import clean_features

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/"models"
MODEL_DIR.mkdir(exist_ok=True)

def risk_level(p: float) -> str:
    if p < .25: return "LOW"
    if p < .50: return "MODERATE"
    if p < .75: return "HIGH"
    return "CRITICAL"

def encode_binary(y: pd.Series) -> pd.Series | None:
    vals=list(pd.Series(y.dropna()).unique())
    if len(vals)!=2: return None
    # Robust binary mapping: numeric 0/1 stays intuitive; otherwise first/second class.
    if set(vals).issubset({0,1,0.0,1.0}):
        return pd.to_numeric(y, errors="coerce").astype("Int64")
    mapping={vals[0]:0, vals[1]:1}
    return y.map(mapping).astype("Int64")

def train_models(df: pd.DataFrame, features: list[str], target: str | None):
    X=clean_features(df, features)
    MODEL_DIR.mkdir(exist_ok=True)
    artifacts={"features":features,"target":target,"task":"classification"}
    metrics={}
    if target is None:
        model=Pipeline([("imputer",SimpleImputer(strategy="median")),
                        ("model",IsolationForest(n_estimators=400, contamination="auto", random_state=42))])
        model.fit(X)
        artifacts["models"]={"isolation_forest":model}
        artifacts["task"]="unsupervised_anomaly"
        joblib.dump(artifacts, MODEL_DIR/"risk_models.joblib")
        return artifacts, {"task":"unsupervised_anomaly","note":"No binary target detected; anomaly score is not a landslide probability."}

    y=encode_binary(df[target])
    valid=y.notna()
    X=X.loc[valid]; y=y.loc[valid].astype(int)
    if y.nunique()!=2:
        raise ValueError(f"Target '{target}' is not binary after encoding.")
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.20,random_state=42,stratify=y)
    models={
        "random_forest":Pipeline([("imputer",SimpleImputer(strategy="median")),
            ("model",RandomForestClassifier(n_estimators=400,max_depth=14,min_samples_leaf=2,class_weight="balanced",random_state=42,n_jobs=-1))]),
        "hist_gradient_boosting":Pipeline([("imputer",SimpleImputer(strategy="median")),
            ("model",HistGradientBoostingClassifier(max_iter=300,learning_rate=.06,max_leaf_nodes=31,random_state=42))])
    }
    best_name=None; best_auc=-1
    for name,m in models.items():
        m.fit(Xtr,ytr)
        p=m.predict_proba(Xte)[:,1]
        auc=roc_auc_score(yte,p)
        metrics[name]={
            "roc_auc":float(auc),
            "accuracy":float(accuracy_score(yte,(p>=.5).astype(int))),
            "precision":float(precision_score(yte,(p>=.5).astype(int),zero_division=0)),
            "recall":float(recall_score(yte,(p>=.5).astype(int),zero_division=0)),
            "f1":float(f1_score(yte,(p>=.5).astype(int),zero_division=0))
        }
        if auc>best_auc: best_auc=auc; best_name=name
    artifacts["models"]=models
    artifacts["best_model"]=best_name
    artifacts["class_mapping"]="0=non-landslide/safe, 1=landslide/affected (inferred)"
    joblib.dump(artifacts, MODEL_DIR/"risk_models.joblib")
    (MODEL_DIR/"metrics.json").write_text(json.dumps(metrics,indent=2))
    return artifacts, metrics

def load_artifacts():
    return joblib.load(MODEL_DIR/"risk_models.joblib")

def predict(df: pd.DataFrame, artifacts=None):
    artifacts=artifacts or load_artifacts()
    X=clean_features(df, artifacts["features"])
    if artifacts["task"]=="unsupervised_anomaly":
        score=artifacts["models"]["isolation_forest"].decision_function(X)
        # Convert anomaly score to a presentation-friendly 0..1 risk scale.
        risk=1/(1+np.exp(5*score))
        return risk
    model=artifacts["models"][artifacts["best_model"]]
    return model.predict_proba(X)[:,1]

def feature_importance(artifacts):
    name=artifacts.get("best_model")
    if not name: return pd.DataFrame(columns=["feature","importance"])
    m=artifacts["models"][name].named_steps["model"]
    if not hasattr(m,"feature_importances_"):
        return pd.DataFrame(columns=["feature","importance"])
    return pd.DataFrame({"feature":artifacts["features"],"importance":m.feature_importances_}).sort_values("importance",ascending=False)
