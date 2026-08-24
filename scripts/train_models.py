from pathlib import Path
import argparse, json, sys
import pandas as pd

# Make project-root imports work when this file is launched directly with
# `python scripts/train_models.py` on Windows/Linux.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.data import load_csv, infer_target, numeric_features, map_columns, make_case_text
from core.ml import train_models
from core.rag import build_from_dataframe
from core.db import init_db, clear_dynamic, insert_observations, insert_cases

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--target",default=None)
    args=ap.parse_args()
    path=Path(args.data)
    if not path.is_absolute():
        path = Path.cwd() / path
    df=load_csv(path)
    target=infer_target(df,args.target)
    features=numeric_features(df,target)
    if len(features)<2: raise SystemExit("Need at least 2 numeric predictor columns.")
    lat,lon=map_columns(df)
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"Target: {target or 'NONE -> unsupervised anomaly mode'}")
    print(f"Features: {features}")
    artifacts,metrics=train_models(df,features,target)
    texts=[make_case_text(r,target) for _,r in df.iterrows()]
    build_from_dataframe(df,texts,target,lat,lon)
    init_db(); clear_dynamic()
    insert_observations(df,"Kaggle",lat,lon)
    insert_cases(df,"Kaggle",texts,target,lat,lon)
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "models" / "run_summary.json").write_text(json.dumps({"target":target,"features":features,"metrics":metrics},indent=2))
    print(json.dumps(metrics,indent=2))
    print("Training complete. Run: streamlit run app.py")
if __name__=="__main__": main()
