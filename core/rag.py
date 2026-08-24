from __future__ import annotations
from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(__file__).resolve().parents[1]
RAG_DIR=ROOT/"rag"; RAG_DIR.mkdir(exist_ok=True)

class CaseRAG:
    def __init__(self, texts=None, metadata=None):
        self.texts=texts or []
        self.metadata=metadata or []
        self.vectorizer=TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words="english")
        self.matrix=None
        if self.texts: self.matrix=self.vectorizer.fit_transform(self.texts)

    def retrieve(self, query: str, k=5):
        if not self.texts or self.matrix is None: return []
        q=self.vectorizer.transform([query])
        sims=cosine_similarity(q,self.matrix)[0]
        idx=np.argsort(-sims)[:k]
        return [{"text":self.texts[i],"score":float(sims[i]),"metadata":self.metadata[i] if i < len(self.metadata) else {}} for i in idx]

    def save(self):
        joblib.dump(self, RAG_DIR/"case_rag.joblib")

    @staticmethod
    def load():
        return joblib.load(RAG_DIR/"case_rag.joblib")

def build_from_dataframe(df: pd.DataFrame, case_texts: list[str], target=None, lat_col=None, lon_col=None):
    meta=[]
    for i in range(len(df)):
        r=df.iloc[i]
        meta.append({
            "row":i,
            "outcome":None if target is None or pd.isna(r[target]) else str(r[target]),
            "latitude":None if lat_col is None or pd.isna(r[lat_col]) else float(r[lat_col]),
            "longitude":None if lon_col is None or pd.isna(r[lon_col]) else float(r[lon_col])
        })
    rag=CaseRAG(case_texts,meta); rag.save(); return rag

def grounded_explanation(risk: float, row: pd.Series, retrieved: list[dict], importances: pd.DataFrame) -> str:
    level = "LOW" if risk<.25 else "MODERATE" if risk<.5 else "HIGH" if risk<.75 else "CRITICAL"
    top=[]
    for _,r in importances.head(3).iterrows():
        c=r["feature"]
        if c in row.index:
            top.append(f"{c}={row[c]}")
    evidence="\n".join([f"- Similar case {i+1}: {x['text'][:220]}" for i,x in enumerate(retrieved[:3])])
    if not evidence: evidence="- No historical cases were retrieved."
    return (f"**{level} risk ({risk*100:.1f}%)**\n\n"
            f"**Model evidence:** The strongest model features include {', '.join(top) if top else 'the selected environmental variables'}.\n\n"
            f"**Retrieved historical evidence:**\n{evidence}\n\n"
            f"**Why this matters:** The explanation is grounded in the current row's model features and the retrieved historical records. "
            f"It is a prototype explanation and must not be treated as an operational warning.")
