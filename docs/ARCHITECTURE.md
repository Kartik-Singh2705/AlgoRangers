# TerraGuard NER prototype architecture

```text
                 TODAY (Kaggle prototype)
Kaggle CSV
   |
   v
[Data Adapter]
   |
   +--> SQLite --------------------------+
   |                                     |
   v                                     v
[Feature Builder]                    [Historical Cases]
   |                                     |
   v                                     v
[Random Forest / HGB]              [RAG Retrieval]
   |                                     |
   +------------> Risk Score <------------+
                         |
                         v
                [Explanation Engine]
                         |
                         v
                  Streamlit Dashboard


                 LATER (live system)
IMD API ─┐
GSI ─────┤
Bhuvan ──┤
DEM ─────┤--> [Scheduled ingestion] --> PostgreSQL + PostGIS
Sensors ─┤                                  |
Field App┘                                  v
                                     ML + RAG services
                                             |
                                             v
                                    React/Leaflet + Alerts
```

The critical design decision is the **adapter boundary**. A government source should produce the same normalized observation fields used by the Kaggle adapter. The dashboard and model code should not care whether a row came from Kaggle or an API.

## ML

- Supervised: Random Forest + HistGradientBoosting.
- Model selection: highest validation ROC-AUC.
- If no binary target is present: Isolation Forest anomaly mode.
- Risk levels: LOW <25%, MODERATE 25–50%, HIGH 50–75%, CRITICAL >=75%.
- These thresholds are presentation thresholds, not government warning standards.

## RAG

Historical dataset rows become retrievable case documents. The current observation is converted into a query. Similar cases are retrieved with TF-IDF cosine similarity. The explanation combines:
1. current model risk,
2. top model features,
3. retrieved historical evidence.

For a production system, replace TF-IDF with embeddings + FAISS/Pinecone and optionally place an LLM behind the same generator interface.

## Database

SQLite is intentional for the prototype because it runs without credentials or a server. `database/postgres_postgis_schema.sql` is the migration target.
