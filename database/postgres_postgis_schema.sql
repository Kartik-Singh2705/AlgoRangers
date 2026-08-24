-- Future production schema
-- Keep SQLite for the presentation. Move to PostgreSQL + PostGIS when government feeds are connected.

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS observations (
  id BIGSERIAL PRIMARY KEY,
  source TEXT NOT NULL,
  observed_at TIMESTAMPTZ,
  latitude DOUBLE PRECISION,
  longitude DOUBLE PRECISION,
  geometry GEOGRAPHY(POINT,4326),
  payload_json JSONB NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS predictions (
  id BIGSERIAL PRIMARY KEY,
  observation_id BIGINT REFERENCES observations(id),
  risk_score DOUBLE PRECISION NOT NULL,
  risk_level TEXT NOT NULL,
  model_name TEXT NOT NULL,
  model_version TEXT,
  explanation TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS historical_cases (
  id BIGSERIAL PRIMARY KEY,
  source TEXT NOT NULL,
  case_text TEXT NOT NULL,
  outcome TEXT,
  latitude DOUBLE PRECISION,
  longitude DOUBLE PRECISION,
  geometry GEOGRAPHY(POINT,4326),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS alerts (
  id BIGSERIAL PRIMARY KEY,
  prediction_id BIGINT REFERENCES predictions(id),
  severity TEXT NOT NULL,
  status TEXT NOT NULL DEFAULT 'PENDING',
  message TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS model_runs (
  id BIGSERIAL PRIMARY KEY,
  model_name TEXT NOT NULL,
  metrics_json JSONB NOT NULL,
  dataset_source TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_observations_geometry ON observations USING GIST (geometry);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions (risk_level);
