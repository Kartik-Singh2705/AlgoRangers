CREATE TABLE IF NOT EXISTS alerts
(
    id BIGSERIAL PRIMARY KEY,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,

    place_name TEXT,
    district TEXT,
    state TEXT,

    risk_score DOUBLE PRECISION,
    risk_level TEXT,

    primary_cause TEXT,

    recipients TEXT,

    channel TEXT,

    delivery_status TEXT,

    message TEXT
);
