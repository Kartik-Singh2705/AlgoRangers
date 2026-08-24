-- ============================================================
-- TERRAGUARD NER
-- REAL-TIME POSTGRESQL + POSTGIS SCHEMA
-- ============================================================

CREATE EXTENSION IF NOT EXISTS postgis;


-- ============================================================
-- LOCATIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS locations (

    id BIGSERIAL PRIMARY KEY,

    latitude DOUBLE PRECISION NOT NULL,

    longitude DOUBLE PRECISION NOT NULL,

    place_name TEXT,

    district TEXT,

    state TEXT,

    country TEXT DEFAULT 'India',

    geometry GEOGRAPHY(
        POINT,
        4326
    ),

    created_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE (
        latitude,
        longitude
    )
);


-- ============================================================
-- REAL-TIME OBSERVATIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS observations (

    id BIGSERIAL PRIMARY KEY,

    location_id BIGINT
        REFERENCES locations(id),

    observed_at TIMESTAMPTZ NOT NULL,

    source TEXT NOT NULL,

    rainfall_1h DOUBLE PRECISION,

    rainfall_3h DOUBLE PRECISION,

    rainfall_6h DOUBLE PRECISION,

    rainfall_24h DOUBLE PRECISION,

    rainfall_72h DOUBLE PRECISION,

    slope_angle DOUBLE PRECISION,

    soil_saturation DOUBLE PRECISION,

    soil_moisture DOUBLE PRECISION,

    vegetation_cover DOUBLE PRECISION,

    ndvi DOUBLE PRECISION,

    elevation DOUBLE PRECISION,

    earthquake_activity DOUBLE PRECISION,

    proximity_to_water DOUBLE PRECISION,

    distance_to_road_m DOUBLE PRECISION,

    prior_events_5y DOUBLE PRECISION,

    soil_type TEXT,

    raw_data JSONB,

    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- ML PREDICTIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS predictions (

    id BIGSERIAL PRIMARY KEY,

    observation_id BIGINT
        REFERENCES observations(id)
        ON DELETE CASCADE,

    risk_score DOUBLE PRECISION NOT NULL,

    risk_percentage DOUBLE PRECISION NOT NULL,

    risk_level TEXT NOT NULL,

    model_name TEXT,

    model_version TEXT,

    primary_cause TEXT,

    explanation TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- HISTORICAL LANDSLIDE EVENTS
-- ============================================================

CREATE TABLE IF NOT EXISTS historical_landslides (

    id BIGSERIAL PRIMARY KEY,

    event_date TIMESTAMPTZ,

    latitude DOUBLE PRECISION NOT NULL,

    longitude DOUBLE PRECISION NOT NULL,

    location_id BIGINT
        REFERENCES locations(id),

    severity TEXT,

    cause TEXT,

    rainfall_24h DOUBLE PRECISION,

    soil_saturation DOUBLE PRECISION,

    slope_angle DOUBLE PRECISION,

    description TEXT,

    geometry GEOGRAPHY(
        POINT,
        4326
    ),

    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- ALERT RECIPIENTS
-- ============================================================

CREATE TABLE IF NOT EXISTS alert_recipients (

    id BIGSERIAL PRIMARY KEY,

    name TEXT NOT NULL,

    phone TEXT,

    email TEXT,

    telegram_id TEXT,

    role TEXT,

    state TEXT,

    district TEXT,

    minimum_alert_level TEXT
        DEFAULT 'HIGH',

    active BOOLEAN DEFAULT TRUE,

    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- ALERTS
-- ============================================================

CREATE TABLE IF NOT EXISTS alerts (

    id BIGSERIAL PRIMARY KEY,

    prediction_id BIGINT
        REFERENCES predictions(id)
        ON DELETE CASCADE,

    recipient_id BIGINT
        REFERENCES alert_recipients(id),

    severity TEXT NOT NULL,

    channel TEXT,

    message TEXT,

    status TEXT DEFAULT 'PENDING',

    sent_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);


-- ============================================================
-- SPATIAL INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_locations_geometry
ON locations
USING GIST (geometry);


CREATE INDEX IF NOT EXISTS idx_historical_geometry
ON historical_landslides
USING GIST (geometry);


-- ============================================================
-- OBSERVATION INDEXES
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_observations_location
ON observations(location_id);


CREATE INDEX IF NOT EXISTS idx_observations_time
ON observations(observed_at);


CREATE INDEX IF NOT EXISTS idx_predictions_time
ON predictions(created_at);


CREATE INDEX IF NOT EXISTS idx_predictions_level
ON predictions(risk_level);


-- ============================================================
-- AUTOMATIC LOCATION GEOMETRY
-- ============================================================

CREATE OR REPLACE FUNCTION set_location_geometry()
RETURNS TRIGGER AS $$
BEGIN

    NEW.geometry =
        ST_SetSRID(
            ST_MakePoint(
                NEW.longitude,
                NEW.latitude
            ),
            4326
        )::geography;

    RETURN NEW;

END;
$$ LANGUAGE plpgsql;


DROP TRIGGER IF EXISTS
location_geometry_trigger
ON locations;


CREATE TRIGGER location_geometry_trigger

BEFORE INSERT OR UPDATE OF
latitude,
longitude

ON locations

FOR EACH ROW

EXECUTE FUNCTION set_location_geometry();


-- ============================================================
-- AUTOMATIC HISTORICAL EVENT GEOMETRY
-- ============================================================

CREATE OR REPLACE FUNCTION set_historical_geometry()
RETURNS TRIGGER AS $$
BEGIN

    NEW.geometry =
        ST_SetSRID(
            ST_MakePoint(
                NEW.longitude,
                NEW.latitude
            ),
            4326
        )::geography;

    RETURN NEW;

END;
$$ LANGUAGE plpgsql;


DROP TRIGGER IF EXISTS
historical_geometry_trigger
ON historical_landslides;


CREATE TRIGGER historical_geometry_trigger

BEFORE INSERT OR UPDATE OF
latitude,
longitude

ON historical_landslides

FOR EACH ROW

EXECUTE FUNCTION set_historical_geometry();