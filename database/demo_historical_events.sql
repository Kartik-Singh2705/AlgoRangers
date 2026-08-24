INSERT INTO historical_landslides
(
    event_date,
    latitude,
    longitude,
    severity,
    cause,
    rainfall_24h,
    soil_saturation,
    slope_angle,
    description
)
VALUES
(
    '2025-07-15 10:30:00+05:30',
    27.4728,
    94.9120,
    'HIGH',
    'Heavy rainfall and saturated soil',
    185.0,
    89.0,
    42.0,
    'Rainfall-triggered slope failure.'
),
(
    '2024-08-21 14:20:00+05:30',
    27.4900,
    94.9300,
    'CRITICAL',
    'Extreme rainfall and steep slope',
    245.0,
    94.0,
    48.0,
    'Major rainfall-induced landslide.'
),
(
    '2023-06-12 09:10:00+05:30',
    27.4500,
    94.8900,
    'MODERATE',
    'Heavy rainfall',
    130.0,
    78.0,
    35.0,
    'Localized slope instability after rainfall.'
),
(
    '2025-09-03 16:45:00+05:30',
    27.5200,
    94.9500,
    'HIGH',
    'Rainfall and soil saturation',
    210.0,
    91.0,
    44.0,
    'Slope failure following prolonged rainfall.'
);
