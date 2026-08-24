"""
TerraGuard NER - Historical Evidence Engine

Finds historical landslide events similar to the
current real-time observation using:

1. Geographic distance
2. Rainfall similarity
3. Soil saturation similarity
4. Slope similarity
"""

from __future__ import annotations

import psycopg


# ============================================================
# DATABASE
# ============================================================

DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "terraguard"
DB_USER = "terraguard"
DB_PASSWORD = "terraguard_dev"


def get_connection():
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


# ============================================================
# FIND HISTORICAL EVENTS
# ============================================================

def find_similar_events(
    observation: dict,
    limit: int = 5,
):
    """
    Find historical landslide events close to the
    current location and environmentally similar.
    """

    latitude = float(
        observation["latitude"]
    )

    longitude = float(
        observation["longitude"]
    )

    rainfall = float(
        observation.get(
            "rainfall_24h",
            0,
        )
    )

    saturation = float(
        observation.get(
            "soil_saturation",
            0,
        )
    )

    slope = float(
        observation.get(
            "slope_angle",
            0,
        )
    )

    query = """
        SELECT

            id,
            event_date,
            latitude,
            longitude,
            severity,
            cause,
            rainfall_24h,
            soil_saturation,
            slope_angle,
            description,

            ST_Distance(
                geometry,
                ST_SetSRID(
                    ST_MakePoint(
                        %s,
                        %s
                    ),
                    4326
                )::geography
            ) / 1000.0 AS distance_km

        FROM historical_landslides

        ORDER BY

            (
                ST_Distance(
                    geometry,
                    ST_SetSRID(
                        ST_MakePoint(
                            %s,
                            %s
                        ),
                        4326
                    )::geography
                ) / 1000.0
            )

            +

            ABS(
                COALESCE(
                    rainfall_24h,
                    0
                ) - %s
            ) * 0.02

            +

            ABS(
                COALESCE(
                    soil_saturation,
                    0
                ) - %s
            ) * 0.1

            +

            ABS(
                COALESCE(
                    slope_angle,
                    0
                ) - %s
            )

        LIMIT %s;
    """

    with get_connection() as conn:

        with conn.cursor() as cur:

            cur.execute(
                query,
                (
                    longitude,
                    latitude,

                    longitude,
                    latitude,

                    rainfall,
                    saturation,
                    slope,

                    limit,
                ),
            )

            rows = cur.fetchall()

            columns = [
                description.name
                for description in cur.description
            ]

    return [
        dict(
            zip(
                columns,
                row,
            )
        )
        for row in rows
    ]


# ============================================================
# BUILD EVIDENCE SUMMARY
# ============================================================

def build_evidence_summary(
    observation: dict,
    events: list,
) -> dict:
    """
    Convert historical events into a concise
    evidence summary.
    """

    if not events:

        return {
            "historical_events_found": 0,

            "nearest_event_km": None,

            "historical_causes": [],

            "evidence_text":
                "No similar historical landslide "
                "events were found in the database.",
        }

    causes = []

    for event in events:

        cause = event.get(
            "cause"
        )

        if cause and cause not in causes:

            causes.append(
                cause
            )

    nearest_distance = events[0].get(
        "distance_km"
    )

    if nearest_distance is not None:

        nearest_distance = round(
            float(
                nearest_distance
            ),
            2,
        )

    if causes:

        cause_text = ", ".join(
            causes[:3]
        )

        evidence_text = (
            f"{len(events)} historical "
            f"landslide event(s) were found "
            f"near or environmentally similar "
            f"to the current observation. "
            f"Reported historical causes include "
            f"{cause_text}."
        )

    else:

        evidence_text = (
            f"{len(events)} historical "
            f"landslide event(s) were found, "
            f"but their causes are not recorded."
        )

    return {
        "historical_events_found": len(
            events
        ),

        "nearest_event_km":
            nearest_distance,

        "historical_causes":
            causes,

        "evidence_text":
            evidence_text,
    }


# ============================================================
# COMPLETE EVIDENCE SEARCH
# ============================================================

def get_historical_evidence(
    observation: dict,
    limit: int = 5,
) -> dict:
    """
    Complete historical evidence retrieval.
    """

    events = find_similar_events(
        observation,
        limit=limit,
    )

    summary = build_evidence_summary(
        observation,
        events,
    )

    summary["events"] = events

    return summary