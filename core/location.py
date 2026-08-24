"""
Location intelligence for TerraGuard NER.

Converts latitude/longitude into human-readable
location information.

The first version uses a local NER location database
so the system does not depend on an external API.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2
from typing import Optional


@dataclass
class LocationInfo:
    latitude: float
    longitude: float
    place_name: str
    district: str
    state: str
    country: str
    distance_km: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "place_name": self.place_name,
            "district": self.district,
            "state": self.state,
            "country": self.country,
            "distance_km": self.distance_km,
        }


# ------------------------------------------------------------------
# Initial NER reference locations
# ------------------------------------------------------------------
#
# These are reference points for the prototype.
# Later we will replace/extend this with a proper GIS boundary
# dataset and reverse-geocoding service.
#
# Coordinates are approximate reference coordinates, NOT exact
# administrative boundaries.
# ------------------------------------------------------------------

NER_LOCATIONS = [
    {
        "place_name": "Dhemaji",
        "district": "Dhemaji",
        "state": "Assam",
        "country": "India",
        "latitude": 27.4728,
        "longitude": 94.9120,
    },
    {
        "place_name": "Guwahati",
        "district": "Kamrup Metropolitan",
        "state": "Assam",
        "country": "India",
        "latitude": 26.1445,
        "longitude": 91.7362,
    },
    {
        "place_name": "Itanagar",
        "district": "Papum Pare",
        "state": "Arunachal Pradesh",
        "country": "India",
        "latitude": 27.0844,
        "longitude": 93.6053,
    },
    {
        "place_name": "Gangtok",
        "district": "Gangtok",
        "state": "Sikkim",
        "country": "India",
        "latitude": 27.3389,
        "longitude": 88.6065,
    },
    {
        "place_name": "Shillong",
        "district": "East Khasi Hills",
        "state": "Meghalaya",
        "country": "India",
        "latitude": 25.5788,
        "longitude": 91.8933,
    },
    {
        "place_name": "Aizawl",
        "district": "Aizawl",
        "state": "Mizoram",
        "country": "India",
        "latitude": 23.7271,
        "longitude": 92.7176,
    },
    {
        "place_name": "Kohima",
        "district": "Kohima",
        "state": "Nagaland",
        "country": "India",
        "latitude": 25.6751,
        "longitude": 94.1086,
    },
    {
        "place_name": "Imphal",
        "district": "Imphal West",
        "state": "Manipur",
        "country": "India",
        "latitude": 24.8170,
        "longitude": 93.9368,
    },
    {
        "place_name": "Agartala",
        "district": "West Tripura",
        "state": "Tripura",
        "country": "India",
        "latitude": 23.8315,
        "longitude": 91.2868,
    },
]


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate approximate distance between two coordinates
    using the Haversine formula.

    Returns:
        Distance in kilometres.
    """

    earth_radius_km = 6371.0

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    delta_lat = radians(lat2 - lat1)
    delta_lon = radians(lon2 - lon1)

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad)
        * cos(lat2_rad)
        * sin(delta_lon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return earth_radius_km * c


def find_nearest_location(
    latitude: float,
    longitude: float,
) -> LocationInfo:
    """
    Find the nearest reference location in the NER database.

    This is a prototype resolver. It does NOT claim that the
    nearest city is the exact administrative location.
    """

    latitude = float(latitude)
    longitude = float(longitude)

    if not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")

    if not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")

    nearest = None
    nearest_distance = float("inf")

    for location in NER_LOCATIONS:

        distance = haversine_distance(
            latitude,
            longitude,
            location["latitude"],
            location["longitude"],
        )

        if distance < nearest_distance:
            nearest_distance = distance
            nearest = location

    if nearest is None:
        raise RuntimeError("No reference locations available.")

    return LocationInfo(
        latitude=latitude,
        longitude=longitude,
        place_name=nearest["place_name"],
        district=nearest["district"],
        state=nearest["state"],
        country=nearest["country"],
        distance_km=round(nearest_distance, 2),
    )


def resolve_location(
    latitude: float,
    longitude: float,
) -> dict:
    """
    Public location resolver used by the rest of TerraGuard.
    """

    location = find_nearest_location(
        latitude,
        longitude,
    )

    return location.to_dict()