"""
TerraGuard NER - Alert Engine

Handles:
1. Risk threshold
2. Alert cooldown
3. Telegram notification
4. PostgreSQL alert history
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from core.realtime_db import get_connection


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "",
)

TELEGRAM_CHAT_IDS = os.getenv(
    "TELEGRAM_CHAT_IDS",
    "",
)

ALERT_MIN_RISK = float(
    os.getenv(
        "ALERT_MIN_RISK",
        "70",
    )
)

ALERT_COOLDOWN_MINUTES = float(
    os.getenv(
        "ALERT_COOLDOWN_MINUTES",
        "30",
    )
)


# ============================================================
# COOLDOWN MEMORY
# ============================================================

_last_alert_time = {}


# ============================================================
# RISK LEVEL
# ============================================================

def get_alert_level(
    risk_percentage: float,
) -> str:

    risk = float(risk_percentage)

    if risk >= 85:
        return "CRITICAL"

    if risk >= 70:
        return "HIGH"

    if risk >= 50:
        return "WARNING"

    return "NORMAL"


# ============================================================
# COOLDOWN
# ============================================================

def should_send_alert(
    risk_percentage: float,
    location_key: str,
) -> bool:

    risk = float(risk_percentage)

    if risk < ALERT_MIN_RISK:
        return False

    now = time.time()

    previous = _last_alert_time.get(
        location_key
    )

    if previous is not None:

        elapsed_minutes = (
            now - previous
        ) / 60

        if (
            elapsed_minutes
            < ALERT_COOLDOWN_MINUTES
        ):

            return False

    _last_alert_time[
        location_key
    ] = now

    return True


# ============================================================
# CREATE MESSAGE
# ============================================================

def create_alert_message(
    result: dict,
) -> str:

    risk = float(
        result.get(
            "risk_percentage",
            0,
        )
    )

    level = get_alert_level(
        risk
    )

    place = result.get(
        "place_name",
        "Unknown",
    )

    district = result.get(
        "district",
        "Unknown",
    )

    state = result.get(
        "state",
        "Unknown",
    )

    latitude = result.get(
        "latitude",
        "Unknown",
    )

    longitude = result.get(
        "longitude",
        "Unknown",
    )

    cause = result.get(
        "primary_cause",
        "Unknown",
    )

    events = result.get(
        "historical_events_found",
        0,
    )

    nearest = result.get(
        "nearest_historical_event_km"
    )

    explanation = result.get(
        "explanation",
        "",
    )

    message = f"""
🚨 TERRAGUARD NER ALERT 🚨

Risk Level: {level}
Risk Score: {risk:.2f}%

📍 LOCATION
Place: {place}
District: {district}
State: {state}

🌐 COORDINATES
Latitude: {latitude}
Longitude: {longitude}

🔎 MAIN RISK FACTOR
{cause}

📚 HISTORICAL EVIDENCE
Similar events: {events}
"""

    if nearest is not None:

        try:

            message += (
                f"Nearest historical event: "
                f"{float(nearest):.2f} km\n"
            )

        except (
            TypeError,
            ValueError,
        ):
            pass

    if explanation:

        message += (
            f"\n🧠 ASSESSMENT\n"
            f"{explanation}\n"
        )

    message += (
        "\n⚠️ AI-based early-warning assessment. "
        "Verify with authorized "
        "disaster-management personnel."
    )

    return message.strip()


# ============================================================
# SAVE ALERT TO DATABASE
# ============================================================

def save_alert_to_database(
    result: dict,
    recipients: str,
    channel: str,
    delivery_status: str,
    message: str,
) -> bool:
    """
    Save an alert into the existing alerts table.
    """

    query = """
        INSERT INTO alerts
        (
            latitude,
            longitude,
            place_name,
            district,
            state,
            risk_score,
            risk_level,
            primary_cause,
            recipients,
            channel,
            delivery_status,
            message
        )
        VALUES
        (
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s,
            %s
        )
    """

    try:

        with get_connection() as conn:

            with conn.cursor() as cursor:

                cursor.execute(
                    query,
                    (
                        result.get(
                            "latitude"
                        ),

                        result.get(
                            "longitude"
                        ),

                        result.get(
                            "place_name",
                            "Unknown",
                        ),

                        result.get(
                            "district",
                            "Unknown",
                        ),

                        result.get(
                            "state",
                            "Unknown",
                        ),

                        float(
                            result.get(
                                "risk_score",
                                0,
                            )
                        ),

                        result.get(
                            "risk_level",
                            "UNKNOWN",
                        ),

                        result.get(
                            "primary_cause",
                            "Unknown",
                        ),

                        recipients,

                        channel,

                        delivery_status,

                        message,
                    ),
                )

            conn.commit()

        print(
            "Alert saved to PostgreSQL."
        )

        return True

    except Exception as error:

        print(
            "WARNING: Could not save alert "
            f"to PostgreSQL: {error}"
        )

        return False


# ============================================================
# SEND TELEGRAM
# ============================================================

def send_telegram(
    message: str,
) -> tuple[bool, str]:
    """
    Send message to all configured Telegram recipients.

    Returns:
        (success, recipient_string)
    """

    if not TELEGRAM_BOT_TOKEN:

        print(
            "Telegram not configured."
        )

        return False, ""

    if not TELEGRAM_CHAT_IDS:

        print(
            "No Telegram recipients configured."
        )

        return False, ""

    url = (
        "https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}"
        "/sendMessage"
    )

    chat_ids = [
        chat_id.strip()
        for chat_id
        in TELEGRAM_CHAT_IDS.split(",")
        if chat_id.strip()
    ]

    successful = []
    failed = []

    for chat_id in chat_ids:

        try:

            response = requests.post(
                url,
                data={
                    "chat_id": chat_id,
                    "text": message,
                },
                timeout=10,
            )

            response.raise_for_status()

            successful.append(
                chat_id
            )

            print(
                f"Alert sent to Telegram "
                f"chat {chat_id}"
            )

        except requests.RequestException as error:

            failed.append(
                chat_id
            )

            print(
                f"Telegram failed for "
                f"{chat_id}: {error}"
            )

    recipients = ",".join(
        chat_ids
    )

    return (
        len(successful) > 0,
        recipients,
    )


# ============================================================
# PROCESS ALERT
# ============================================================

def process_alert(
    result: dict,
) -> bool:

    risk = float(
        result.get(
            "risk_percentage",
            0,
        )
    )

    level = get_alert_level(
        risk
    )

    location_key = (
        f"{result.get('latitude')}:"
        f"{result.get('longitude')}"
    )

    print()
    print(
        f"Alert Engine: "
        f"{level} "
        f"({risk:.2f}%)"
    )

    # --------------------------------------------------------
    # NO ALERT
    # --------------------------------------------------------

    if not should_send_alert(
        risk,
        location_key,
    ):

        print(
            "No alert sent."
        )

        return False

    # --------------------------------------------------------
    # CREATE MESSAGE
    # --------------------------------------------------------

    message = create_alert_message(
        result
    )

    print()
    print(
        "=" * 70
    )

    print(
        "ALERT MESSAGE"
    )

    print(
        "=" * 70
    )

    print(
        message
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # SEND TELEGRAM
    # --------------------------------------------------------

    telegram_success, recipients = (
        send_telegram(
            message
        )
    )

    # --------------------------------------------------------
    # DELIVERY STATUS
    # --------------------------------------------------------

    if telegram_success:

        delivery_status = "SENT"

        channel = "telegram"

    else:

        delivery_status = "FAILED"

        channel = "telegram"

    # --------------------------------------------------------
    # SAVE TO DATABASE
    # --------------------------------------------------------

    save_alert_to_database(
        result=result,
        recipients=recipients,
        channel=channel,
        delivery_status=delivery_status,
        message=message,
    )

    return telegram_success


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_result = {

        "risk_score": 0.915,

        "risk_percentage": 91.5,

        "risk_level": "CRITICAL",

        "place_name":
            "Dhemaji",

        "district":
            "Dhemaji",

        "state":
            "Assam",

        "latitude":
            27.4728,

        "longitude":
            94.9120,

        "primary_cause":
            "24-hour rainfall",

        "historical_events_found":
            4,

        "nearest_historical_event_km":
            2.84,

        "explanation":
            "Heavy rainfall and high soil "
            "saturation are strongly influencing "
            "the predicted risk.",
    }

    process_alert(
        test_result
    )