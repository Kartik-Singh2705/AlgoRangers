"""
TerraGuard NER
Live Monitoring Controller

Runs the real-time ML pipeline continuously.

Flow:

Simulator / Real Data
        ↓
Pipeline
        ↓
Prediction
        ↓
PostgreSQL
        ↓
Alert Engine
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

INTERVAL_SECONDS = 30


# ============================================================
# RUN ONE PIPELINE CYCLE
# ============================================================

def run_pipeline():

    print()
    print("=" * 70)

    print(
        "TERRAGUARD LIVE MONITOR"
    )

    print(
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    print("=" * 70)

    try:

        result = subprocess.run(

            [
                sys.executable,
                "-m",
                "realtime.pipeline",
            ],

            capture_output=True,

            text=True,

        )

        print(
            result.stdout
        )

        if result.stderr:

            print(
                result.stderr
            )

        if result.returncode == 0:

            print(
                "✓ Pipeline cycle completed."
            )

        else:

            print(
                "✗ Pipeline failed."
            )

    except Exception as error:

        print(
            f"✗ Monitor error: {error}"
        )


# ============================================================
# MAIN LOOP
# ============================================================

def main():

    print()
    print(
        "🌍 TerraGuard NER Live Monitoring Started"
    )

    print(
        f"Running every {INTERVAL_SECONDS} seconds."
    )

    print(
        "Press Ctrl+C to stop."
    )

    print()

    try:

        while True:

            run_pipeline()

            print()
            print(
                f"Next prediction in "
                f"{INTERVAL_SECONDS} seconds..."
            )

            time.sleep(
                INTERVAL_SECONDS
            )

    except KeyboardInterrupt:

        print()
        print(
            "TerraGuard monitoring stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()