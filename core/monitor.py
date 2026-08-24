"""
TerraGuard NER - Continuous Monitoring Engine

Continuously receives environmental observations,
runs the ML model, and produces risk predictions.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from realtime.simulator import (
    generate_validated_observation,
)

from realtime.pipeline import (
    predict_observation,
    print_prediction,
    load_model,
)
from core.realtime_db import (
    save_observation,
    save_prediction,
)


class TerraGuardMonitor:
    """
    Continuous TerraGuard monitoring engine.
    """

    def __init__(
        self,
        interval_seconds: int = 10,
        location: dict | None = None,
    ):
        self.interval_seconds = interval_seconds
        self.location = location

        self.running = False

        # Load the trained model only once.
        # We don't want to load the model from disk
        # every 10 seconds.
        self.artifacts = load_model()

        self.observation_count = 0

    # ========================================================
    # PROCESS ONE OBSERVATION
    # ========================================================

    def process_once(self) -> dict:
        """
        Generate and process one observation.
        """

        # ----------------------------------------------------
        # 1. Generate real-time observation
        # ----------------------------------------------------

        observation = generate_validated_observation(
            location=self.location
        )

        # ----------------------------------------------------
        # 2. Run ML prediction
        # ----------------------------------------------------

        result = predict_observation(
            observation,
            artifacts=self.artifacts,
        )
        # --------------------------------------------------------
# Save observation to PostgreSQL
# --------------------------------------------------------

        observation_id = save_observation(
            observation
)

# --------------------------------------------------------
# Save prediction to PostgreSQL
# --------------------------------------------------------

        prediction_id = save_prediction(
            observation_id,
            result
)

        result["observation_id"] = observation_id
        result["prediction_id"] = prediction_id

        # ----------------------------------------------------
        # 3. Count observation
        # ----------------------------------------------------

        self.observation_count += 1

        # ----------------------------------------------------
        # 4. Add monitoring metadata
        # ----------------------------------------------------

        result["observation_number"] = (
            self.observation_count
        )

        result["monitoring_timestamp"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        return result

    # ========================================================
    # CONTINUOUS LOOP
    # ========================================================

    def start(self):
        """
        Start continuous monitoring.
        """

        self.running = True

        print()
        print("=" * 70)
        print("TERRAGUARD NER - CONTINUOUS MONITORING")
        print("=" * 70)

        print(
            f"Monitoring interval: "
            f"{self.interval_seconds} seconds"
        )

        print(
            "Press Ctrl+C to stop monitoring."
        )

        print("=" * 70)

        try:

            while self.running:

                try:

                    # ----------------------------------------
                    # Process one observation
                    # ----------------------------------------

                    result = self.process_once()

                    # ----------------------------------------
                    # Display result
                    # ----------------------------------------

                    print_prediction(
                        result
                    )

                    # ----------------------------------------
                    # Monitoring information
                    # ----------------------------------------

                    print(
                        f"Observation #: "
                        f"{result['observation_number']}"
                    )

                    print(
                        f"Next observation "
                        f"in {self.interval_seconds} seconds..."
                    )

                    # ----------------------------------------
                    # Wait
                    # ----------------------------------------

                    time.sleep(
                        self.interval_seconds
                    )

                except Exception as error:

                    print()
                    print("⚠️ Monitoring error:")
                    print(error)

                    print(
                        "The monitoring system will "
                        "continue running."
                    )

                    time.sleep(
                        self.interval_seconds
                    )

        except KeyboardInterrupt:

            print()
            print("=" * 70)
            print("TERRAGUARD MONITORING STOPPED")
            print("=" * 70)

            print(
                f"Total observations processed: "
                f"{self.observation_count}"
            )

            self.running = False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # Import the demo locations.
    from realtime.simulator import (
        SIMULATION_LOCATIONS,
    )

    # --------------------------------------------------------
    # Use a fixed location for the SIH demonstration.
    # --------------------------------------------------------
    #
    # This is intentional.
    #
    # We want to watch the risk at ONE location change
    # over time as simulated environmental conditions change.
    #
    # Later, the real system will monitor many locations.
    # --------------------------------------------------------

    demo_location = SIMULATION_LOCATIONS[0]

    monitor = TerraGuardMonitor(
        interval_seconds=10,
        location=demo_location,
    )

    monitor.start()