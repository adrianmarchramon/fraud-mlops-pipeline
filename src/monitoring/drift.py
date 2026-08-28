"""Data drift and concept drift detection with Evidently.

Implementation phase: Phase 8 - Monitoring and Drift.
Current status: minimal placeholder (Phase 7 / Step 3). detect_drift() returns a
fixed False so pipelines/monitoring_pipeline.py has a real, orchestrable
dependency to wrap and the closed loop can be wired and verified end to end
before the detection itself exists.

Phase 8 replaces the body of detect_drift() with real Evidently-based detection
comparing recent traffic against the training distribution, reading the
prediction records the API has been appending to config.PREDICTIONS_LOG since
Phase 4 (the contract frozen by docs/decisions/0021-prediction-log-and-api-tests.md).
The signature is expected to stay stable, so nothing in pipelines/ should need
to change when that lands.
"""

import logging

logger = logging.getLogger(__name__)


def detect_drift() -> bool:
    """Report whether recent data has drifted from the training distribution.

    Phase 7 placeholder: always returns False, so the monitoring flow takes its
    "no drift" branch and never triggers a retrain. Phase 8 replaces this body
    with the real Evidently comparison.

    Returns:
        False, unconditionally, until Phase 8 implements real detection.
    """
    logger.info(
        "detect_drift() placeholder invoked (Phase 7): always returns False, "
        "real Evidently detection arrives in Phase 8"
    )
    return False
