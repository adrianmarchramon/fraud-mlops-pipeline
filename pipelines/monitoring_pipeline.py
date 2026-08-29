"""Prefect flow that checks for drift on a schedule and triggers retraining.

This is the project's closed loop, and as of Phase 8 every link in it is real.
Phase 7 built the wiring around a placeholder predicate pinned to False; Phase
8 replaced that predicate with an Evidently comparison and added the alert, so
a scheduled run now measures drift against a versioned reference, announces a
detection, and reaches a live retraining trigger with no human in the path.

The wiring itself is unchanged from the Phase 7 version verified live -- same
task, same retry budget, same fire-and-forget trigger. Phase 8 added exactly
one line to the drift branch.

The detect_drift import sits at module level, unlike the reference material's
function-level import. There is no circular-import risk -- src/monitoring/
imports nothing from pipelines/ -- and module-level imports are the convention
everywhere else in this repository.
"""

from prefect import flow, get_run_logger, task
from prefect.deployments import run_deployment

from src.monitoring.drift import detect_drift, send_alert

# The deployment pipelines/serve.py registers for the training flow. This string
# is a real coupling between two files: "training-pipeline" is the @flow name in
# pipelines/training_pipeline.py and "on-demand" is the to_deployment() name in
# pipelines/serve.py. A typo in either place breaks the closed loop silently --
# nothing errors, retraining simply never fires -- so tests/test_pipelines.py
# pins it against both sources rather than trusting the literal.
TRAINING_DEPLOYMENT = "training-pipeline/on-demand"


@task(retries=2, retry_delay_seconds=30, name="Check drift")
def check_drift_task() -> bool:
    """Evaluate whether recent data has drifted.

    Returns:
        True if drift was detected and retraining should be triggered.
    """
    logger = get_run_logger()
    drift_detected = detect_drift()
    logger.info("Drift detected? %s", drift_detected)
    return drift_detected


@flow(name="monitoring-pipeline", log_prints=True)
def monitoring_pipeline() -> None:
    """Check for drift and trigger retraining if it is found.

    timeout=0 makes the trigger fire-and-forget: this flow submits the retrain
    and returns instead of blocking for the minutes training takes. A monitoring
    run that waited would hold its scheduled slot open and could still be
    running when the next cron tick arrives.
    """
    logger = get_run_logger()
    logger.info("Starting monitoring check")

    if check_drift_task():
        logger.warning("Drift detected: triggering retraining")
        # Announce before triggering, and through a different sink: this
        # logger writes to the Prefect dashboard, send_alert() writes to the
        # module log and, when configured, to a webhook someone actually reads.
        send_alert("Significant drift detected. Triggering retraining.")
        run_deployment(name=TRAINING_DEPLOYMENT, timeout=0)
    else:
        logger.info("No significant drift. Retraining skipped.")


if __name__ == "__main__":
    monitoring_pipeline()
