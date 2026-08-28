"""Prefect flow orchestrating training: validate -> preprocess -> train -> register.

This module adds orchestration, not logic. Every task is a one-line call into a
function that already existed and is already covered by its own tests; nothing
from src/ is reimplemented, reconfigured or wrapped in new behaviour. That is
only possible because those four entry points have been zero-argument and
side-effect-complete since Phase 1 (see docs/decisions/0031-prefect-dependency.md
for the layering rationale).

DVC and Prefect solve different problems and both stay: dvc.yaml answers "what
ran and with what result" (deterministic, cached, content-addressed), while this
flow answers "when it ran and how reliably" (scheduling, retries, observability).
Calling the Python functions directly rather than shelling out to `dvc repro`
buys per-stage visibility in the dashboard and gives up DVC's cache -- this flow
re-runs every stage whether or not its inputs changed.
"""

from prefect import flow, get_run_logger, task

from src.data.preprocess import preprocess as run_preprocess
from src.data.validate import ValidationReport, validate_raw_data
from src.models.register import main as run_register
from src.models.train import train as run_train


# Retry budgets are calibrated per task rather than set globally, because the
# cost of an attempt and the likelihood that a retry helps differ sharply across
# these four stages. See docs/decisions/0033-orchestration-design.md.
#
# Validation is cheap (one CSV read plus a Pandera contract) and its realistic
# failure modes are transient I/O, so tolerance is high and nearly free.
@task(retries=3, retry_delay_seconds=10, name="Validate data")
def validate_task() -> ValidationReport:
    """Validate the raw dataset against its Pandera contract.

    Returns:
        The validation report produced by src.data.validate, so the flow can
        surface row count and fraud rate in the dashboard logs.
    """
    logger = get_run_logger()
    report = validate_raw_data()
    logger.info(
        "Validation OK: %s rows, fraud rate %.4f",
        report["n_rows"],
        report["fraud_rate"],
    )
    return report


# Preprocessing writes the parquet splits and the fitted preprocessor. Data
# faults were already caught by validate_task, so a failure here is far more
# likely deterministic than transient: a lower budget avoids repeating work that
# will fail identically.
@task(retries=2, retry_delay_seconds=10, name="Preprocess data")
def preprocess_task() -> None:
    """Split, scale and persist the processed train/test data."""
    logger = get_run_logger()
    run_preprocess()
    logger.info("Preprocessing completed")


# Training is the expensive stage (cross-validation plus a threshold sweep,
# minutes rather than seconds). Same low retry count, but a longer delay: if the
# cause is MLflow being briefly unreachable, 30s gives it room to recover, and
# the wait is negligible next to the cost of the attempt itself.
@task(retries=2, retry_delay_seconds=30, name="Train model")
def train_task() -> None:
    """Train the model and log the run to MLflow."""
    logger = get_run_logger()
    run_train()
    logger.info("Training completed")


# Registration is network-bound to the MLflow Model Registry -- genuinely
# transient-prone -- but each attempt is cheap, so the short delay applies here
# rather than the long one.
@task(retries=2, retry_delay_seconds=10, name="Register and promote")
def register_task() -> None:
    """Register the best run and promote it if it beats @production."""
    logger = get_run_logger()
    run_register()
    logger.info("Registration and promotion completed")


@flow(name="training-pipeline", log_prints=True)
def training_pipeline() -> None:
    """Run the full training pipeline as an orchestrated sequence.

    Tasks are called in order rather than submitted concurrently: each stage
    consumes what the previous one produced. Validation is deliberately first
    and acts as the gate -- if the raw data violates its contract the flow stops
    before anything trains on it.

    No task catches the exceptions raised by the functions it wraps. Prefect
    must see the original src/exceptions.py error (DataValidationError,
    ModelTrainingError, ...) so its retry machinery acts on the real cause and
    the dashboard reports it verbatim.
    """
    logger = get_run_logger()
    logger.info("Starting the training pipeline")
    validate_task()
    preprocess_task()
    train_task()
    register_task()
    logger.info("Training pipeline finished")


if __name__ == "__main__":
    training_pipeline()
