"""Training and MLflow experiment tracking for the fraud-detection model.

Planned implementation phase: Phase 2 — Training and Experiment Tracking.
Current status: provides load_split(), the reusable feature loader that feeds
all training code. The rest of the module — build_model(), the metrics, and the
MLflow-logging train() entrypoint — is implemented in later steps of Phase 2.

Quality standard (as for every production module here):
    - Strict typing (mypy --strict as reference; avoid unjustified `Any`).
    - Structured logging (never `print()`).
    - Custom application exception hierarchy (never bare `except:`).
    - Pytest coverage arrives with the model tests (Phase 2, Step 12).
"""

import logging
from pathlib import Path

import pandas as pd
import pyarrow as pa

from src.config import TARGET
from src.exceptions import ModelTrainingError

logger = logging.getLogger(__name__)


def load_split(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a processed split and separate features (X) from target (y).

    No preprocessing happens here: the parquet arrives clean, validated, and
    scaled from the Phase 1 pipeline, with the train/test split already made.
    This function only reads it and splits off the target column.

    Args:
        path: Path to a processed split (e.g. TRAIN_PATH or TEST_PATH).

    Returns:
        A (X, y) tuple: the feature DataFrame and the target Series.

    Raises:
        ModelTrainingError: if the file is missing, cannot be read as parquet,
            or does not contain the target column.
    """
    logger.info("Loading split from %s", path)
    try:
        df = pd.read_parquet(path)
    except FileNotFoundError as exc:
        logger.error("Processed split not found at %s", path)
        raise ModelTrainingError(f"Processed split not found at {path}") from exc
    except pa.ArrowInvalid as exc:
        logger.error("Processed split at %s could not be read as parquet", path)
        raise ModelTrainingError(f"Could not read {path} as parquet") from exc

    if TARGET not in df.columns:
        logger.error("Target column %r missing from split at %s", TARGET, path)
        raise ModelTrainingError(f"Target column {TARGET!r} not found in {path}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    logger.info(
        "Loaded split from %s: %d rows, %d features, fraud rate %.6f",
        path,
        len(X),
        X.shape[1],
        float(y.mean()),
    )
    return X, y
