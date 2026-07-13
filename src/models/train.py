"""Training and MLflow experiment tracking for the fraud-detection model.

Planned implementation phase: Phase 2 — Training and Experiment Tracking.
Current status: provides load_split() (feature loader), build_model()
(unfitted-classifier factory), and compute_metrics() (the fraud metrics). The
MLflow-logging train() entrypoint is implemented in a later step of Phase 2.

Quality standard (as for every production module here):
    - Strict typing (mypy --strict as reference; avoid unjustified `Any`).
    - Structured logging (never `print()`).
    - Custom application exception hierarchy (never bare `except:`).
    - Pytest coverage arrives with the model tests (Phase 2, Step 12).
"""

import logging
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd
import pyarrow as pa
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

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


def build_model(params: dict[str, Any]) -> ClassifierMixin:
    """Build an unfitted classifier from a parameter dictionary.

    Args:
        params: must contain a "model" key naming the algorithm. Only
            "logistic_regression" is supported in this interaction; the
            "xgboost" branch is added in Step 7 without modifying this
            function's existing branches.

    Returns:
        An unfitted, ready-to-fit classifier instance.

    Raises:
        ModelTrainingError: if params["model"] names an unsupported
            algorithm, or a required hyperparameter is missing.
    """
    name = params.get("model")
    if name == "logistic_regression":
        try:
            return LogisticRegression(
                class_weight="balanced",
                max_iter=params["max_iter"],
                random_state=params["random_state"],
            )
        except KeyError as exc:
            raise ModelTrainingError(
                f"Missing required hyperparameter for logistic_regression: {exc}"
            ) from exc

    raise ModelTrainingError(f"Unsupported model: {name!r}")


class MetricsReport(TypedDict):
    """Shape of the metrics dictionary produced by compute_metrics()."""

    precision: float
    recall: float
    f1: float
    pr_auc: float


def compute_metrics(
    y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
) -> MetricsReport:
    """Compute the metrics that matter for a severely imbalanced problem.

    Accuracy is deliberately absent: with fraud at a fraction of a percent,
    a model that always predicts "not fraud" would score above 99% accuracy
    while being entirely useless.

    Args:
        y_true: ground-truth labels.
        y_pred: binarized predictions at whatever threshold the caller chose.
        y_proba: predicted probability of the positive (fraud) class — used
            only by pr_auc, which measures ranking quality independently of
            any single threshold.

    Returns:
        precision, recall, f1, and pr_auc for this set of predictions.

    Raises:
        ModelTrainingError: if scikit-learn cannot compute a metric because
            y_true, y_pred, or y_proba are inconsistent with each other.
    """
    try:
        report: MetricsReport = {
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "pr_auc": average_precision_score(y_true, y_proba),
        }
    except ValueError as exc:
        raise ModelTrainingError(f"Could not compute metrics: {exc}") from exc

    logger.info(
        "Metrics computed: precision=%.4f recall=%.4f f1=%.4f pr_auc=%.4f",
        report["precision"],
        report["recall"],
        report["f1"],
        report["pr_auc"],
    )
    return report
