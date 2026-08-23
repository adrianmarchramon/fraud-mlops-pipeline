"""Inference logic: production model loading and prediction.

This module is the API's only point of contact with the Model Registry, and
it knows nothing about HTTP: no FastAPI import, no status codes, no request
objects. It resolves an alias, loads an artifact, and scores a transaction,
raising PredictionError when it cannot. Translating that into a response is
src/api/main.py's job.

Nothing here reimplements preprocessing or the decision threshold. Both live
inside the packaged artifact registered in Phase 3
(docs/decisions/0015-packaged-model-contract.md), which is precisely what
lets this module stay this small.
"""

import logging
from typing import NamedTuple

import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME
from src.exceptions import ModelRegistrationError, PredictionError
from src.models.register import PRODUCTION_ALIAS, load_production_model

logger = logging.getLogger(__name__)

# The columns FraudModel.predict() returns. Named here so a change to the
# packaged contract breaks in one place instead of surfacing as a KeyError
# deep inside a request.
PROBABILITY_COLUMN = "fraud_probability"
DECISION_COLUMN = "is_fraud"


class Prediction(NamedTuple):
    """One scored transaction, as the packaged model reports it."""

    fraud_probability: float
    is_fraud: int


def resolve_production_version(model_name: str = MODEL_NAME) -> str:
    """Report which concrete version the @production alias currently names.

    The API never selects a version, but it must be able to say which one it
    is serving: that is what makes a promotion auditable after the fact, and
    what /model-info exists to expose.

    Args:
        model_name: registered model to inspect; defaults to the project's.

    Returns:
        The version number, as the string MLflow stores it.

    Raises:
        PredictionError: if no version holds the alias, or the Registry is
            unreachable.
    """
    # Built here rather than imported from register.py, whose equivalent
    # helper is private: the URI is the shared contract, not the client.
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    try:
        version = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except MlflowException as exc:
        raise PredictionError(
            f"Could not resolve @{PRODUCTION_ALIAS} for {model_name!r}"
        ) from exc

    return str(version.version)


def load_production(model_name: str = MODEL_NAME) -> tuple[PyFuncModel, str]:
    """Load the @production model and report the version it resolved to.

    Delegates the load itself to register.py's load_production_model() rather
    than rebuilding the models:/ URI here — the URI stays written once, in the
    module that owns the Registry.

    Args:
        model_name: registered model to load; defaults to the project's.

    Returns:
        The loaded artifact and its version number.

    Raises:
        PredictionError: if the alias cannot be resolved or the load fails.
    """
    try:
        model = load_production_model(model_name)
    except ModelRegistrationError as exc:
        raise PredictionError(f"Could not load the production {model_name!r}") from exc

    version = resolve_production_version(model_name)
    logger.info("Serving %s version %s", model_name, version)
    return model, version


def predict_transaction(model: PyFuncModel, features: dict[str, float]) -> Prediction:
    """Score one raw transaction.

    The features go in exactly as received: no scaling, no reordering, no
    threshold applied here. MLflow matches the registered signature by column
    name, and the artifact does the rest.

    Args:
        model: a loaded production artifact.
        features: one raw transaction — the columns of the training CSV minus
            the target, unscaled.

    Returns:
        The probability and the 0/1 decision taken at the packaged threshold.

    Raises:
        PredictionError: if scoring fails, or the artifact returns a shape
            this module does not recognise.
    """
    frame = pd.DataFrame([features])

    try:
        scored = model.predict(frame)
    except Exception as exc:
        logger.error("Scoring failed for a transaction with %d fields", len(features))
        raise PredictionError(
            "The production model could not score the request"
        ) from exc

    # A pyfunc model may return any of several container types. Checking here
    # keeps the contract explicit and turns a mismatch into this project's own
    # exception instead of an AttributeError escaping as a 500.
    if not isinstance(scored, pd.DataFrame):
        raise PredictionError(
            f"The production model returned {type(scored).__name__}, "
            "expected a DataFrame"
        )

    try:
        probability = float(scored[PROBABILITY_COLUMN].iloc[0])
        decision = int(scored[DECISION_COLUMN].iloc[0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PredictionError(
            "The production model returned an unexpected shape: expected columns "
            f"{PROBABILITY_COLUMN!r} and {DECISION_COLUMN!r}"
        ) from exc

    return Prediction(fraud_probability=probability, is_fraud=decision)
