"""Inference logic: production model loading and prediction.

This module is the API's only point of contact with the Model Registry, and
it knows nothing about HTTP: no FastAPI import, no status codes, no request
objects. It resolves an alias, loads an artifact, and scores a transaction,
raising PredictionError when it cannot. Translating that into a response is
src/api/main.py's job.

From Phase 9 there are two ways the artifact arrives, and load_production()
picks between them on configuration alone. Unset config.MODEL_PATH — every
local run, every Compose run, every test — resolves the @production alias
against a live Registry, unchanged since Phase 4. Set, as it is only on the
public deployment, the model is loaded from a directory inside the image and
nothing is asked of any Registry, because a free-tier service has none
(docs/decisions/0042-bundled-model.md). Both paths return the same pair and
raise the same exception, so main.py cannot tell them apart.

Nothing here reimplements preprocessing or the decision threshold. Both live
inside the packaged artifact registered in Phase 3
(docs/decisions/0015-packaged-model-contract.md), which is precisely what
lets this module stay this small.
"""

import logging
from pathlib import Path
from typing import NamedTuple

import mlflow.pyfunc
import pandas as pd
import yaml
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from mlflow.tracking import MlflowClient

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME, MODEL_PATH
from src.exceptions import ModelRegistrationError, PredictionError
from src.models.register import PRODUCTION_ALIAS, load_production_model

logger = logging.getLogger(__name__)

# The columns FraudModel.predict() returns. Named here so a change to the
# packaged contract breaks in one place instead of surfacing as a KeyError
# deep inside a request.
PROBABILITY_COLUMN = "fraud_probability"
DECISION_COLUMN = "is_fraud"

# MLflow writes this file at the root of every artifact that was registered,
# recording which registered model and version the bundle IS. Reading it is
# what lets /model-info keep answering truthfully with no Registry to ask --
# the alternative was a version number hardcoded in the image, which would go
# stale silently on the first re-export somebody forgot to edit.
# scripts/export_model.py imports these three names rather than repeating them.
BUNDLED_METADATA_FILE = "registered_model_meta"
BUNDLED_NAME_KEY = "model_name"
BUNDLED_VERSION_KEY = "model_version"


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


def load_bundled(
    directory: Path, model_name: str = MODEL_NAME
) -> tuple[PyFuncModel, str]:
    """Load a model that travels inside the image, with no Registry involved.

    This is the public deployment's path, and the only one available there: a
    free-tier service has no MLflow server to resolve an alias against, so the
    artifact is exported into deploy/model/ and shipped with the image
    (docs/decisions/0042-bundled-model.md).

    The version is read from the bundle's own metadata rather than passed in or
    hardcoded, so the answer /model-info gives cannot drift from the artifact
    actually loaded. A bundle for a different registered model is refused
    outright: it means the image was built from the wrong export, and serving it
    while reporting this project's model name would be a lie the caller has no
    way to detect.

    Nothing here reaches the network. Verified by loading this artifact with
    MLFLOW_TRACKING_URI pointed at an unreachable address, from a working
    directory outside the repository: MLflow resolves the preprocessor through
    the relative path in MLmodel, never the absolute URI recorded beside it.

    Args:
        directory: the exported artifact's root, holding MLmodel and the
            registered-model metadata.
        model_name: the registered model the bundle is required to be.

    Returns:
        The loaded artifact and the version recorded inside it.

    Raises:
        PredictionError: if the metadata is missing, malformed, describes a
            different registered model, or the artifact cannot be loaded.
    """
    metadata_path = directory / BUNDLED_METADATA_FILE

    try:
        raw = metadata_path.read_text()
    except OSError as exc:
        raise PredictionError(
            f"No bundled model at {directory}: {metadata_path} is unreadable"
        ) from exc

    try:
        metadata: object = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise PredictionError(f"{metadata_path} is not valid YAML") from exc

    if not isinstance(metadata, dict):
        raise PredictionError(f"{metadata_path} does not contain a mapping")

    if BUNDLED_NAME_KEY not in metadata or BUNDLED_VERSION_KEY not in metadata:
        raise PredictionError(
            f"{metadata_path} is missing {BUNDLED_NAME_KEY!r} or "
            f"{BUNDLED_VERSION_KEY!r}"
        )

    bundled_name = str(metadata[BUNDLED_NAME_KEY])
    if bundled_name != model_name:
        raise PredictionError(
            f"The bundled model at {directory} is {bundled_name!r}, not {model_name!r}"
        )

    # Annotated for the same reason load_production_model() annotates it:
    # mlflow.pyfunc.load_model() is untyped and would otherwise widen the
    # return type of this function to Any.
    model: PyFuncModel
    try:
        model = mlflow.pyfunc.load_model(str(directory))
    except Exception as exc:
        raise PredictionError(
            f"Could not load the bundled model at {directory}"
        ) from exc

    version = str(metadata[BUNDLED_VERSION_KEY])
    logger.info(
        "Serving the bundled %s version %s from %s", model_name, version, directory
    )
    return model, version


def load_production(model_name: str = MODEL_NAME) -> tuple[PyFuncModel, str]:
    """Load the production model and report the version it resolved to.

    Two sources, chosen by configuration rather than by environment sniffing.
    When config.MODEL_PATH is set the model is read from that directory and no
    Registry is contacted; otherwise the @production alias is resolved exactly
    as it has been since Phase 4. The default is empty, so every local and
    Compose workflow keeps the alias indirection — promote a version, restart,
    and the service serves the new model with no rebuild.

    That ordering is deliberate: an explicit MODEL_PATH always wins, so a
    deployment can never half-resolve against a Registry it was not meant to
    reach. This follows 0024-environment-based-tracking-uri.md — one variable,
    a working local default, and no second code path to keep in sync.

    Delegates the alias load to register.py's load_production_model() rather
    than rebuilding the models:/ URI here — the URI stays written once, in the
    module that owns the Registry.

    Args:
        model_name: registered model to load; defaults to the project's.

    Returns:
        The loaded artifact and its version number.

    Raises:
        PredictionError: if the bundle is unusable, or the alias cannot be
            resolved, or the load fails.
    """
    if MODEL_PATH:
        return load_bundled(Path(MODEL_PATH), model_name)

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
