"""Model packaging for the MLflow Model Registry.

Planned implementation phase: Phase 3 — Model Registry and Packaging.
Current status: packaging implemented (Phase 3, Steps 2-3). Provides
FraudModel, the mlflow.pyfunc artifact that turns a RAW transaction into a
thresholded fraud decision by carrying the Phase 1 preprocessor, the winning
Phase 2 classifier and the versioned decision threshold as one unit, and
load_threshold(), the scoped params.yaml reader that supplies that threshold.
Registration itself (find_best_run(), build_packaged_model(), register_model())
and the promotion quality gate (get_production_metric(), promote_if_better())
are still to come in Steps 4-5, along with the MODEL_NAME constant they
register under. Nothing in this module writes to the Model Registry yet.

Quality standard (as for every production module here):
    - Strict typing (mypy --strict as reference; avoid unjustified `Any`).
    - Structured logging (never `print()`).
    - Custom application exception hierarchy (never bare `except:`).
    - Pytest coverage arrives with the packaging tests (Phase 3, Step 8).
"""

import logging
from typing import Any, Protocol

import joblib
import mlflow.sklearn
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from src.config import PROJECT_ROOT
from src.exceptions import ModelRegistrationError

logger = logging.getLogger(__name__)

# Binding artifact-key contract for the packaged model. load_context() reads
# exactly these two keys, so the registration step (Step 4) must build its
# `artifacts={...}` mapping with the same names; renaming either is a visible,
# two-sided change, never a silent one.
PREPROCESSOR_ARTIFACT = "preprocessor"
MODEL_ARTIFACT = "model"


class SupportsTransform(Protocol):
    """The only thing FraudModel needs from the fitted preprocessor."""

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the already-fitted transformation to raw features."""
        ...


class SupportsPredictProba(Protocol):
    """The only thing FraudModel needs from the fitted classifier."""

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        """Return per-class probabilities, positive class in column 1."""
        ...


class FraudModel(PythonModel):
    """Packages preprocessor + classifier + threshold into a single artifact.

    Phase 2 trained on `train.parquet`: data already scaled by the Phase 1
    preprocessor. The Phase 4 API will receive raw transactions instead, with
    the schema of `data/raw/creditcard.csv` minus the target. Bridging that gap
    by hand in the API is *training-serving skew* waiting to happen — the
    reimplemented preprocessing drifts from the trained one and the model
    silently degrades without anything appearing broken.

    This artifact closes the gap structurally: it carries the very same fitted
    preprocessor object, so "preprocess before predicting" stops being a manual
    step that production could omit or get subtly wrong. Consumers hand it raw
    transactions and receive both the fraud probability and the thresholded
    decision.
    """

    preprocessor: SupportsTransform
    model: SupportsPredictProba

    def __init__(self, threshold: float) -> None:
        """Store the decision threshold applied at predict time.

        The threshold is the only instance state. The preprocessor and the
        classifier are file-backed dependencies and arrive through
        context.artifacts in load_context() instead: a scalar needs nothing
        more than `python_model` serialization, whereas the two fitted objects
        are better rebuilt through their own MLflow flavors — each with the
        requirements MLflow recorded for it — than collapsed into this
        instance's pickle alongside everything else.

        Args:
            threshold: probability at or above which a transaction is labelled
                fraud. Normally load_threshold(), i.e. params.yaml's
                `train.threshold`.
        """
        self.threshold = threshold

    def load_context(self, context: PythonModelContext) -> None:
        """Deserialize the fitted preprocessor and classifier from artifacts.

        MLflow calls this once, as soon as mlflow.pyfunc.load_model()
        reconstructs the model, so both objects are deserialized a single time
        and reused across every predict() call.

        Paths come exclusively from context.artifacts, never from src.config:
        the packaged artifact has to stay loadable where this repository's
        `data/processed/` does not exist — a container, or any machine that
        only ever pulled the model.

        Args:
            context: the MLflow-supplied context. Must carry the
                PREPROCESSOR_ARTIFACT and MODEL_ARTIFACT keys.

        Raises:
            ModelRegistrationError: if either key is absent, or its target
                cannot be read or deserialized.
        """
        try:
            preprocessor_path = context.artifacts[PREPROCESSOR_ARTIFACT]
            model_path = context.artifacts[MODEL_ARTIFACT]
        except (KeyError, TypeError) as exc:
            raise ModelRegistrationError(
                f"Model context is missing the artifact key {exc}"
            ) from exc

        try:
            self.preprocessor = joblib.load(preprocessor_path)
        except Exception as exc:
            raise ModelRegistrationError(
                f"Could not load the preprocessor from {preprocessor_path}"
            ) from exc

        try:
            self.model = mlflow.sklearn.load_model(model_path)
        except Exception as exc:
            raise ModelRegistrationError(
                f"Could not load the model from {model_path}"
            ) from exc

        logger.info(
            "FraudModel context loaded: preprocessor=%s, model=%s, threshold=%s",
            preprocessor_path,
            model_path,
            self.threshold,
        )

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Score raw transactions: preprocess, get probability, apply threshold.

        The three steps run strictly in that order, and the preprocessor is
        only ever `.transform()`-ed — never re-fit. Re-fitting at inference
        time would leak the scoring batch's own statistics into the
        transformation and produce features the classifier never learned; it is
        the same boundary Phase 1 defends at training time, enforced here at
        serving time.

        No preprocessing beyond that: no imputation, no row filtering, no
        column reordering. Whatever the fitted preprocessor does is the whole
        contract.

        Args:
            context: part of the PythonModel interface; unused, since
                load_context() already resolved the artifacts.
            model_input: one or more RAW transactions — the schema of
                `data/raw/creditcard.csv` without the target column. Never
                pre-scaled data.
            params: part of the PythonModel interface; this model takes no
                inference-time parameters.

        Returns:
            A DataFrame indexed like `model_input`, with `fraud_probability`
            (the positive-class probability) and `is_fraud` (0/1 at the
            packaged threshold). Both travel together because the label drives
            the decision while the probability is what prediction logging and
            downstream drift monitoring consume.

        Raises:
            Whatever the preprocessor or the classifier raises, unmodified. A
            malformed batch is an inference-time failure, not a registration
            one, so it is logged and re-raised rather than relabelled as a
            ModelRegistrationError; translating it into an HTTP response is the
            API layer's job, not the artifact's.
        """
        logger.info(
            "Scoring %d raw transaction(s) at threshold %s",
            len(model_input),
            self.threshold,
        )

        try:
            X = self.preprocessor.transform(model_input)
            proba = self.model.predict_proba(X)[:, 1]
        except Exception:
            logger.error(
                "Scoring failed for a batch of %d raw transaction(s)",
                len(model_input),
            )
            raise

        predictions = pd.DataFrame(
            {
                "fraud_probability": proba,
                "is_fraud": (proba >= self.threshold).astype(int),
            },
            index=model_input.index,
        )

        logger.info(
            "Scored %d transaction(s): %d flagged as fraud (%.4f%%)",
            len(predictions),
            int(predictions["is_fraud"].sum()),
            float(predictions["is_fraud"].mean()) * 100,
        )
        return predictions


def load_threshold() -> float:
    """Load the versioned decision threshold from params.yaml.

    Mirrors src.data.preprocess.load_params() and src.models.train.load_params(),
    but extracts a single scalar (`train.threshold`) instead of a whole block:
    FraudModel needs the number, not the training configuration around it.

    Deliberately does not import train.load_params(). Every module in this
    project owns its own scoped params.yaml reader with its own exception
    mapping, so a failure is always reported in the vocabulary of the module
    that hit it — the same separation that keeps load_split() ignorant of
    MLflow.

    Returns:
        The `train.threshold` value, as a float.

    Raises:
        ModelRegistrationError: if params.yaml is missing, is not valid YAML,
            has no `train.threshold` key, or that key is not numeric.
    """
    params_path = PROJECT_ROOT / "params.yaml"
    try:
        with open(params_path) as f:
            all_params = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ModelRegistrationError(f"params.yaml not found at {params_path}") from exc
    except yaml.YAMLError as exc:
        raise ModelRegistrationError(
            f"params.yaml is not valid YAML: {params_path}"
        ) from exc

    try:
        threshold = all_params["train"]["threshold"]
    except (KeyError, TypeError) as exc:
        raise ModelRegistrationError(
            "params.yaml has no 'train.threshold' key"
        ) from exc

    # bool is a subclass of int, so a stray YAML `true` would otherwise pass.
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ModelRegistrationError(
            f"train.threshold must be numeric, got {type(threshold).__name__}"
        )

    logger.info("Loaded decision threshold from params.yaml: %s", threshold)
    return float(threshold)
