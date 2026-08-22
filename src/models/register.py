"""Model packaging for the MLflow Model Registry.

Planned implementation phase: Phase 3 — Model Registry and Packaging.
Current status: packaging, registration and promotion implemented (Phase 3,
Steps 2-5). Provides FraudModel, the mlflow.pyfunc artifact that turns a RAW
transaction into a thresholded fraud decision by carrying the Phase 1
preprocessor, the winning Phase 2 classifier and the versioned decision
threshold as one unit; load_threshold(), the scoped params.yaml reader that
supplies that threshold; find_best_run() / build_packaged_model() /
register_model(), which turn the best tracked run into a numbered version of
the MODEL_NAME registered model; and get_production_metric() /
promote_if_better(), the quality gate that moves the @production alias only
when a candidate actually beats the incumbent. main() wires the three together
as the `make register` workflow.

Unlike Steps 2-3, this module now WRITES to the Model Registry: register_model()
creates versions and promote_if_better() moves an alias. Still to come: the
formal production-loading deliverable (Step 6), the registry UI walkthrough
(Step 7), and the packaging/promotion tests (Step 8).

Quality standard (as for every production module here):
    - Strict typing (mypy --strict as reference; avoid unjustified `Any`).
    - Structured logging (never `print()`).
    - Custom application exception hierarchy (never bare `except:`).
    - Pytest coverage arrives with the packaging tests (Phase 3, Step 8).
"""

import logging
from typing import Any, Protocol

import joblib
import mlflow
import mlflow.artifacts
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature
from mlflow.pyfunc.model import PythonModel, PythonModelContext
from mlflow.tracking import MlflowClient

from src.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PREPROCESSOR_PATH,
    PROJECT_ROOT,
    TARGET,
)
from src.data.ingest import load_raw_data
from src.exceptions import ModelRegistrationError

logger = logging.getLogger(__name__)

# Binding artifact-key contract for the packaged model. load_context() reads
# exactly these two keys, so the registration step (Step 4) must build its
# `artifacts={...}` mapping with the same names; renaming either is a visible,
# two-sided change, never a silent one.
PREPROCESSOR_ARTIFACT = "preprocessor"
MODEL_ARTIFACT = "model"

# The artifact path src/models/train.py logged its sklearn pipeline under
# (`name="model"`), so a Phase 2 run's model is addressable as
# runs:/<run_id>/model. Named here rather than inlined, since it couples this
# module to a choice made in train.py.
RUN_MODEL_PATH = "model"

# One string, two roles, deliberately: pr_auc is the run metric that ranks
# candidates AND the version tag the quality gate reads back. Keeping them
# identical is what lets promote_if_better() compare without recomputing.
PR_AUC_KEY = "pr_auc"

# The single alias this project uses. Champion/challenger A/B testing is a
# documented stretch goal, not current scope, so no other alias is created.
PRODUCTION_ALIAS = "production"

# Rows of raw data used to infer the model signature and as its input example.
SIGNATURE_SAMPLE_ROWS = 5


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


def _client() -> MlflowClient:
    """Build a registry/tracking client pinned to the project's backend.

    The tracking URI is passed explicitly rather than relying on global state,
    so every function here works standalone — the fluent `mlflow.*` calls in
    register_model() still need the global URI set, and set it themselves.
    """
    return MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


def _read_pr_auc_tag(version: ModelVersion) -> float:
    """Read a registered version's pr_auc tag as a float.

    Args:
        version: the model version whose tag to read.

    Returns:
        The tagged pr_auc.

    Raises:
        ModelRegistrationError: if the tag is absent or not a valid float. A
            version without its metric cannot take part in the quality gate,
            so this is a hard failure rather than a silent default.
    """
    try:
        raw = version.tags[PR_AUC_KEY]
    except KeyError as exc:
        raise ModelRegistrationError(
            f"Version {version.version} of {version.name!r} has no "
            f"{PR_AUC_KEY!r} tag; it cannot be compared"
        ) from exc

    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ModelRegistrationError(
            f"Version {version.version} of {version.name!r} has a non-numeric "
            f"{PR_AUC_KEY!r} tag: {raw!r}"
        ) from exc


def find_best_run(experiment_name: str = EXPERIMENT_NAME) -> Run:
    """Find the run with the highest pr_auc in the given experiment.

    Automates the manual run comparison performed at the close of Phase 2 —
    the judgement recorded in docs/decisions/0013-winning-model-xgboost.md is
    now something the code re-derives rather than something a human remembers.

    Ties are broken by start time, newest first. That is not cosmetic here:
    re-running an unchanged pipeline produces runs with byte-identical metrics,
    so without an explicit second key the winner would depend on MLflow's
    undeclared secondary ordering.

    Args:
        experiment_name: experiment to search; defaults to the project's.

    Returns:
        The best run, as a full MLflow Run entity.

    Raises:
        ModelRegistrationError: if the experiment does not exist, contains no
            runs, or its best run has no pr_auc metric.
    """
    client = _client()
    try:
        experiment = client.get_experiment_by_name(experiment_name)
    except MlflowException as exc:
        raise ModelRegistrationError(
            f"Could not query experiment {experiment_name!r}"
        ) from exc

    if experiment is None:
        raise ModelRegistrationError(
            f"Experiment {experiment_name!r} does not exist; run training first"
        )

    try:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{PR_AUC_KEY} DESC", "attributes.start_time DESC"],
            max_results=1,
        )
    except MlflowException as exc:
        raise ModelRegistrationError(
            f"Could not search runs in experiment {experiment_name!r}"
        ) from exc

    if not runs:
        raise ModelRegistrationError(
            f"Experiment {experiment_name!r} has no runs; run training first"
        )

    best = runs[0]
    if PR_AUC_KEY not in best.data.metrics:
        raise ModelRegistrationError(
            f"No run in {experiment_name!r} has a logged {PR_AUC_KEY!r} metric"
        )

    logger.info(
        "Best run in %r: run_id=%s, %s=%.6f",
        experiment_name,
        best.info.run_id,
        PR_AUC_KEY,
        best.data.metrics[PR_AUC_KEY],
    )
    return best


def build_packaged_model(run: Run) -> tuple[FraudModel, dict[str, str]]:
    """Build a FraudModel and the artifacts mapping MLflow must attach to it.

    Packaging only: nothing is logged and nothing is registered. The returned
    model is deliberately left un-loaded — calling load_context() on it would
    attach the deserialized preprocessor and classifier as instance state, and
    cloudpickle would then embed both in the logged artifact, defeating the
    artifacts mapping that exists precisely to keep them separate.

    The model artifact is a `runs:/` URI rather than a local path: MLflow
    resolves it while logging, so no manual download is needed here.

    Args:
        run: the tracked run whose trained model should be packaged.

    Returns:
        A (model, artifacts) pair ready for mlflow.pyfunc.log_model().

    Raises:
        ModelRegistrationError: if the threshold cannot be read, or the fitted
            preprocessor is missing from disk.
    """
    threshold = load_threshold()

    if not PREPROCESSOR_PATH.exists():
        raise ModelRegistrationError(
            f"Fitted preprocessor not found at {PREPROCESSOR_PATH}; "
            "run the data pipeline first"
        )

    artifacts = {
        PREPROCESSOR_ARTIFACT: str(PREPROCESSOR_PATH),
        MODEL_ARTIFACT: f"runs:/{run.info.run_id}/{RUN_MODEL_PATH}",
    }

    logger.info(
        "Packaged model built for run %s: threshold=%s, artifacts=%s",
        run.info.run_id,
        threshold,
        artifacts,
    )
    return FraudModel(threshold=threshold), artifacts


def register_model(run: Run) -> ModelVersion:
    """Package, log and register the given run as a new fraud-detector version.

    Tags the resulting version with the run's own pr_auc — the value measured
    in Phase 2, never a recomputation — so the quality gate can compare cheaply
    and auditably later. Assigns no alias: promotion is promote_if_better()'s
    exclusive responsibility, and a registered version that is not in
    production is a perfectly valid state.

    Logging happens in a NEW run, never by reopening the training run: that run
    is history from a closed phase (phase-2-complete) and must not be mutated.

    Args:
        run: the tracked training run to register.

    Returns:
        The freshly created ModelVersion, re-fetched after tagging so that the
        returned entity already carries its pr_auc tag.

    Raises:
        ModelRegistrationError: if packaging, signature inference, logging, or
            tagging fails, or the run has no pr_auc metric.
    """
    model, artifacts = build_packaged_model(run)

    try:
        pr_auc = run.data.metrics[PR_AUC_KEY]
    except KeyError as exc:
        raise ModelRegistrationError(
            f"Run {run.info.run_id} has no {PR_AUC_KEY!r} metric to register"
        ) from exc

    raw_sample = load_raw_data().drop(columns=[TARGET]).head(SIGNATURE_SAMPLE_ROWS)

    # The signature is inferred from RAW input and the artifact's REAL output,
    # so it documents the contract Phase 4's API will actually consume:
    # transactions exactly as they arrive, in -> probability + decision, out.
    try:
        local_model_path = mlflow.artifacts.download_artifacts(
            artifact_uri=artifacts[MODEL_ARTIFACT]
        )
        # MLflow's PythonModelContext.__init__ carries no annotations, so a
        # --strict run flags the call, not the arguments; both are correct.
        probe_context = PythonModelContext(  # type: ignore[no-untyped-call]
            artifacts={
                PREPROCESSOR_ARTIFACT: artifacts[PREPROCESSOR_ARTIFACT],
                MODEL_ARTIFACT: local_model_path,
            },
            model_config=None,
        )
        # A throwaway probe, never the instance about to be logged — see
        # build_packaged_model() on why the logged model stays un-loaded.
        probe = FraudModel(threshold=model.threshold)
        probe.load_context(probe_context)
        output_example = probe.predict(probe_context, raw_sample)
        signature = infer_signature(raw_sample, output_example)
    except ModelRegistrationError:
        raise
    except Exception as exc:
        raise ModelRegistrationError(
            f"Could not infer a signature for run {run.info.run_id}"
        ) from exc

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        with mlflow.start_run():
            mlflow.log_metric(PR_AUC_KEY, pr_auc)
            info = mlflow.pyfunc.log_model(
                name=RUN_MODEL_PATH,
                python_model=model,
                artifacts=artifacts,
                signature=signature,
                input_example=raw_sample,
                registered_model_name=MODEL_NAME,
                # FraudModel is cloudpickled BY REFERENCE to src.models.register,
                # so the artifact is only loadable where that module is
                # importable. Shipping src/ inside the version makes it loadable
                # from a container or a bare clone — and versions are immutable,
                # so this cannot be retrofitted later.
                code_paths=[str(PROJECT_ROOT / "src")],
            )
    except Exception as exc:
        raise ModelRegistrationError(
            f"Could not log and register a model version for run {run.info.run_id}"
        ) from exc

    version_number = info.registered_model_version
    if version_number is None:
        raise ModelRegistrationError(
            f"MLflow logged the model for run {run.info.run_id} but registered "
            f"no version under {MODEL_NAME!r}"
        )

    client = _client()
    try:
        client.set_model_version_tag(
            name=MODEL_NAME,
            version=str(version_number),
            key=PR_AUC_KEY,
            value=str(pr_auc),
        )
        version = client.get_model_version(MODEL_NAME, str(version_number))
    except MlflowException as exc:
        raise ModelRegistrationError(
            f"Could not tag version {version_number} of {MODEL_NAME!r}"
        ) from exc

    logger.info(
        "Registered %r version %s from run %s (%s=%.6f), no alias assigned",
        MODEL_NAME,
        version.version,
        run.info.run_id,
        PR_AUC_KEY,
        pr_auc,
    )
    return version


def get_production_metric(model_name: str = MODEL_NAME) -> float | None:
    """Read the pr_auc tag of the version currently aliased @production.

    Args:
        model_name: registered model to inspect; defaults to the project's.

    Returns:
        The production version's tagged pr_auc, or None when no version holds
        the alias. None is a valid system state — the first deployment ever —
        not a failure, which is why it is a return value and not an exception.

    Raises:
        ModelRegistrationError: for any failure other than a missing model or
            alias, including a present but non-numeric pr_auc tag.
    """
    client = _client()

    # "No production model yet" arrives in two shapes, and MLflow reports them
    # with different error codes: an unknown registered model raises
    # RESOURCE_DOES_NOT_EXIST, while a known model without the alias raises
    # INVALID_PARAMETER_VALUE — a code that also covers genuine errors, so it
    # cannot simply be swallowed. Testing the alias map turns the second case
    # into an explicit membership check instead of exception-code archaeology.
    try:
        registered = client.get_registered_model(model_name)
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            logger.info("Registered model %r does not exist yet", model_name)
            return None
        raise ModelRegistrationError(
            f"Could not read the registered model {model_name!r}"
        ) from exc

    if PRODUCTION_ALIAS not in registered.aliases:
        logger.info(
            "No version of %r holds the @%s alias yet", model_name, PRODUCTION_ALIAS
        )
        return None

    try:
        version = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    except MlflowException as exc:
        raise ModelRegistrationError(
            f"Could not read the @{PRODUCTION_ALIAS} version of {model_name!r}"
        ) from exc

    metric = _read_pr_auc_tag(version)
    logger.info(
        "Current @%s: %r version %s (%s=%.6f)",
        PRODUCTION_ALIAS,
        model_name,
        version.version,
        PR_AUC_KEY,
        metric,
    )
    return metric


def promote_if_better(version: ModelVersion, model_name: str = MODEL_NAME) -> bool:
    """Move the @production alias to `version` only if it beats the incumbent.

    This is the phase's quality gate: an automatic check that an inferior model
    cannot reach production by accident. It compares tagged metrics — never
    re-evaluating anything — so the decision is cheap, deterministic and
    auditable after the fact. The same idea returns in Phase 6 as the CI model
    validation gate.

    Pure comparison and decision: it does not register anything and knows
    nothing about how the version it receives was built.

    Args:
        version: the candidate version, carrying its own pr_auc tag.
        model_name: registered model to promote within.

    Returns:
        True if the alias now points at `version`, False if it was left where
        it was. Either way the decision is logged with both metrics — a
        rejection is never a silent no-op.

    Raises:
        ModelRegistrationError: if the candidate's pr_auc tag is missing or
            non-numeric, or the alias assignment fails.
    """
    candidate = _read_pr_auc_tag(version)
    production = get_production_metric(model_name)

    if production is None:
        reason = f"no version held @{PRODUCTION_ALIAS} yet"
    elif candidate > production:
        reason = f"{PR_AUC_KEY} {candidate:.6f} beats the incumbent's {production:.6f}"
    else:
        logger.info(
            "Version %s of %r NOT promoted: %s %.6f does not beat the "
            "@%s version's %.6f — the alias stays where it is",
            version.version,
            model_name,
            PR_AUC_KEY,
            candidate,
            PRODUCTION_ALIAS,
            production,
        )
        return False

    try:
        _client().set_registered_model_alias(
            model_name, PRODUCTION_ALIAS, str(version.version)
        )
    except MlflowException as exc:
        raise ModelRegistrationError(
            f"Could not move the @{PRODUCTION_ALIAS} alias of {model_name!r} "
            f"to version {version.version}"
        ) from exc

    logger.info(
        "Version %s of %r promoted to @%s: %s",
        version.version,
        model_name,
        PRODUCTION_ALIAS,
        reason,
    )
    return True


def main() -> None:
    """Run the registration workflow: select the best run, register, promote.

    The entry point behind `make register`, mirroring how `make train` invokes
    src.models.train. Selection is automatic here; registering a specific run
    by hand is done by calling register_model() directly.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run = find_best_run()
    version = register_model(run)
    promoted = promote_if_better(version)
    logger.info(
        "Registration finished: %r version %s, promoted=%s",
        MODEL_NAME,
        version.version,
        promoted,
    )


if __name__ == "__main__":
    main()
