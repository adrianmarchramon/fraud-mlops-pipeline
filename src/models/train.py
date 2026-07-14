"""Training and MLflow experiment tracking for the fraud-detection model.

Planned implementation phase: Phase 2 — Training and Experiment Tracking.
Current status: end-to-end training with MLflow tracking. Provides
load_split() (feature loader), build_model() (classifier factory for
logistic_regression and xgboost), build_training_pipeline() (optional,
mutually-exclusive SMOTE resampling), compute_metrics() (the fraud metrics),
the MLflow-pure artifact writers save_confusion_matrix() / save_pr_curve(),
load_params() / resolve_model_params() (per-model hyperparameter resolution),
and the train() entrypoint that logs a full run to MLflow. evaluate.py's
threshold optimization (Step 10) is still to come.

Quality standard (as for every production module here):
    - Strict typing (mypy --strict as reference; avoid unjustified `Any`).
    - Structured logging (never `print()`).
    - Custom application exception hierarchy (never bare `except:`).
    - Pytest coverage arrives with the model tests (Phase 2, Step 12).
"""

import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import matplotlib
import mlflow
import mlflow.sklearn
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from src.exceptions import ModelTrainingError

matplotlib.use("Agg")  # headless backend; must be set before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402

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
        params: must contain a "model" key naming the algorithm.
            Supports "logistic_regression" and "xgboost".

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

    if name == "xgboost":
        try:
            return XGBClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                learning_rate=params["learning_rate"],
                scale_pos_weight=params["scale_pos_weight"],
                random_state=params["random_state"],
                eval_metric="aucpr",
            )
        except KeyError as exc:
            raise ModelTrainingError(
                f"Missing required hyperparameter for xgboost: {exc}"
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


def build_training_pipeline(
    model: ClassifierMixin, params: dict[str, Any]
) -> ImbPipeline:
    """Wrap an unfitted classifier in an imbalanced-learn pipeline.

    Two imbalance strategies are offered, and they are mutually exclusive
    per run (a deliberate project decision): each run isolates a single
    strategy so the MLflow comparison attributes its effect cleanly,
    rather than stacking two corrections on the same run.

    - params["resampling"] == "smote": a SMOTE oversampling step is
      inserted before the classifier AND the classifier's own class
      weighting is switched off (class_weight=None), so the imbalance is
      corrected by resampling alone. build_model() is closed and
      unconditionally sets class_weight="balanced", so the toggle is
      applied here on the already-built estimator — no re-instantiation,
      no change to build_model().
    - params["resampling"] == "none": no resampling; the classifier keeps
      the class_weight="balanced" it was built with.

    Because imblearn's Pipeline only applies resampling steps during
    .fit(), never during .predict() or .predict_proba(), this guarantees
    resampling touches exclusively the training fold — the same class of
    data-leakage protection that ColumnTransformer gave scaling in
    Phase 1, now applied to resampling.

    Args:
        model: an unfitted classifier, typically the output of
            build_model().
        params: must contain a "resampling" key ("smote" or "none") and,
            when it is "smote", a "random_state" key reused from the rest
            of this params dict.

    Returns:
        An unfitted imblearn Pipeline exposing fit/predict/predict_proba.

    Raises:
        ModelTrainingError: if "resampling" is missing, names an
            unsupported strategy, or "smote" is requested without a
            "random_state".
    """
    strategy = params.get("resampling")
    if strategy is None:
        raise ModelTrainingError("Missing required param: resampling")

    steps: list[tuple[str, Any]] = []
    if strategy == "smote":
        try:
            random_state = params["random_state"]
        except KeyError as exc:
            raise ModelTrainingError("Missing required param: random_state") from exc
        steps.append(("smote", SMOTE(random_state=random_state)))
        model.set_params(class_weight=None)
    elif strategy != "none":
        raise ModelTrainingError(f"Unsupported resampling strategy: {strategy}")
    steps.append(("classifier", model))

    logger.info("Training pipeline built with resampling strategy=%s", strategy)
    return ImbPipeline(steps)


def load_params() -> dict[str, Any]:
    """Load the versioned training parameters from params.yaml.

    Mirrors src.data.preprocess.load_params(), but reads the `train`
    block instead of `preprocess`. Returns a plain dict[str, Any]
    because the block is heterogeneous config consumed by build_model(),
    build_training_pipeline(), and mlflow.log_params().

    Returns:
        The `train` block of params.yaml.

    Raises:
        ModelTrainingError: if params.yaml is missing, is not valid YAML,
            or has no top-level `train` key.
    """
    params_path = PROJECT_ROOT / "params.yaml"
    try:
        with open(params_path) as f:
            all_params = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise ModelTrainingError(f"params.yaml not found at {params_path}") from exc
    except yaml.YAMLError as exc:
        raise ModelTrainingError(
            f"params.yaml is not valid YAML: {params_path}"
        ) from exc

    try:
        return all_params["train"]
    except (KeyError, TypeError) as exc:
        raise ModelTrainingError("params.yaml has no top-level 'train' key") from exc


def resolve_model_params(train_params: dict[str, Any]) -> dict[str, Any]:
    """Flatten the active model's hyperparameters into a single flat dict.

    params.yaml versions the hyperparameters of every supported model
    simultaneously, so switching train.model never requires rewriting the rest
    of the file. build_model() and build_training_pipeline() stay unaware of
    this: they only ever see a flat dictionary scoped to whichever model is
    active. This function is the single place that resolves that translation.

    Args:
        train_params: the raw "train" block as loaded from params.yaml, with
            shared keys (model, threshold, resampling, random_state) plus one
            nested sub-block per supported model.

    Returns:
        A flat dictionary merging the shared keys with the hyperparameters of
        train_params["model"].

    Raises:
        ModelTrainingError: if a required shared key is missing, or the active
            model has no matching sub-block.
    """
    try:
        active_model = train_params["model"]
        model_params = train_params[active_model]
        shared = {
            "model": active_model,
            "threshold": train_params["threshold"],
            "resampling": train_params["resampling"],
            "random_state": train_params["random_state"],
        }
    except KeyError as exc:
        raise ModelTrainingError(f"Malformed train params: missing key {exc}") from exc

    logger.info("Resolved train params for active model: %s", active_model)
    return {**shared, **model_params}


def save_confusion_matrix(
    y_true: pd.Series, y_pred: npt.NDArray[np.int_], output_dir: Path
) -> Path:
    """Render and save a confusion matrix as a PNG.

    Pure with respect to MLflow: it receives already-computed predictions
    and only touches the filesystem, so train() stays the single owner of
    MLflow calls.

    Args:
        y_true: ground-truth labels.
        y_pred: binarized predictions (already thresholded).
        output_dir: directory the PNG is written to; created if missing.

    Returns:
        The path of the saved PNG.

    Raises:
        ModelTrainingError: if rendering or saving the figure fails.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
        path = output_dir / "confusion_matrix.png"
        disp.figure_.savefig(path, bbox_inches="tight")
        plt.close(disp.figure_)
        return path
    except Exception as exc:
        raise ModelTrainingError("Failed to save confusion matrix") from exc


def save_pr_curve(
    y_true: pd.Series, y_proba: npt.NDArray[np.float64], output_dir: Path
) -> Path:
    """Render and save a precision-recall curve as a PNG.

    Pure with respect to MLflow, like save_confusion_matrix().

    Args:
        y_true: ground-truth labels.
        y_proba: predicted probabilities for the positive class.
        output_dir: directory the PNG is written to; created if missing.

    Returns:
        The path of the saved PNG.

    Raises:
        ModelTrainingError: if rendering or saving the figure fails.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        disp = PrecisionRecallDisplay.from_predictions(y_true, y_proba)
        path = output_dir / "pr_curve.png"
        disp.figure_.savefig(path, bbox_inches="tight")
        plt.close(disp.figure_)
        return path
    except Exception as exc:
        raise ModelTrainingError("Failed to save PR curve") from exc


def train() -> None:
    """Train a model end-to-end and log the full experiment to MLflow.

    Loads versioned parameters and the processed train/test splits, builds
    and fits a classifier (optionally wrapped with SMOTE resampling),
    evaluates it at the configured threshold, and logs parameters,
    metrics, visual artifacts, and the model itself to MLflow within a
    single run. Also persists metrics.json to disk so Step 11 can wire it
    into the DVC pipeline.

    The threshold read from params is a deliberately provisional fixed
    value; its principled optimization (F1 or business-cost based) is the
    responsibility of Step 10, not this function.

    Raises:
        ModelTrainingError: if any stage of loading, training, evaluating,
            or logging fails. The mlflow.start_run() context manager runs
            inside the try block, so a failing run is marked FAILED in
            MLflow before the exception is translated to the caller.
    """
    params = resolve_model_params(load_params())
    logger.info(
        "Starting training run: model=%s, resampling=%s, threshold=%s",
        params["model"],
        params["resampling"],
        params["threshold"],
    )

    X_train, y_train = load_split(TRAIN_PATH)
    X_test, y_test = load_split(TEST_PATH)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    try:
        with mlflow.start_run(run_name=params["model"]):
            model = build_model(params)
            pipeline = build_training_pipeline(model, params)
            pipeline.fit(X_train, y_train)

            threshold = params["threshold"]
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)

            metrics = compute_metrics(y_test, y_pred, y_proba)

            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(save_confusion_matrix(y_test, y_pred, MODELS_DIR)))
            mlflow.log_artifact(str(save_pr_curve(y_test, y_proba, MODELS_DIR)))

            signature = infer_signature(X_test, y_pred)
            # cloudpickle, not MLflow 3.x's default skops backend: skops
            # refuses to (de)serialize imblearn.pipeline.Pipeline without an
            # ever-growing skops_trusted_types list, and that list would keep
            # changing per resampling strategy and per model (Step 7). Cloud-
            # pickle serializes the whole pipeline uniformly and is always
            # installed with MLflow.
            mlflow.sklearn.log_model(
                pipeline,
                name="model",
                signature=signature,
                input_example=X_test.iloc[:5],
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )

            metrics_path = REPORTS_DIR / "metrics.json"
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

            run_id = mlflow.active_run().info.run_id
            logger.info("Training completed: run_id=%s, metrics=%s", run_id, metrics)
    except Exception as exc:
        logger.error("Training run failed: %s", exc)
        raise ModelTrainingError("train() failed") from exc


if __name__ == "__main__":
    train()
