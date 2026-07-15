"""Rigorous evaluation: cross-validation and decision-threshold optimization.

Implemented in Phase 2 — Training and Experiment Tracking (Step 10).
Provides cross_validate_pr_auc() (stability of the PR-AUC estimate across
stratified folds), optimal_threshold_f1() (the F1-maximizing operating
point), and cost_optimal_threshold() (the operating point that minimizes
expected business cost, closing the loop with the FP/FN asymmetry recorded
in ADR 0002).

All three are pure: they receive already-loaded data and return numbers.
They never touch MLflow, params.yaml, or the filesystem — orchestrating
that is the caller's job, exactly as with compute_metrics() in train.py.
This module therefore depends on nothing else in src/models/.

Quality standard (as for every production module here):
    - Strict typing (mypy --strict as reference; avoid unjustified `Any`).
    - Structured logging (never `print()`).
    - Custom application exception hierarchy (never bare `except:`).
    - Pytest coverage arrives with the model tests (Phase 2, Step 12).
"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.exceptions import ModelEvaluationError

logger = logging.getLogger(__name__)


def cross_validate_pr_auc(
    estimator: ImbPipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int,
    random_state: int,
) -> tuple[float, float]:
    """Estimate PR-AUC via stratified cross-validation, training split only.

    Evaluating on a single fixed split can mislead: the result may depend on
    the luck of that split. Cross-validation refits the model on several
    splits and reports not just a mean but a standard deviation — a model
    with a good mean and high variance is less trustworthy than one slightly
    worse but stable. StratifiedKFold (not plain KFold) is required by the
    class imbalance: it preserves the fraud proportion in every fold, the
    same guarantee the stratified split gave in Phase 1.

    `estimator` must be the UNFITTED output of
    build_training_pipeline(build_model(params), params), never the bare
    classifier from build_model() alone. cross_val_score clones and refits
    the estimator once per fold; passing the full imblearn Pipeline is what
    keeps SMOTE (when active) confined to each fold's training portion,
    extending to cross-validation the exact leakage protection Steps 5-6
    established for a single train() call.

    Args:
        estimator: unfitted imblearn Pipeline (classifier + optional SMOTE).
        X_train: training features only — never X_test.
        y_train: training target only — never y_test.
        n_splits: number of stratified folds, from
            params["evaluate"]["cv_folds"].
        random_state: reused from params["train"]["random_state"].

    Returns:
        (mean_pr_auc, std_pr_auc) across the n_splits folds.

    Raises:
        ModelEvaluationError: if cross-validation fails.
    """
    try:
        # Inside the try: StratifiedKFold itself rejects n_splits < 2, and that
        # failure must reach the caller as ModelEvaluationError like any other.
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            estimator, X_train, y_train, cv=cv, scoring="average_precision"
        )
    # Broad by intent: cross_val_score refits an arbitrary estimator per fold,
    # so any failure inside it (sklearn, imblearn, xgboost) surfaces here and
    # must reach the caller as ModelEvaluationError, never as a raw library
    # exception.
    except Exception as exc:
        logger.error("Cross-validation failed: %s", exc)
        raise ModelEvaluationError(f"Cross-validation failed: {exc}") from exc

    mean_score, std_score = float(scores.mean()), float(scores.std())
    logger.info(
        "Cross-validated PR-AUC: mean=%.4f std=%.4f (n_splits=%d)",
        mean_score,
        std_score,
        n_splits,
    )
    return mean_score, std_score


def optimal_threshold_f1(
    y_true: pd.Series, y_proba: npt.NDArray[np.floating[Any]]
) -> tuple[float, float]:
    """Find the decision threshold maximizing F1 on the precision-recall curve.

    The default 0.5 is almost never optimal under severe imbalance. This is
    the metric-driven answer to where the boundary belongs; the
    business-driven answer is cost_optimal_threshold().

    Args:
        y_true: ground-truth labels.
        y_proba: predicted probabilities of the positive class. Typed
            precision-agnostic because xgboost returns float32 while
            scikit-learn returns float64, and neither is preferable here.

    Returns:
        (threshold, f1_at_that_threshold).

    Raises:
        ModelEvaluationError: if the precision-recall curve cannot be
            computed.
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    except ValueError as exc:
        logger.error("Could not compute PR curve: %s", exc)
        raise ModelEvaluationError(f"Could not compute PR curve: {exc}") from exc

    # precision/recall carry one element more than thresholds (the final
    # point has no threshold), so drop it to keep the arrays aligned.
    f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
    best_idx = int(np.argmax(f1_scores[:-1]))
    threshold, f1_at_best = float(thresholds[best_idx]), float(f1_scores[best_idx])

    logger.info("Optimal F1 threshold: %.4f (f1=%.4f)", threshold, f1_at_best)
    return threshold, f1_at_best


def cost_optimal_threshold(
    y_true: pd.Series,
    y_proba: npt.NDArray[np.floating[Any]],
    cost_fp: float,
    cost_fn: float,
) -> tuple[float, float]:
    """Find the decision threshold minimizing total expected business cost.

    Where optimal_threshold_f1() optimizes an abstract metric, this one
    optimizes money: it sweeps candidate thresholds and returns the one
    whose confusion matrix costs least, given the per-error costs. This is
    what turns the FP/FN asymmetry recorded in ADR 0002 into a concrete
    operating point — the costs are parameters, never hardcoded, precisely
    because ADR 0002 marks them illustrative and pending real data.

    Args:
        y_true: ground-truth labels.
        y_proba: predicted probabilities of the positive class.
        cost_fp: business cost of a single false positive, from
            params["evaluate"]["cost_fp"].
        cost_fn: business cost of a single false negative, from
            params["evaluate"]["cost_fn"].

    Returns:
        (threshold, total_expected_cost_at_that_threshold).

    Raises:
        ModelEvaluationError: if cost_fp/cost_fn are not strictly positive.
    """
    if cost_fp <= 0 or cost_fn <= 0:
        raise ModelEvaluationError(
            f"cost_fp and cost_fn must be positive, got "
            f"cost_fp={cost_fp}, cost_fn={cost_fn}"
        )

    y_true_arr = np.asarray(y_true)
    candidates = np.linspace(0.01, 0.99, 99)
    costs: list[float] = []
    for t in candidates:
        y_pred = (y_proba >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
        fn = int(((y_pred == 0) & (y_true_arr == 1)).sum())
        costs.append(fp * cost_fp + fn * cost_fn)

    best_idx = int(np.argmin(costs))
    threshold, total_cost = float(candidates[best_idx]), float(costs[best_idx])

    logger.info(
        "Cost-optimal threshold: %.4f (expected_cost=%.2f, cost_fp=%.2f, cost_fn=%.2f)",
        threshold,
        total_cost,
        cost_fp,
        cost_fn,
    )
    return threshold, total_cost
