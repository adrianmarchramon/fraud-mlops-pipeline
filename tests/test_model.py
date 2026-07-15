"""Model contract tests — verify that the build_model() factory builds what
it promises and rejects what it cannot build, and that the business logic in
src.models.evaluate guarantees what it claims: a threshold in a valid range,
and a cost-sensitive threshold that moves in the direction the cost asymmetry
dictates — each invalid input rejected for the specific reason it is invalid,
not merely because "some exception was raised".

Deliberately out of scope: cross_validate_pr_auc(), which needs a full Pipeline
and several real folds (an integration test, not a fast contract test); the
train() orchestrator; and anything touching real parquet files or MLflow. These
tests use small synthetic arrays so the suite stays fast and deterministic.
"""

import numpy as np
import pytest

from src.exceptions import ModelEvaluationError, ModelTrainingError
from src.models.evaluate import cost_optimal_threshold, optimal_threshold_f1
from src.models.train import build_model


def test_build_model_builds_logistic_regression() -> None:
    model = build_model(
        {"model": "logistic_regression", "max_iter": 100, "random_state": 42}
    )
    params = model.get_params()
    assert model.__class__.__name__ == "LogisticRegression"
    assert params["max_iter"] == 100
    assert params["random_state"] == 42
    # build_model() always builds the class-weighted variant; switching it off
    # for a SMOTE run is build_training_pipeline()'s job, not the factory's.
    assert params["class_weight"] == "balanced"


def test_build_model_builds_xgboost() -> None:
    model = build_model(
        {
            "model": "xgboost",
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "scale_pos_weight": 577.29,
            "random_state": 42,
        }
    )
    params = model.get_params()
    assert model.__class__.__name__ == "XGBClassifier"
    assert params["n_estimators"] == 50
    assert params["max_depth"] == 3
    # xgboost's imbalance knob, in place of the class_weight it silently ignores.
    assert params["scale_pos_weight"] == 577.29


def test_build_model_rejects_unsupported_model() -> None:
    with pytest.raises(ModelTrainingError, match="Unsupported model"):
        build_model({"model": "random_forest", "random_state": 42})


@pytest.mark.parametrize(
    "params",
    [
        {"model": "logistic_regression", "random_state": 42},
        {
            "model": "xgboost",
            "max_depth": 3,
            "learning_rate": 0.1,
            "scale_pos_weight": 577.29,
            "random_state": 42,
        },
    ],
    ids=["logistic_regression-missing-max_iter", "xgboost-missing-n_estimators"],
)
def test_build_model_rejects_missing_hyperparameter(params: dict) -> None:
    with pytest.raises(ModelTrainingError, match="Missing required hyperparameter"):
        build_model(params)


def test_optimal_threshold_f1_in_valid_range() -> None:
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.6])
    threshold, f1 = optimal_threshold_f1(y_true, y_proba)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= f1 <= 1.0


# Deliberately overlapping scores: one negative (0.6) ranked above one positive
# (0.4). Perfectly separable data would admit a zero-cost threshold and return
# the same answer whatever the costs, so it could not tell a cost-sensitive
# function apart from one that ignores costs entirely.
OVERLAPPING_Y_TRUE = np.array([0, 0, 1, 1])
OVERLAPPING_Y_PROBA = np.array([0.3, 0.6, 0.4, 0.7])


def test_cost_optimal_threshold_favors_recall_when_fn_is_expensive() -> None:
    # A false negative costing 1000x a false positive must push the boundary
    # down: catching fraud is worth tolerating false alarms.
    threshold, _ = cost_optimal_threshold(
        OVERLAPPING_Y_TRUE, OVERLAPPING_Y_PROBA, cost_fp=1, cost_fn=1000
    )
    assert threshold < 0.4


def test_cost_optimal_threshold_favors_precision_when_fp_is_expensive() -> None:
    # The mirror image: inverting the asymmetry must move the threshold the
    # other way. Together with the test above, this is what actually pins the
    # DIRECTION rather than a single low value.
    threshold, _ = cost_optimal_threshold(
        OVERLAPPING_Y_TRUE, OVERLAPPING_Y_PROBA, cost_fp=1000, cost_fn=1
    )
    assert threshold > 0.6


def test_cost_optimal_threshold_direction_responds_to_asymmetry() -> None:
    cheap_fn, _ = cost_optimal_threshold(
        OVERLAPPING_Y_TRUE, OVERLAPPING_Y_PROBA, cost_fp=1000, cost_fn=1
    )
    expensive_fn, _ = cost_optimal_threshold(
        OVERLAPPING_Y_TRUE, OVERLAPPING_Y_PROBA, cost_fp=1, cost_fn=1000
    )
    assert expensive_fn < cheap_fn


@pytest.mark.parametrize(
    "cost_fp, cost_fn",
    [(0, 10), (10, 0), (-1, 10), (10, -1)],
    ids=["zero-fp", "zero-fn", "negative-fp", "negative-fn"],
)
def test_cost_optimal_threshold_rejects_non_positive_costs(
    cost_fp: float, cost_fn: float
) -> None:
    y_true = np.array([0, 1])
    y_proba = np.array([0.4, 0.6])
    with pytest.raises(ModelEvaluationError, match="must be positive"):
        cost_optimal_threshold(y_true, y_proba, cost_fp=cost_fp, cost_fn=cost_fn)
