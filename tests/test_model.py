"""Model contract tests — verify that the build_model() factory builds what
it promises and rejects what it cannot build, and that the business logic in
src.models.evaluate guarantees what it claims: a threshold in a valid range,
and a cost-sensitive threshold that moves in the direction the cost asymmetry
dictates — each invalid input rejected for the specific reason it is invalid,
not merely because "some exception was raised".

Phase 3 adds two more contracts: that FraudModel turns raw rows into the
probability/decision pair the API and the drift monitor both consume, and that
the promotion quality gate moves the @production alias only when a candidate
genuinely wins.

Deliberately out of scope: cross_validate_pr_auc(), which needs a full Pipeline
and several real folds (an integration test, not a fast contract test); the
train() orchestrator; register_model(), which is an MLflow integration rather
than business logic; and anything touching real parquet files or a live
Registry. These tests use small synthetic arrays and hand-written doubles so
the suite stays fast, offline and deterministic.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from mlflow.entities.model_registry import ModelVersion, ModelVersionTag
from mlflow.exceptions import MlflowException

from src.config import MODEL_NAME
from src.exceptions import (
    ModelEvaluationError,
    ModelRegistrationError,
    ModelTrainingError,
)
from src.models.evaluate import cost_optimal_threshold, optimal_threshold_f1
from src.models.register import (
    PRODUCTION_ALIAS,
    FraudModel,
    get_production_metric,
    load_production_model,
    load_threshold,
    promote_if_better,
)
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


# --------------------------------------------------------------------------
# Phase 3, Step 8 — packaging (FraudModel) and the promotion quality gate
# --------------------------------------------------------------------------


class _FakePreprocessor:
    """Stands in for the fitted ColumnTransformer: passes the frame through.

    Returns the DataFrame unchanged rather than a numpy array, because the real
    preprocessor is built with .set_output(transform="pandas") and FraudModel's
    SupportsTransform protocol promises a DataFrame.
    """

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X


class _FakeModel:
    """Stands in for the trained classifier: fraud probability is column 0."""

    def predict_proba(self, X: pd.DataFrame) -> npt.NDArray[np.float64]:
        p = np.asarray(X, dtype=float)[:, 0]
        return np.column_stack([1.0 - p, p])


def _packaged(threshold: float) -> FraudModel:
    """Build a FraudModel with doubles injected, bypassing load_context().

    load_context() resolves real artifact paths off disk, which is an
    integration concern; predict() is the unit under test here, so the two
    attributes it consumes are set directly.
    """
    model = FraudModel(threshold=threshold)
    model.preprocessor = _FakePreprocessor()
    model.model = _FakeModel()
    return model


def test_packaged_model_returns_probability_and_decision() -> None:
    # Both columns travel together deliberately: the label drives the decision,
    # and the probability is what prediction logging and drift monitoring read.
    result = _packaged(threshold=0.5).predict(None, pd.DataFrame({"feat": [0.2, 0.9]}))
    assert list(result.columns) == ["fraud_probability", "is_fraud"]
    assert len(result) == 2


def test_packaged_model_preserves_the_input_index() -> None:
    # The API pairs each prediction back to the request that produced it.
    raw = pd.DataFrame({"feat": [0.2, 0.9]}, index=[10, 11])
    assert list(_packaged(threshold=0.5).predict(None, raw).index) == [10, 11]


def test_packaged_model_applies_the_threshold() -> None:
    raw = pd.DataFrame({"feat": [0.3, 0.8]})  # probabilities 0.3 and 0.8
    result = _packaged(threshold=0.5).predict(None, raw)
    assert result["is_fraud"].tolist() == [0, 1]


def test_packaged_model_threshold_is_inclusive() -> None:
    # predict() compares with >=, so a probability landing exactly on the
    # threshold is fraud. Pinning it stops a silent flip to > later.
    result = _packaged(threshold=0.5).predict(None, pd.DataFrame({"feat": [0.5]}))
    assert result["is_fraud"].tolist() == [1]


def test_packaged_model_carries_the_threshold_it_was_given() -> None:
    # The same probabilities must decide differently under a different
    # threshold — proof the artifact really applies its own, not a default 0.5.
    raw = pd.DataFrame({"feat": [0.1, 0.4]})
    assert _packaged(threshold=0.5).predict(None, raw)["is_fraud"].tolist() == [0, 0]
    assert _packaged(threshold=0.03).predict(None, raw)["is_fraud"].tolist() == [1, 1]


def test_load_threshold_returns_the_versioned_value() -> None:
    # Reads the real params.yaml: the tuned operating point must survive, since
    # register_model() packages whatever this returns. A silent revert to the
    # meaningless 0.5 default is exactly what this guards against.
    threshold = load_threshold()
    assert isinstance(threshold, float)
    assert threshold != 0.5
    assert 0.0 < threshold < 1.0


def _version(number: str, pr_auc: str | None) -> ModelVersion:
    """Build a real ModelVersion entity — no client, no network."""
    tags = [] if pr_auc is None else [ModelVersionTag(key="pr_auc", value=pr_auc)]
    return ModelVersion(
        name=MODEL_NAME, version=number, creation_timestamp=0, tags=tags
    )


class _FakeRegisteredModel:
    """Carries only the alias map get_production_metric() inspects."""

    def __init__(self, aliases: dict[str, str]) -> None:
        self.aliases = aliases


class _FakeClient:
    """Minimal stand-in for MlflowClient, recording alias writes.

    Only the three methods the promotion path actually calls are implemented;
    anything else would be scaffolding for behaviour under test nowhere.
    """

    def __init__(self, production: ModelVersion | None) -> None:
        self._production = production
        self.alias_calls: list[tuple[str, str, str]] = []

    def get_registered_model(self, name: str) -> _FakeRegisteredModel:
        aliases = (
            {} if self._production is None else {"production": self._production.version}
        )
        return _FakeRegisteredModel(aliases)

    def get_model_version_by_alias(self, name: str, alias: str) -> ModelVersion:
        assert self._production is not None
        return self._production

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self.alias_calls.append((name, alias, version))


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch):
    """Patch register._client so the gate runs entirely in memory.

    Both get_production_metric() and promote_if_better() build their own client
    internally, so there is no argument to inject a double through.
    """

    def _install(production: ModelVersion | None) -> _FakeClient:
        client = _FakeClient(production)
        monkeypatch.setattr("src.models.register._client", lambda: client)
        return client

    return _install


def test_get_production_metric_is_none_when_no_alias_is_set(fake_client) -> None:
    # The first deployment ever is a valid state, not a failure.
    fake_client(None)
    assert get_production_metric() is None


def test_get_production_metric_reads_the_tag(fake_client) -> None:
    fake_client(_version("1", "0.8760"))
    assert get_production_metric() == pytest.approx(0.8760)


def test_promote_when_nothing_is_in_production(fake_client) -> None:
    client = fake_client(None)
    assert promote_if_better(_version("1", "0.7249")) is True
    assert client.alias_calls == [(MODEL_NAME, PRODUCTION_ALIAS, "1")]


def test_promote_when_the_candidate_is_better(fake_client) -> None:
    # The non-trivial branch: a real comparison against the incumbent's tag.
    client = fake_client(_version("1", "0.7249"))
    assert promote_if_better(_version("2", "0.8760")) is True
    assert client.alias_calls == [(MODEL_NAME, PRODUCTION_ALIAS, "2")]


def test_refuse_to_promote_a_worse_candidate(fake_client) -> None:
    # The gate's whole reason to exist: the alias must not move at all.
    client = fake_client(_version("2", "0.8760"))
    assert promote_if_better(_version("3", "0.7249")) is False
    assert client.alias_calls == []


def test_refuse_to_promote_an_equal_candidate(fake_client) -> None:
    # Comparison is strict: matching the incumbent is not beating it, so a
    # re-registration of the same model cannot churn the alias.
    client = fake_client(_version("1", "0.8760"))
    assert promote_if_better(_version("2", "0.8760")) is False
    assert client.alias_calls == []


def test_promotion_rejects_a_candidate_without_a_metric(fake_client) -> None:
    # An untagged version cannot be compared, so it must fail loudly rather
    # than default its way into production.
    client = fake_client(None)
    with pytest.raises(ModelRegistrationError, match="no 'pr_auc' tag"):
        promote_if_better(_version("1", None))
    assert client.alias_calls == []


def test_load_production_model_asks_for_the_alias_not_a_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The whole point of the phase: consumers name a role. If this ever starts
    # building models:/fraud-detector/<number>, promoting a new version would
    # silently stop reaching callers.
    monkeypatch.setattr(
        "src.models.register._client", lambda: _FakeClient(_version("1", "0.8760"))
    )
    seen: list[str] = []

    def _fake_load(uri: str) -> str:
        seen.append(uri)
        return "loaded-model"

    monkeypatch.setattr("mlflow.pyfunc.load_model", _fake_load)

    assert load_production_model() == "loaded-model"
    assert seen == [f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"]


def test_load_production_model_fails_clearly_when_nothing_is_promoted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A registry with versions but no @production is a real state — the error
    # must say that, not surface an opaque URI-resolution failure.
    class _NoAliasClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> ModelVersion:
            raise MlflowException("Registered model alias production not found.")

    monkeypatch.setattr("src.models.register._client", lambda: _NoAliasClient())
    with pytest.raises(ModelRegistrationError, match="holds the @production alias"):
        load_production_model()


def test_load_production_model_translates_a_failing_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A corrupt or unreachable artifact must arrive as this project's own
    # exception type, like every other failure in the module.
    monkeypatch.setattr(
        "src.models.register._client", lambda: _FakeClient(_version("1", "0.8760"))
    )

    def _boom(uri: str) -> str:
        raise OSError("artifact store unreachable")

    monkeypatch.setattr("mlflow.pyfunc.load_model", _boom)
    with pytest.raises(ModelRegistrationError, match="Could not load"):
        load_production_model()
