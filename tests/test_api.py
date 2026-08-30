"""Tests for the inference API (Phases 4 and 9).

The HTTP tests never touch MLflow. A hand-written double is injected straight
into the application state and the TestClient is built WITHOUT its context
manager, so the real lifespan never runs and no Registry lookup happens — the
same technique tests/test_model.py uses for the Registry client, and what makes
this suite runnable in CI with no mlflow.db present.

The double reports version "7" on purpose: the live Registry serves version
"1", so a test passing here proves it read the injected double rather than
quietly reaching the real production model.

The Phase 9 section at the bottom is the exception, and deliberately so: it
loads the real artifact committed under deploy/model/. That is the one thing no
double can check. CI runs from a bare checkout with no dataset, no DVC pull and
no mlflow.db, so a bundle that loads there is a bundle that will load on a
free-tier container with no MLflow in reach — which is the entire premise of
the public deployment.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import main, predict
from src.config import DEPLOY_MODEL_DIR
from src.exceptions import PredictionError

FAKE_VERSION = "7"
FAKE_PROBABILITY = 0.9
FAKE_DECISION = 1


class _FakeModel:
    """Stands in for the packaged artifact, honouring its output contract."""

    def predict(self, model_input: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "fraud_probability": [FAKE_PROBABILITY] * len(model_input),
                "is_fraud": [FAKE_DECISION] * len(model_input),
            }
        )


def _payload() -> dict[str, float]:
    payload = {"Time": 0.0, "Amount": 100.0}
    payload.update({f"V{i}": 0.0 for i in range(1, 29)})
    return payload


@pytest.fixture(autouse=True)
def predictions_log(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Redirect the prediction log for every test in this module.

    autouse, not opt-in: a test that forgot to ask would append synthetic rows
    to the real logs/predictions.jsonl, which Phase 8 treats as the production
    distribution. Poisoning that baseline is a bug that only surfaces phases
    later, as drift nobody can explain.

    Patches the name on src.api.main, not src.config: main imported the
    constant by value, so patching config would silently no-op and the tests
    would write to the real file while appearing to pass.
    """
    log = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(main, "PREDICTIONS_LOG", log)
    return log


@pytest.fixture
def bare_client():
    """A client whose app holds no model — the degraded-startup case."""
    for key in (main.MODEL_STATE_KEY, main.VERSION_STATE_KEY):
        if hasattr(main.app.state, key):
            delattr(main.app.state, key)
    return TestClient(main.app)


@pytest.fixture
def client():
    """A client whose app holds the double, torn down afterwards."""
    setattr(main.app.state, main.MODEL_STATE_KEY, _FakeModel())
    setattr(main.app.state, main.VERSION_STATE_KEY, FAKE_VERSION)
    yield TestClient(main.app)
    for key in (main.MODEL_STATE_KEY, main.VERSION_STATE_KEY):
        if hasattr(main.app.state, key):
            delattr(main.app.state, key)


def test_health_reports_ok_when_a_model_is_loaded(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_no_model_when_none_is_loaded(bare_client) -> None:
    # /health must answer even in the degraded state: reporting it is its job.
    response = bare_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "no_model"


def test_model_info_reports_the_served_version(client) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json() == {
        "model_name": "fraud-detector",
        "version": FAKE_VERSION,
        "alias": "production",
    }


def test_model_info_is_unavailable_without_a_model(bare_client) -> None:
    response = bare_client.get("/model-info")
    assert response.status_code == 503


def test_predict_returns_the_packaged_probability_and_decision(client) -> None:
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200

    body = response.json()
    assert body["is_fraud"] == FAKE_DECISION
    assert body["fraud_probability"] == pytest.approx(FAKE_PROBABILITY)
    assert body["model_version"] == FAKE_VERSION


def test_predict_rejects_a_payload_missing_a_field(client) -> None:
    payload = _payload()
    del payload["Amount"]

    response = client.post("/predict", json=payload)

    # 422 comes from Pydantic, with no validation code of ours involved.
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "Amount"]


def test_predict_rejects_a_negative_amount(client) -> None:
    # Well-formed and correctly typed, but out of range: the constraint, not
    # merely the field's presence, has to be enforced.
    response = client.post("/predict", json={**_payload(), "Amount": -5.0})

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "greater_than_equal"


def test_predict_is_unavailable_without_a_model(bare_client) -> None:
    response = bare_client.post("/predict", json=_payload())
    assert response.status_code == 503


def test_predict_appends_one_record_to_the_log(client, predictions_log) -> None:
    assert not predictions_log.exists()

    client.post("/predict", json=_payload())

    assert predictions_log.exists()
    assert len(predictions_log.read_text().splitlines()) == 1


def test_logged_record_matches_the_monitoring_contract(client, predictions_log) -> None:
    client.post("/predict", json=_payload())
    record = json.loads(predictions_log.read_text().splitlines()[0])

    # Phase 8 reads json.loads(line)["input"] to rebuild the production
    # feature distribution; these key names are its contract, not a detail.
    assert set(record) == {
        "timestamp",
        "input",
        "fraud_probability",
        "is_fraud",
        "model_version",
    }
    assert record["input"] == _payload()
    assert record["timestamp"].endswith("+00:00")
    assert record["model_version"] == FAKE_VERSION


def test_rejected_payloads_are_not_logged(client, predictions_log) -> None:
    payload = _payload()
    del payload["Amount"]

    client.post("/predict", json=payload)

    # Nothing was predicted, so nothing belongs in the monitoring baseline.
    assert not predictions_log.exists()


# --------------------------------------------------------------------------
# Loading the model that travels inside the image (Phase 9)
# --------------------------------------------------------------------------


def _forbid_registry(*args, **kwargs):
    raise AssertionError("the bundled path must not contact a Model Registry")


def _write_metadata(directory: Path, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / predict.BUNDLED_METADATA_FILE).write_text(text)
    return directory


def test_the_committed_bundle_loads_and_scores_with_no_registry(monkeypatch) -> None:
    # The load-bearing test of the whole deployment. It runs against the real
    # deploy/model/ in Git, and CI has no dataset, no DVC pull and no
    # mlflow.db — so passing here means the artifact resolves everything from
    # inside its own directory. MlflowClient is replaced with a landmine
    # rather than merely unused: an accidental Registry call must fail the
    # test, not silently succeed on a developer machine where one is running.
    monkeypatch.setattr(predict, "MlflowClient", _forbid_registry)
    monkeypatch.setattr(predict, "MODEL_PATH", str(DEPLOY_MODEL_DIR))

    model, version = predict.load_production()
    scored = model.predict(pd.DataFrame([_payload()]))

    assert version == "1"
    assert set(scored.columns) == {"fraud_probability", "is_fraud"}
    assert 0.0 <= float(scored["fraud_probability"].iloc[0]) <= 1.0
    assert int(scored["is_fraud"].iloc[0]) in (0, 1)


def test_an_unset_model_path_still_resolves_the_alias(monkeypatch) -> None:
    # The other half of the branch, and the one that protects Phases 4-8:
    # nothing about the default path may change because a bundle now exists.
    # MODEL_PATH is patched on predict, not on src.config — predict imported
    # it by value, so patching config would silently no-op
    # (docs/decisions/0021-prediction-log-and-api-tests.md).
    monkeypatch.setattr(predict, "MODEL_PATH", "")
    monkeypatch.setattr(predict, "load_production_model", lambda name: "aliased")
    monkeypatch.setattr(predict, "resolve_production_version", lambda name: "42")

    assert predict.load_production() == ("aliased", "42")


def test_the_reported_version_comes_from_the_bundle_not_from_the_code(
    tmp_path, monkeypatch
) -> None:
    # The committed bundle is version 1, so every assertion above would still
    # pass if the version were hardcoded to "1" — mutation-tested and confirmed.
    # A bundle claiming 99 must be reported as 99, or an image shipping a stale
    # export would announce whatever number somebody last typed into the code.
    bundle = _write_metadata(
        tmp_path, "model_name: fraud-detector\nmodel_version: 99\n"
    )
    monkeypatch.setattr("mlflow.pyfunc.load_model", lambda uri: "loaded")

    assert predict.load_bundled(bundle) == ("loaded", "99")


def test_a_bundle_for_another_registered_model_is_refused(tmp_path) -> None:
    # An image built from the wrong export must fail loudly at startup rather
    # than serve someone else's model under this project's name.
    bundle = _write_metadata(tmp_path, "model_name: other-model\nmodel_version: 3\n")

    with pytest.raises(PredictionError, match="is 'other-model', not 'fraud-detector'"):
        predict.load_bundled(bundle)


def test_a_missing_bundle_reports_where_it_looked(tmp_path) -> None:
    with pytest.raises(PredictionError, match="No bundled model at"):
        predict.load_bundled(tmp_path / "absent")


def test_bundle_metadata_that_is_not_a_mapping_is_refused(tmp_path) -> None:
    bundle = _write_metadata(tmp_path, "just a string\n")

    with pytest.raises(PredictionError, match="does not contain a mapping"):
        predict.load_bundled(bundle)


def test_bundle_metadata_missing_the_version_is_refused(tmp_path) -> None:
    bundle = _write_metadata(tmp_path, "model_name: fraud-detector\n")

    with pytest.raises(PredictionError, match="is missing"):
        predict.load_bundled(bundle)
