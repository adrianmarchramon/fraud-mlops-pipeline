"""Orchestration tests — verify the Prefect wiring in pipelines/ without a
Prefect server, without MLflow, and without touching data or the Registry.

What is under test is OUR wiring, never Prefect's engine: that each task calls
exactly the function it wraps, that the flow sequences the four stages in the
right order, that the monitoring flow triggers retraining only when drift is
reported, and that the deployment name it fires actually matches the one
pipelines/serve.py registers. Whether Prefect really retries and really logs to
the dashboard is Prefect's own code, verified by observation in Step 6.

Two facts about Prefect 3 drive the mocking strategy (measured, not assumed):
get_run_logger() raises MissingContextError outside a run context, and calling a
@task outside a flow EXECUTES it -- spinning up a temporary server and running
the real function. So both are patched at the module level: the tasks are
replaced with plain callables before the flow body ever reaches them. This keeps
the suite in milliseconds and offline, the standard set by
docs/decisions/0017-registry-testing-and-visibility.md.
"""

import logging
from typing import Any

import pytest

from pipelines import monitoring_pipeline as mp
from pipelines import serve as sv
from pipelines import training_pipeline as tp


@pytest.fixture
def silence_run_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace get_run_logger() in both flow modules with a plain logger.

    Prefect's get_run_logger() raises MissingContextError when there is no
    active flow or task run, which is exactly the situation these tests create
    on purpose.
    """
    plain = logging.getLogger("test.pipelines")
    monkeypatch.setattr(tp, "get_run_logger", lambda: plain)
    monkeypatch.setattr(mp, "get_run_logger", lambda: plain)


# --------------------------------------------------------------------------
# Each task wraps exactly one existing function, and nothing else
# --------------------------------------------------------------------------


def test_validate_task_calls_validate_raw_data_and_returns_its_report(
    monkeypatch: pytest.MonkeyPatch, silence_run_logger: None
) -> None:
    calls: list[str] = []
    report = {"n_rows": 10, "n_fraud": 1, "fraud_rate": 0.1, "status": "ok"}

    def fake_validate() -> dict[str, Any]:
        calls.append("validate")
        return report

    monkeypatch.setattr(tp, "validate_raw_data", fake_validate)

    assert tp.validate_task.fn() == report
    assert calls == ["validate"]


@pytest.mark.parametrize(
    "task_attr, wrapped_name",
    [
        ("preprocess_task", "run_preprocess"),
        ("train_task", "run_train"),
        ("register_task", "run_register"),
    ],
)
def test_task_calls_its_wrapped_function_exactly_once(
    monkeypatch: pytest.MonkeyPatch,
    silence_run_logger: None,
    task_attr: str,
    wrapped_name: str,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(tp, wrapped_name, lambda: calls.append(wrapped_name))

    getattr(tp, task_attr).fn()

    assert calls == [wrapped_name]


# --------------------------------------------------------------------------
# The flow sequences the four stages in order
# --------------------------------------------------------------------------


def test_training_pipeline_runs_the_four_stages_in_order(
    monkeypatch: pytest.MonkeyPatch, silence_run_logger: None
) -> None:
    calls: list[str] = []

    # The TASK OBJECTS are replaced, not the functions they wrap: calling a real
    # @task outside a flow would run it for real through Prefect's engine.
    monkeypatch.setattr(tp, "validate_task", lambda: calls.append("validate"))
    monkeypatch.setattr(tp, "preprocess_task", lambda: calls.append("preprocess"))
    monkeypatch.setattr(tp, "train_task", lambda: calls.append("train"))
    monkeypatch.setattr(tp, "register_task", lambda: calls.append("register"))

    tp.training_pipeline.fn()

    assert calls == ["validate", "preprocess", "train", "register"]


# --------------------------------------------------------------------------
# Retry budgets are a deliberate per-task calibration, not a default
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task_attr, retries, delay",
    [
        ("validate_task", 3, 10),
        ("preprocess_task", 2, 10),
        ("train_task", 2, 30),
        ("register_task", 2, 10),
    ],
)
def test_task_retry_budget(task_attr: str, retries: int, delay: int) -> None:
    task = getattr(tp, task_attr)
    assert task.retries == retries
    assert task.retry_delay_seconds == delay


def test_check_drift_task_retry_budget() -> None:
    assert mp.check_drift_task.retries == 2
    assert mp.check_drift_task.retry_delay_seconds == 30


# --------------------------------------------------------------------------
# The closed loop: drift decides whether retraining fires
# --------------------------------------------------------------------------


def test_monitoring_pipeline_does_not_trigger_retraining_without_drift(
    monkeypatch: pytest.MonkeyPatch, silence_run_logger: None
) -> None:
    triggered: list[dict[str, Any]] = []

    monkeypatch.setattr(mp, "check_drift_task", lambda: False)
    monkeypatch.setattr(mp, "run_deployment", lambda **kwargs: triggered.append(kwargs))

    mp.monitoring_pipeline.fn()

    assert triggered == []


def test_monitoring_pipeline_triggers_retraining_on_drift(
    monkeypatch: pytest.MonkeyPatch, silence_run_logger: None
) -> None:
    triggered: list[dict[str, Any]] = []

    monkeypatch.setattr(mp, "check_drift_task", lambda: True)
    monkeypatch.setattr(mp, "run_deployment", lambda **kwargs: triggered.append(kwargs))

    mp.monitoring_pipeline.fn()

    # timeout=0 is load-bearing: it makes the trigger fire-and-forget, so a
    # monitoring run never blocks for the minutes a retrain takes.
    assert triggered == [{"name": "training-pipeline/on-demand", "timeout": 0}]


def test_check_drift_task_reports_what_detect_drift_returns(
    monkeypatch: pytest.MonkeyPatch, silence_run_logger: None
) -> None:
    monkeypatch.setattr(mp, "detect_drift", lambda: True)
    assert mp.check_drift_task.fn() is True


# The real, unpatched detect_drift() is deliberately no longer asserted here.
# Phase 7 pinned it to False while it was a placeholder; Phase 8 replaced that
# body with an Evidently comparison that reads a DVC-versioned artifact no CI
# runner has, so keeping the assertion would have given this offline suite a
# dataset dependency (docs/decisions/0017-registry-testing-and-visibility.md).
# Its replacement lives in tests/test_drift.py, added in the testing step.


# --------------------------------------------------------------------------
# The deployment name is a coupling across three files — pin it
# --------------------------------------------------------------------------


def test_flow_names_are_the_ones_deployments_are_addressed_by() -> None:
    assert tp.training_pipeline.name == "training-pipeline"
    assert mp.monitoring_pipeline.name == "monitoring-pipeline"


def test_triggered_deployment_matches_the_one_serve_registers() -> None:
    # If either half of this string drifts, retraining silently stops firing:
    # run_deployment would resolve a deployment that was never registered.
    expected = f"{tp.training_pipeline.name}/{sv.TRAINING_DEPLOYMENT_NAME}"
    assert mp.TRAINING_DEPLOYMENT == expected


def test_monitoring_is_scheduled_daily() -> None:
    minute, hour, *rest = sv.MONITORING_CRON.split()
    assert (minute, hour) == ("0", "6")
    assert rest == ["*", "*", "*"]
