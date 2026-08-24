"""Model quality gate — fails the build when the versioned model has degraded.

This is the specifically MLOps part of the CI pipeline. Every other test in this
suite asserts something about *code*: they would all still pass if the model's
PR-AUC collapsed to 0.2, because the control flow would be identical and only
the floats inside the artifact would differ. This test is the only one that
asserts a property of the *model* — which in fraud detection is the failure that
matters, since a degraded model does not crash: it silently stops catching fraud
while every endpoint keeps answering 200.

It reads reports/metrics.json directly and never contacts MLflow. That is
possible because Phase 2 declared the file under `metrics:` with `cache: false`
in dvc.yaml, so the numbers live in Git rather than in DVC's content-addressed
cache: a runner with no dataset, no credentials and no tracking server still has
them after a plain checkout. That is load-bearing rather than merely convenient,
because the DVC remote is a local filesystem path
(docs/decisions/0005-dvc-local-remote.md) that no GitHub runner could ever reach.
"""

import json

import pytest

from src.config import REPORTS_DIR

METRICS_FILE = REPORTS_DIR / "metrics.json"

# PR-AUC, never accuracy or ROC-AUC: at a fraud rate under 0.2% ROC-AUC stays
# flattering while the model is useless, and accuracy above 99% is reachable by
# predicting "not fraud" every time (docs/decisions/0001-business-metric.md,
# which states that CI's gate must assert a minimum PR-AUC specifically).
METRIC_KEY = "pr_auc"

# Provenance of the floor, read from the Model Registry when this gate was
# written: v1 (XGBoost, @production) scored 0.8759962787477742, while v2 and v3
# (the logistic-regression baseline) both scored 0.7249139606556327 — and v3 was
# *refused* promotion by the gate in
# docs/decisions/0016-promotion-quality-gate.md.
#
# 0.75 sits just above that rejected baseline, so the rule it encodes is "no
# model may merge that fails to beat the linear baseline this project already
# refused", and the review-time gate cannot contradict the promotion-time one. A
# floor below 0.7249 would have blessed exactly what Phase 3 rejected.
#
# It leaves 0.126 of headroom under the current value — far more than retraining
# variance — so an honest retrain never turns CI red. What it does catch: a PR
# that flips params.yaml back to logistic_regression and re-runs `dvc repro`
# drops the committed metric to 0.7249 and fails this test. That is the point.
MIN_PR_AUC = 0.75


def test_pr_auc_meets_minimum_threshold() -> None:
    # Deliberately not a skipif on METRICS_FILE.exists(). The file is committed
    # to Git, so it is present in every checkout; a guard would protect against
    # nothing while silently turning a wrong path into a passing suite — a gate
    # that quietly disables itself is worse than no gate.
    if not METRICS_FILE.exists():
        pytest.fail(
            f"{METRICS_FILE} is missing, so the model quality gate cannot run. "
            "It is a `cache: false` metric of the train stage and belongs in "
            "Git — run `dvc repro` and commit the result."
        )

    metrics: dict[str, float] = json.loads(METRICS_FILE.read_text())

    if METRIC_KEY not in metrics:
        pytest.fail(
            f"{METRICS_FILE} has no {METRIC_KEY!r} key (found: {sorted(metrics)}). "
            "The gate asserts PR-AUC specifically, so a renamed key must not "
            "pass unnoticed."
        )

    pr_auc = metrics[METRIC_KEY]

    assert pr_auc >= MIN_PR_AUC, (
        f"PR-AUC {pr_auc:.4f} is below the required minimum {MIN_PR_AUC:.4f}: "
        "the versioned model is degraded and must not be merged or deployed."
    )
