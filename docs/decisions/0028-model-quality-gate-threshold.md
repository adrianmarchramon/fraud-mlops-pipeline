# Decision 28: The model quality gate — `MIN_PR_AUC = 0.75` and how it was chosen

- **Date:** 2026-08-24
- **Status:** Accepted (inherits [0002](0002-cost-asymmetry.md)'s illustrative-cost caveat)

## Context

Every other test in this suite asserts something about **code**. All 48 of them would still pass
if the model's PR-AUC collapsed to 0.2, because the control flow would be identical and only the
floats inside the artifact would differ. Conventional CI is structurally blind to that failure —
and in fraud detection it is the failure that matters, because a degraded model does not crash:
it silently stops catching fraud while every endpoint keeps answering 200.

[0001](0001-business-metric.md) already fixed the metric for this gate in Phase 0: *"CI's
model-validation gate (Phase 6) must assert a **minimum PR-AUC**, not accuracy."* What remained
open was the number.

## Decision

`tests/test_model_quality.py`, one test, reading `reports/metrics.json`:

```python
METRICS_FILE = REPORTS_DIR / "metrics.json"
METRIC_KEY   = "pr_auc"
MIN_PR_AUC   = 0.75
```

**`0.75` sits just above `0.7249139606556327`** — the PR-AUC of registered versions v2 and v3,
the logistic-regression baseline that [0016](0016-promotion-quality-gate.md)'s promotion gate
**refused to promote**. The rule it encodes is therefore statable in one sentence: *no model may
merge that fails to beat the linear baseline this project already rejected.*

**No `skipif` guard.** A missing metrics file calls `pytest.fail()` with a diagnostic, not skip.

**The constants live in the test, not in `src/config.py`** — except the path, which is built from
the existing `REPORTS_DIR` constant rather than re-deriving the project root.

## Alternatives considered

- **A floor *below* the lowest historical candidate** (e.g. `0.70`) — rejected, and this was the
  key call. It would let CI bless a model scoring exactly what Phase 3's promotion gate refused,
  putting the review-time and promotion-time gates in direct contradiction.
- **The reference's `MIN_PR_AUC = 0.75` copied without derivation** — the value coincides, but
  arriving at it by reasoning is the point; a threshold nobody can justify is a threshold nobody
  will trust when it eventually fires.
- **The exact baseline `0.7249139606556327`** — maximally traceable, but a model scoring `0.7250`
  would pass, which is not a bar.
- **A proportional floor such as `0.85`** (~3% under current) — catches smaller regressions, but
  any genuine drift-driven retrain in Phase 8 would trip it, and the constant would end up being
  tuned rather than trusted.
- **Querying MLflow from the runner** — unnecessary, and it would not work: see below.
- **Putting `MIN_PR_AUC` in `src/config.py`** — rejected. It is a CI policy value; placing it
  there would inject a gate threshold into the configuration the API and training pipeline import
  at runtime.

## Justification

The gate is **self-contained because of a Phase 2 decision**, and that is load-bearing rather
than merely convenient. `dvc.yaml` declares the metrics file under `metrics:` with
`cache: false`, so the numbers live in Git as plain JSON rather than in DVC's content-addressed
cache. Verified directly: a bare `git archive HEAD` tree ships `reports/metrics.json` intact and
**no dataset at all**, and no test references `RAW_DATA`, `TRAIN_PATH`, `read_parquet` or
`creditcard`. Had the metric been a cached output, the gate would have required a `dvc pull` —
and [0005](0005-dvc-local-remote.md) records the DVC remote as a **local filesystem path** no
GitHub runner could ever reach.

The gate was proven to **block**, not merely to pass. Forcing `pr_auc` to `0.61` on a branch:

```
E  AssertionError: PR-AUC 0.6100 is below the required minimum 0.7500: the versioned
   model is degraded and must not be merged or deployed.
```

CI went red on the `pull_request` event and branch protection refused the merge (PR #1,
`merged=null`). A gate never seen refusing anything is unverified.

## Trade-offs / consequences

- **The gate guards the versioned training output, not the Registry.** `reports/metrics.json` is
  whatever `dvc repro` last produced, which need not be what holds `@production`. That is the
  useful reading: a pull request that flips `params.yaml` back to `logistic_regression` and
  re-runs `dvc repro` drops the committed metric to `0.7249` and turns CI red. The gate blocks a
  *worse model configuration* at review time, which nothing previously did.
- **`0.75` will need revisiting when the model genuinely improves.** A floor 0.126 below the
  current `0.8759962787477742` is generous by design; once Phase 8's retraining loop is
  producing models, a tighter floor becomes both safer and affordable.
- **One metric, one threshold.** Precision, recall and F1 are logged and versioned but not
  gated — consistent with [0016](0016-promotion-quality-gate.md), which promotes on PR-AUC alone.
- **A missing metrics file fails the build.** Deliberate: the reference's `skipif` would protect
  against a condition that cannot occur (the file is committed) while silently turning a wrong
  path into a green suite with a dead gate.
