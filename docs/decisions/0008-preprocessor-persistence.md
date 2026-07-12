# Decision 8: Persist the fitted preprocessor to avoid training-serving skew

- **Date:** 2026-07-12
- **Status:** Accepted

## Context

Because the scaler is fit only on the training split ([0006](0006-no-data-leakage.md)), the
fitted transformation carries state (the learned means and standard deviations of `Time` and
`Amount`). At inference time the API must apply the **identical** transformation, or the model
receives inputs in a form it never learned — *training-serving skew*, a silent and common cause
of production degradation.

Evidence in repo: `src/data/preprocess.py` line 144 —
`joblib.dump(preprocessor, PREPROCESSOR_PATH)`; `PREPROCESSOR_PATH` is centralized in
`src/config.py`; `dvc.yaml` declares `data/processed/preprocessor.joblib` as a tracked output of
the `preprocess` stage (hashed in `dvc.lock`).

## Decision

Serialize the fitted `ColumnTransformer` to `preprocessor.joblib` and treat it as a
**first-class, versioned pipeline artifact**. The future inference API (Phase 4) will load this
exact object and call `.transform()` on raw input — never reimplement the preprocessing.

## Alternatives considered

- **Reimplement the transformation in the API** — rejected: two code paths for the same logic
  inevitably drift, reintroducing skew — the exact failure this decision prevents.
- **Fold preprocessing into the model object only** — deferred: packaging preprocessing +
  estimator into a single sklearn `Pipeline` is a natural Phase 2/3 step, but the standalone
  fitted preprocessor is still the correct Phase 1 output and keeps the data pipeline decoupled
  from modelling.

## Justification

Persisting the fitted object guarantees that production and training "speak the same language".
Tracking it with DVC (not Git) versions it alongside the data that produced it, so a given
model can always be paired with the precise preprocessor it was trained against.

## Trade-offs / consequences

- The artifact must be **version-matched** to the model that consumes it — a constraint the
  Phase 3 model registry will formalize.
- It is a binary blob, so it lives in the DVC remote, not in Git; reproducing it from scratch
  requires `dvc pull` or a `dvc repro` run.
