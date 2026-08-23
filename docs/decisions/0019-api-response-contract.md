# Decision 19: API response contract — probability alongside the decision

- **Date:** 2026-08-23
- **Status:** Accepted

## Context

`project_context/mlops_phase4.md` specifies a three-field `PredictionResponse`
(`fraud_probability`, `is_fraud`, `model_version`) and populates it by reading two columns off
whatever `FraudModel.predict()` returns (`mlops_phase4.md:184-185`).

Successive planning prompts for this phase asserted the opposite premise: that Phase 3 had left
`predict()` returning a bare `pd.Series` of 0/1 labels, "never a two-column DataFrame", and that
the reference material therefore could not run against the registered model. Acting on that
premise would have meant editing `src/models/register.py` — a closed phase — and registering a
new `fraud-detector` version purely to change an output format.

## Decision

Expose all three fields, and change nothing in `src/models/register.py`.

The premise was checked rather than assumed, three ways:

- **Live**: `mlflow.pyfunc.load_model("models:/fraud-detector@production").predict(raw)` returns
  `<class 'pandas.core.frame.DataFrame'>` with columns `['fraud_probability', 'is_fraud']`.
- **Registered signature**: `outputs: ['fraud_probability': float, 'is_fraud': long]`.
- **Source**: `register.py:260-266` builds exactly that DataFrame.

No contract change was needed, so **no model version was registered and no alias was moved
during Phase 4**. The Registry ended the phase as it ended Phase 3: three versions,
`@production` → v1 ([0018](0018-registry-snapshot-phase-3.md)).

## Alternatives considered

- **Extend `predict()` to "return probability again"** — rejected. It already does. The change
  would have registered a v4 behaviourally identical to v1, then required a manual alias move
  that bypasses the [0016](0016-promotion-quality-gate.md) gate (no `pr_auc` improvement exists
  to compare), all to reach the state the repository was already in.
- **Drop `fraud_probability`, returning label + version only** — rejected, and already rejected
  once in [0015](0015-packaged-model-contract.md). `CLAUDE.md` requires logging "input + output
  + probability + timestamp" for every prediction; a label-only response would break the Phase 4
  log and Phase 8's drift inputs at once, and would discard a number the model has already
  computed.

## Justification

A binary label with no confidence score is a real functional loss for fraud detection: it
prevents risk-tier triage and edge-case auditing, and leaves an operator no way to distinguish a
0.51 from a 0.999. The probability costs nothing — the model computes it to apply the threshold
anyway — so withholding it would be a deliberate downgrade.

`model_version` in every response is traceability: a prediction can be attributed to a concrete
artifact after the fact. Because the field name starts with `model_`, a namespace Pydantic v2
reserves, the schema sets `model_config = ConfigDict(protected_namespaces=())`.

## Trade-offs / consequences

- **The response is coupled to the packaged model's output columns.** `predict.py` names them in
  `PROBABILITY_COLUMN` / `DECISION_COLUMN` and raises `PredictionError` if they are absent, so a
  future contract change fails loudly at one place rather than as a `KeyError` inside a request.
- **`fraud_probability` is constrained `ge=0, le=1`**, which is a genuine assertion about the
  model: a pyfunc returning something outside that range would fail response validation.
