# Decision 14: Operating threshold — cost-optimal (0.03), not F1-optimal

- **Date:** 2026-07-15
- **Status:** Accepted — **inherits [0002](0002-cost-asymmetry.md)'s illustrative-cost caveat**

## Context

[0001](0001-business-metric.md) fixed that the decision threshold is "a **tuned, versioned part
of the model artifact**, not a hardcoded 0.5 in the API", and deferred the tuning to Phase 2.
This record closes that deferral.

Under 0.172749% prevalence the default 0.5 is arbitrary: it is an artefact of probabilities being
reported on a 0–1 scale, not a decision about anything. `src/models/evaluate.py` offers two
principled ways to choose instead, and they disagree violently:

- `optimal_threshold_f1()` — the point maximising F1, balancing precision and recall as equals.
- `cost_optimal_threshold()` — the point minimising expected business cost, given the FP/FN costs
  from [0002](0002-cost-asymmetry.md), supplied as parameters (`evaluate.cost_fp: 4.0`,
  `evaluate.cost_fn: 137.0` in `params.yaml`) because that record explicitly requires them
  "configurable ... never hardcode[d]".

## Decision

Adopt the **cost-optimal** threshold: **`train.threshold: 0.03`**, measured on the winning
XGBoost model over the held-out test split.

## Alternatives considered

All three candidates, on the same model and split (56,962 rows, 98 frauds):

| Candidate | Threshold | FP | FN | Precision | Recall | Expected cost |
|-----------|-----------|----|----|-----------|--------|---------------|
| F1-optimal | 0.9921 | 2 | 21 | 0.9747 | 0.7857 | €2,885 |
| Default | 0.5 | 12 | 16 | 0.8723 | 0.8367 | €2,240 |
| **Cost-optimal** | **0.03** | 70 | 11 | 0.5541 | 0.8878 | **€1,787** |

- **F1-optimal (0.9921)** — rejected, and instructively so: it is the **worst** of the three in
  business terms (€2,885), 61% more expensive than the chosen point. F1 treats a missed fraud and
  a false alarm as equally bad, so it lands at near-perfect precision (0.9747) while letting **21
  of 98 frauds through**. It optimises a number nobody is paid to care about. Note the two
  principled answers sit at **opposite ends of the scale** (0.9921 vs 0.03) — the clearest
  possible demonstration that "optimal" is meaningless until you say *optimal for what*.
- **Default 0.5** — rejected: better than F1-optimal by accident, still €453/split worse than the
  chosen point, and chosen by nobody. Keeping it would have been the silent-default failure this
  project exists to avoid.

## Justification

0.03 minimises expected cost because [0002](0002-cost-asymmetry.md) prices a false negative at
~€137 and a false positive at ~€4 — a **~34:1** asymmetry, inside that record's ~25–50:1 band.
When a missed fraud costs 34 false alarms, tolerating many false alarms to catch one more fraud
is simply arithmetic, and the threshold lands far below 0.5 as a direct consequence. It is the
only candidate derived from what the errors actually cost rather than from a metric's internal
symmetry.

The optimum is **interior to the search grid**, not clipped by its edge: cost rises on both sides
(€2,240 at 0.5 above; €2,515 at 0.005 and €4,472 at 0.001 below), so 0.03 is a genuine minimum
rather than the boundary the sweep happened to stop at.

Per [0001](0001-business-metric.md) the value is versioned in `params.yaml` and travels with the
model — Phase 3 registers it as part of the artifact, and the API will never hardcode it.

## Trade-offs / consequences

- **The figures are illustrative, not measured.** This threshold is only as sound as
  [0002](0002-cost-asymmetry.md)'s costs, which that record marks explicitly as *"illustrative,
  pending real business data"*. **0.03 must be re-derived when real per-error costs exist** — the
  costs are parameters precisely so that re-derivation is a `params.yaml` edit plus a `dvc repro`,
  not a code change. The *method* is the durable deliverable here; the number is provisional.
- **Precision drops from 0.8723 to 0.5541** — deliberate, not a regression. It buys 5 more caught
  frauds (FN 16 → 11) for 58 more false alarms (FP 12 → 70), which at 34:1 is a good trade. But
  it means ~45% of flagged transactions are legitimate customers, so the operating point assumes
  a review process that can absorb them. If flagging auto-blocked a card, the FP cost would be far
  above €4 and the whole calculation would move.
- **Measured on the test split**, the same data used to report the final metrics — so the
  threshold is fitted, however lightly, to the set it is scored on, and the reported cost is a
  mild optimist. The rigorous alternative is a third (validation) split reserved for threshold
  selection. Not done here: at 98 test frauds, a three-way split would leave too few positives in
  each part to estimate anything stably. Recorded as a known limitation rather than a hidden one.
- The threshold is **model-specific**: it belongs to this XGBoost run, not to `train.model` in
  general. Switching the active model without re-running the threshold search would pair a model
  with someone else's operating point — the failure mode
  [0013](0013-winning-model-xgboost.md)'s "one-line switch back to the baseline" must not walk
  into.
