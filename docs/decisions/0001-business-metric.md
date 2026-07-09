# Decision 1: Business metric — prioritize recall, report PR-AUC over ROC-AUC

- **Date:** 2026-07-09
- **Status:** Accepted

## Context

The dataset is extremely imbalanced. Measured in
[`notebooks/01_exploration.ipynb`](../../notebooks/01_exploration.ipynb) (Section 3,
executed against `data/raw/creditcard.csv`):

- 284,807 transactions, of which **492 are fraud → a fraud rate of 0.172749%** (≈ 1 : 578).

At this prevalence, a trivial "predict everything legitimate" classifier scores **99.83%
accuracy** while catching zero fraud. Accuracy is therefore useless, and we must choose a
metric that reflects performance on the rare positive class.

## Decision

- **Primary metric: PR-AUC** (area under the precision–recall curve).
- **Optimize for high recall** (catch as much fraud as possible) while **controlling
  precision** to keep false alarms manageable.
- The **decision threshold is a tuned, versioned part of the model artifact**, not a
  hardcoded 0.5 in the API. Threshold tuning is Phase 2 work; this record fixes the
  *metric* the tuning optimizes against.

## Alternatives considered

- **Accuracy** — rejected: 99.83% is achievable by predicting no fraud at all.
- **ROC-AUC** — rejected as the *primary* metric: under severe imbalance the false-positive
  rate is diluted by the huge negative class, so ROC-AUC looks deceptively optimistic. It
  may be reported as a secondary number for comparability, but decisions are made on PR-AUC.
- **F1 at the default 0.5 threshold** — rejected as an optimization target: it hides the
  precision/recall trade-off that is the whole point here, and 0.5 is meaningless at this
  prevalence.

## Justification

The precision–recall curve only involves the positive class (precision and recall), so it
degrades honestly when the rare class is predicted poorly — exactly the failure mode we
care about. Prioritizing recall follows directly from the cost structure (see
[0002](0002-cost-asymmetry.md)): a missed fraud is far costlier than a false alarm.

## Trade-offs / consequences

- Pushing recall up will cost precision; the acceptable balance is a **business** decision,
  formalized quantitatively in [0002](0002-cost-asymmetry.md) and turned into a concrete
  operating threshold in Phase 2.
- Reviewers used to accuracy/ROC-AUC will see lower headline numbers; the README must
  explain *why* PR-AUC is the honest choice here.
- CI's model-validation gate (Phase 6) must assert a **minimum PR-AUC**, not accuracy.
