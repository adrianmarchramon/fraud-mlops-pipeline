# Decision 3: Dataset choice and known limitations

- **Date:** 2026-07-09
- **Status:** Accepted

## Context

The project needs a fraud-detection dataset that is realistic enough to justify the full
MLOps system, small enough to run on a laptop, and freely redistributable for a portfolio.
The choice was made in the pre-Phase-0 / Steps 2–3 planning and is recorded here for
traceability; it is not reopened.

## Decision

Use the **Kaggle Credit Card Fraud Detection** dataset (`mlg-ulb/creditcardfraud`),
downloaded to `data/raw/creditcard.csv` and managed by DVC (never committed to Git).

Verified in [`notebooks/01_exploration.ipynb`](../../notebooks/01_exploration.ipynb)
against the actual file:

- 284,807 transactions × 31 columns, all numeric.
- 30 features: `Time`, `Amount`, and **`V1`–`V28`** (anonymized PCA components); target `Class`.
- **Fraud rate 0.172749%** (492 frauds) — see [0001](0001-business-metric.md).
- No missing values; 1,081 exact duplicate rows.

## Alternatives considered

- **IEEE-CIS Fraud Detection** — richer and more realistic, but much larger and more complex;
  disproportionate for a project whose deliverable is the *system*, not the model.
- **PaySim (synthetic)** — good for simulating drift with control, but synthetic and less
  credible as a "real problem". Noted as a possible source for drift experiments in Phase 8.
- **A fully synthetic in-house dataset** — rejected: no external credibility.

## Justification

It is the canonical starter fraud dataset: realistically imbalanced, clean, well-known to
reviewers, and small enough for fast iteration — which fits the project's philosophy that the
model is secondary and the infrastructure is the point.

## Trade-offs / consequences — known limitations (stated openly)

- **`V1`–`V28` are anonymized PCA components.** We cannot know what they represent, so
  **interpretable, domain-driven feature engineering on them is impossible** — they are
  consumed as-is. The EDA (Section 7) can only rank them by discriminative power, not explain
  them. Acknowledging this is professional maturity, not a weakness to hide.
- **Only `Time` and `Amount` are interpretable.** `Time` is capture-relative (seconds from the
  first transaction, ~48h span) and **not directly reusable at serving time**; a cyclical
  hour-of-day feature is the interpretable option (Section 5).
- **Single, static, ~2-day capture.** Real drift cannot occur within it, so Phase 8 drift
  monitoring will need **simulated/injected drift** (a candidate use for PaySim) to demonstrate
  the closed loop.
- Class scarcity (492 frauds) makes the **19 duplicate fraud rows** non-trivial to drop —
  handled as a policy decision in the Phase 1 DVC pipeline.
