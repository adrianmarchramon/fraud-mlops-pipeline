# Decision 6: No data leakage — fit the scaler on the training split only

- **Date:** 2026-07-12
- **Status:** Accepted

## Context

Preprocessing standardizes `Time` and `Amount` (see
[0007](0007-scale-time-amount-only.md)). Standardization needs a mean and standard deviation;
**where** those statistics are computed determines whether test-set information leaks into
training. Fitting the scaler on the full dataset before splitting is the single most common way
to leak, and it produces misleadingly optimistic metrics that collapse in production.

## Decision

Split first, then **fit the scaler exclusively on the training split** and only ever *apply*
(never re-fit) it to the test split. Evidence in `src/data/preprocess.py`:

- line 118 — `train_test_split(...)` runs **before** any scaling;
- line 130 — `X_train_t = preprocessor.fit_transform(X_train)` (fit + transform on train);
- line 131 — `X_test_t = preprocessor.transform(X_test)` (transform only — no re-fit).

## Alternatives considered

- **Fit on the full population, then split** — rejected: the scaler's mean/std already encode
  test-set information → textbook leakage.
- **Fit a separate scaler per split** — rejected: train and test would be transformed by
  different mappings, so the model would see inconsistent feature spaces and the test set would
  no longer be a faithful stand-in for unseen data.

## Justification

The test split must be a proxy for genuinely unseen data, so nothing about it may influence how
features are transformed. Fitting only on train and transforming test is the correct, standard
discipline — and it is verified end-to-end: both splits are re-checked against
`processed_schema` (`validate_processed_data`) before being written.

## Trade-offs / consequences

- The fitted transformer's state must now be **persisted and shipped** so inference applies the
  identical mapping — addressed by [0008](0008-preprocessor-persistence.md).
- Marginally more code than a naive "scale everything up front", in exchange for metrics that
  are honest.
