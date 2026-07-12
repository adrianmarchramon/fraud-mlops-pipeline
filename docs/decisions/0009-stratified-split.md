# Decision 9: Stratified train/test split on the target

- **Date:** 2026-07-12
- **Status:** Accepted

## Context

The dataset is extremely imbalanced — fraud rate **0.172749%** (492 of 284,807), as measured in
the EDA and recorded in [0001](0001-business-metric.md) and [0003](0003-dataset-choice.md). With
so few positives, an ordinary random split can, by chance, leave the test set with very few — or
even zero — fraud cases, making evaluation meaningless.

Evidence in repo: `src/data/preprocess.py` line 118 calls `train_test_split(X, y, ...,
stratify=y)`; the accompanying comment (line 117) states the intent.

## Decision

Split with **`stratify=y`**, preserving the fraud-to-legitimate ratio in both the training and
test splits. `test_size` (0.2) and `random_state` (42) are versioned in `params.yaml`, so the
split is reproducible and DVC-tracked.

## Alternatives considered

- **Plain random split** — rejected: at this prevalence it risks a test set with too few or no
  frauds, invalidating recall/PR-AUC estimates (the metrics that matter, per
  [0001](0001-business-metric.md)).
- **Temporal split** (train on earlier `Time`, test on later) — acknowledged as *more realistic*
  for fraud ("train on the past, predict the future") and a better fit for the drift narrative,
  but deferred: the dataset is a single static ~2-day capture ([0003](0003-dataset-choice.md)),
  so a temporal split buys little here and is noted as a future variant / stretch goal.

## Justification

Stratification is the minimal, standard safeguard that makes evaluation trustworthy under severe
imbalance, with a fixed `random_state` guaranteeing the same split every run (confirmed: reverting
a `test_size` change and re-running `dvc repro` restores byte-identical outputs).

## Trade-offs / consequences

- The split is **not temporally realistic**; the temporal variant is recorded as a deliberate
  future improvement rather than an oversight, and connects to the Phase 8 drift work.
- Fixing `random_state` trades split diversity for reproducibility — the right trade for a
  versioned pipeline.
