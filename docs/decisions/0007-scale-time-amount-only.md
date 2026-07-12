# Decision 7: Scale only `Time` and `Amount`, pass `V1`–`V28` through untouched

- **Date:** 2026-07-12
- **Status:** Accepted

## Context

The dataset's 30 features are `Time`, `Amount`, and `V1`–`V28`. As recorded in
[0003](0003-dataset-choice.md), `V1`–`V28` are **anonymized PCA components**; only `Time` and
`Amount` remain on their original, raw scales. A `ColumnTransformer` must be told which columns
to standardize.

Evidence in repo: `params.yaml` → `scale_columns: [Time, Amount]`; `src/data/preprocess.py`
`build_preprocessor()` applies `StandardScaler()` to `scale_columns` with
`remainder="passthrough"`, so every other column (the `V*` set) flows through unchanged.

## Decision

Standardize **only `Time` and `Amount`**; let `V1`–`V28` pass through without scaling. The
choice lives in the versioned `params.yaml`, not hardcoded, so it is tracked and DVC-aware.

## Alternatives considered

- **Scale all 30 features** — rejected: `V1`–`V28` are already outputs of a PCA (approximately
  zero-mean, comparable variance); re-standardizing them is redundant and can distort their
  relative magnitudes for no benefit.
- **Scale nothing** — rejected: `Time` (seconds, ~0–172,792) and `Amount` (unbounded currency)
  are on scales that would dominate any distance- or gradient-based model, drowning the `V*`
  signal.

## Justification

Scaling is applied exactly where it helps — the two raw-scale, interpretable columns — and
withheld where it would only add noise. Documenting *why* some columns are scaled and others are
not is the reasoning expected of someone who understands the data rather than applying a rote
recipe.

## Trade-offs / consequences

- This relies on the **assumption** that the source PCA was fit on standardized inputs, so its
  components are already well-scaled. The assumption is reasonable for this dataset but is stated
  openly rather than hidden.
- `Time` is capture-relative and not directly reusable at serving time (see the limitation noted
  in [0003](0003-dataset-choice.md)); scaling it does not change that caveat, which a later phase
  may address with a cyclical time feature.
