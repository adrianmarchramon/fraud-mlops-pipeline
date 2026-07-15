# Decision 13: Winning model — XGBoost over the logistic-regression baseline

- **Date:** 2026-07-15
- **Status:** Accepted

## Context

Phase 2 trained two algorithms against the same versioned splits so they could be compared like
for like: a **logistic regression** baseline (with and without SMOTE) and **XGBoost**. One has to
be promoted into `params.yaml: train.model` as the model the pipeline actually builds, and Phase 3
will register it. Per [0001](0001-business-metric.md) the deciding metric is **PR-AUC**, never
ROC-AUC, and recall is prioritised while controlling precision.

The comparison is not the formality it first appears, because the two models disagree about
which one is better depending on which number you read.

## Decision

Promote **XGBoost** (`train.model: xgboost`, `resampling: none`, `scale_pos_weight: 577.29`).

## Alternatives considered

The full run history of the `fraud-detection` experiment, sorted by PR-AUC:

| Run | Model | Resampling | PR-AUC | Precision @0.5 | Recall @0.5 |
|-----|-------|-----------|--------|----------------|-------------|
| `4d11e41b` | **xgboost** | none | **0.8760** | 0.8723 | 0.8367 |
| `f97d88a7` | xgboost (`n_estimators=200`) | none | 0.8655 | 0.2949 | 0.8878 |
| `9a897904` | logistic_regression | smote | 0.7249 | 0.0580 | 0.9184 |
| `db6ac47e` | logistic_regression | none | 0.7159 | 0.0610 | 0.9184 |

- **Logistic regression + SMOTE** — the interesting rejection, because **it wins on recall**
  (0.9184 vs 0.8367) and recall is the metric this project says it prioritises. Rejected anyway:
  it loses PR-AUC by 0.15, and its recall is bought at a price the confusion matrix makes
  obvious (below).
- **Logistic regression without SMOTE** — rejected: dominated by the SMOTE variant on PR-AUC
  (0.7159 vs 0.7249) and no better on recall.

## Justification

XGBoost wins the primary metric outright: **PR-AUC 0.8760 vs 0.7249**, a 0.15 margin that is not
a rounding artefact.

The recall contradiction resolves once the errors are counted rather than rated. On the test
split (56,962 rows, 98 frauds), at the default threshold:

| Model | FP | FN | Cost @ [0002](0002-cost-asymmetry.md) |
|-------|----|----|------|
| xgboost | **12** | 16 | **€2,240** |
| logistic_regression + SMOTE | **1,461** | 8 | €6,940 |

Logistic regression catches **8 more frauds** by raising **122× more false alarms**. Its 0.0580
precision means roughly **17 of every 18 transactions it flags are legitimate customers** — an
alert stream no fraud team would operate. Priced with the illustrative costs from
[0002](0002-cost-asymmetry.md), its extra recall costs **3.1× more** than XGBoost's.

The decisive check is whether the comparison is merely an artefact of both models sitting at an
arbitrary 0.5. It is not — **matched on recall, XGBoost still dominates**. Pushing XGBoost's
threshold down to 0.001 reproduces logistic regression's exact 0.9184 recall with **844 false
positives and €4,472**, against LR's **1,461 and €6,940**. There is no operating point at which
the baseline's recall advantage is real: XGBoost buys the same recall more cheaply, so nothing is
sacrificed by promoting it.

## Trade-offs / consequences

- **Interpretability is traded for performance.** Logistic regression offers per-feature
  coefficients; XGBoost is an ensemble whose reasoning needs SHAP or similar to explain. In a
  domain where a declined transaction may require justification to a customer or regulator, this
  is a real cost, accepted here because the gap in error counts (12 vs 1,461 false alarms) is too
  large to trade away for legibility.
- **The baseline is not deleted.** `build_model()` keeps its `logistic_regression` branch and
  `params.yaml` keeps its hyperparameters, so the comparison stays reproducible and the baseline
  remains available as a sanity check — switching back is a one-line change to `train.model`.
- **XGBoost's recall ceiling is lower.** Over the searched threshold grid (0.01–0.99) its maximum
  recall is **0.8878**; reaching 0.9184 requires going *below* the grid, to 0.001. If a future
  requirement demanded ~0.92 recall regardless of cost, this model reaches it only at a threshold
  outside the range [0014](0014-cost-optimal-threshold.md) searches — the model choice and the
  threshold search space would need revisiting together.
- The pairing with `resampling: none` is **mandatory, not incidental**: XGBoost silently ignores
  `class_weight`, so `resampling: smote` would double-correct. See
  [0012](0012-imbalance-strategy.md).
- Consistent with the project's thesis, this decision is deliberately **not** the point of the
  phase: the model is the least important artefact here, and the comparison above matters
  chiefly because it is *reproducible and traceable* — every figure re-derived from the tracked
  runs, not from a remembered result.
