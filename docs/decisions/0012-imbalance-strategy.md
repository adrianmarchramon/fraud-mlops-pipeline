# Decision 12: Class-imbalance strategy — two levels, mutually exclusive within the first

- **Date:** 2026-07-15
- **Status:** Accepted

## Context

Fraud is **0.172749%** of the dataset ([0001](0001-business-metric.md),
[0009](0009-stratified-split.md)); the held-out test split carries **98 frauds in 56,962 rows**
(0.1720%). Left alone, a classifier trained on this is barely penalised for ignoring the positive
class entirely. Three standard corrections exist, and the roadmap names all three:

1. **Class weighting** — make each fraud count more in the loss (`class_weight="balanced"` for
   logistic regression; `scale_pos_weight` for XGBoost).
2. **Resampling** — synthesise minority examples with SMOTE until the classes balance.
3. **Threshold adjustment** — leave training alone and move the decision boundary afterwards.

The question this record settles is not *which one*, but **how they compose**.

## Decision

Treat the three as operating at **two different levels**, and combine them across levels while
keeping them exclusive within a level:

- **In-training level — class weighting and SMOTE are mutually exclusive per run.**
  `build_model()` unconditionally builds the class-weighted variant; when
  `params.yaml: train.resampling == "smote"`, `build_training_pipeline()` inserts the SMOTE step
  **and** switches weighting off (`model.set_params(class_weight=None)`), so exactly one
  correction is in force. Verified: an LR built with `class_weight="balanced"` reports
  `class_weight=None` after passing through the SMOTE branch, and keeps `balanced` under
  `resampling: none`.
- **Post-hoc level — threshold adjustment always applies on top**, whichever in-training strategy
  is active. It is orthogonal: it reshapes `predict_proba` output after the fact and cannot
  double-count anything done during `fit`. Its own value is decided in
  [0014](0014-cost-optimal-threshold.md).

The active configuration is XGBoost + `scale_pos_weight` + `resampling: none` + threshold `0.03`.

## Alternatives considered

- **Stacking class weighting and SMOTE in the same run** — rejected. It corrects the same
  imbalance twice, over-representing the minority relative to either technique alone, and it
  destroys attribution: a run with both moving cannot tell you which one moved the metric. Since
  the entire point of the phase is traceability, a configuration whose effect cannot be
  attributed is a configuration worth refusing. Isolating one per run is what makes the MLflow
  comparison mean anything.
- **Resampling only, dropping class weights entirely** — rejected as the default: SMOTE
  synthesises data that was never observed, which on 30 PCA-projected features
  ([0003](0003-dataset-choice.md)) is a real modelling assumption. Class weighting invents
  nothing. SMOTE stays available behind a parameter rather than becoming the default.
- **Doing nothing in training and relying on the threshold alone** — rejected: a model trained
  without any imbalance correction optimises a loss that is ~99.83% satisfied by predicting
  "legitimate" always. The threshold can only reorder confidences the model already produces; it
  cannot install a signal the training never learned.

## Justification

Measured effect of each level, from the runs in the `fraud-detection` experiment (PR-AUC, the
primary metric per [0001](0001-business-metric.md)):

| Run | In-training strategy | PR-AUC |
|-----|----------------------|--------|
| `db6ac47e` | logistic_regression + class weighting | 0.7159 |
| `9a897904` | logistic_regression + SMOTE | 0.7249 |
| `4d11e41b` | xgboost + `scale_pos_weight` | **0.8760** |

And of the post-hoc level, on the winning model (test split, costs from
[0002](0002-cost-asymmetry.md)): threshold 0.5 → 12 FP / 16 FN → **€2,240**; threshold 0.03 →
70 FP / 11 FN → **€1,787**. The in-training correction decides how good the ranking is; the
threshold decides where to cut it. Neither substitutes for the other, which is precisely why the
levels compose rather than compete.

## Trade-offs / consequences

- **The exclusivity toggle is logistic-regression-specific, and silently so.** XGBoost has no
  `class_weight` parameter at all — verified: `set_params(class_weight=None)` on an
  `XGBClassifier` is **accepted without error and does nothing**, leaving `scale_pos_weight=577.29`
  fully active. So pairing `model: xgboost` with `resampling: smote` **double-corrects**
  (`scale_pos_weight` *and* SMOTE) — exactly the configuration this record rejects, reached
  silently and with no warning. **XGBoost must pair with `resampling: none`**, and it does; this
  is why activating XGBoost in Step 9 flipped `resampling` in the same change.
- The robust fix is to generalise the toggle per estimator (switch off whichever imbalance knob
  the classifier actually exposes) instead of hardcoding `class_weight`. Deferred rather than
  hidden: it requires reopening closed Step 5-6 logic, and the guard-rail today is a comment in
  `params.yaml` plus this record. It is a latent trap for whoever adds a **third** algorithm.
- SMOTE remains reachable by parameter but is **not** on the winning path, so it is the less
  exercised branch. Its leakage safety does not depend on vigilance: imblearn's `Pipeline` applies
  resampling only during `fit`, never during `predict`/`predict_proba`, which is what confines it
  to the training fold under cross-validation too ([0006](0006-no-data-leakage.md)).
