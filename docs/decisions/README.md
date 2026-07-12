# Design Decision Records (ADRs)

Lightweight architecture/design decision records for the fraud-detection MLOps project.
One file per decision, in the format `NNNN-title.md`.

Each record follows the same template: **Context → Decision → Alternatives considered →
Justification → Trade-offs / consequences**. Every quantitative claim is traceable to a
source — most often a cell in [`notebooks/01_exploration.ipynb`](../../notebooks/01_exploration.ipynb)
(Phase 0, Step 8), executed against `data/raw/creditcard.csv`.

| # | Decision | Status |
|---|----------|--------|
| [0001](0001-business-metric.md) | Business metric: PR-AUC + recall priority | Accepted |
| [0002](0002-cost-asymmetry.md) | False-negative vs false-positive cost asymmetry | Accepted (figures illustrative, pending real business data) |
| [0003](0003-dataset-choice.md) | Dataset choice and known limitations | Accepted |
| [0004](0004-stack-summary.md) | Technology stack summary | Accepted |
| [0005](0005-dvc-local-remote.md) | DVC remote: local filesystem store (interim) | Accepted (interim) |
| [0006](0006-no-data-leakage.md) | No data leakage: fit scaler on train split only | Accepted |
| [0007](0007-scale-time-amount-only.md) | Scale only `Time` and `Amount`, pass `V1`–`V28` through | Accepted |
| [0008](0008-preprocessor-persistence.md) | Persist the fitted preprocessor (avoid training-serving skew) | Accepted |
| [0009](0009-stratified-split.md) | Stratified train/test split on the target | Accepted |
| [0010](0010-pandera-strict-lazy.md) | Pandera contract: `strict=True` + `lazy=True` | Accepted |

Records **0001–0004** were established in Phase 0 (foundations); **0005–0010** in Phase 1
(data pipeline and versioning).

> These records feed the "Design decisions" section of the README (Phase 0, Step 10 — out
> of scope here). They are the articulable rationale behind the system, kept in writing so
> the project does not contradict itself in later phases.
