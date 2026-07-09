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

> These records feed the "Design decisions" section of the README (Phase 0, Step 10 — out
> of scope here). They are the articulable rationale behind the system, kept in writing so
> the project does not contradict itself in later phases.
