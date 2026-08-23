# Design Decision Records (ADRs)

Lightweight architecture/design decision records for the fraud-detection MLOps project.
One file per decision, in the format `NNNN-title.md`.

Each record follows the same template: **Context → Decision → Alternatives considered →
Justification → Trade-offs / consequences**. Every quantitative claim is traceable to a
source — for records 0001–0010, most often a cell in
[`notebooks/01_exploration.ipynb`](../../notebooks/01_exploration.ipynb) (Phase 0, Step 8),
executed against `data/raw/creditcard.csv`; for 0011–0014, a tracked run in the MLflow
`fraud-detection` experiment, identified by run id and re-derivable from it; for
0015–0018, the registered versions of `fraud-detector` and the commands that produced
them, quoted in each record; for 0019–0021, live HTTP calls against a running API and the
records it wrote, quoted in each record; and for 0022–0025, the built image, the running
containers and the HTTP calls served from inside them, quoted in each record.

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
| [0011](0011-mlflow-sqlite-backend.md) | MLflow tracking backend: SQLite from day one | Accepted |
| [0012](0012-imbalance-strategy.md) | Class-imbalance strategy: two levels, exclusive within the first | Accepted |
| [0013](0013-winning-model-xgboost.md) | Winning model: XGBoost over the logistic-regression baseline | Accepted |
| [0014](0014-cost-optimal-threshold.md) | Operating threshold: cost-optimal (0.03), not F1-optimal | Accepted (inherits 0002's illustrative caveat) |
| [0015](0015-packaged-model-contract.md) | Packaged model contract: artifacts injection, raw in, probability + decision out | Accepted |
| [0016](0016-promotion-quality-gate.md) | Promotion quality gate: PR-AUC alone, `@production` alone, strictly better | Accepted (inherits 0002's illustrative caveat) |
| [0017](0017-registry-testing-and-visibility.md) | Testing the Registry with hand-written doubles; evidencing it through the API | Accepted |
| [0018](0018-registry-snapshot-phase-3.md) | Registry state at Phase 3 closure (dated snapshot) | Accepted (point-in-time) |
| [0019](0019-api-response-contract.md) | API response contract: probability alongside the decision | Accepted |
| [0020](0020-api-structure-and-state.md) | API structure: inference in `predict.py`, model in `app.state` | Accepted |
| [0021](0021-prediction-log-and-api-tests.md) | Prediction log as Phase 8's contract; API testing strategy | Accepted |
| [0022](0022-container-image-contract.md) | Container image: Python 3.12, non-root, body-matching health check | Accepted |
| [0023](0023-compose-topology.md) | Compose topology: two services, readiness gate, MLflow Host allowlist | Accepted |
| [0024](0024-environment-based-tracking-uri.md) | MLflow address from the environment, with a working local default | Accepted |
| [0025](0025-phase-5-reproducibility-and-key-test-scope.md) | Phase 5 verification strategy and the honest scope of the "key test" | Accepted |

Records **0001–0004** were established in Phase 0 (foundations); **0005–0010** in Phase 1
(data pipeline and versioning); **0011–0014** in Phase 2 (training and experiment tracking);
**0015–0018** in Phase 3 (model registry and packaging); **0019–0021** in Phase 4 (inference
API); and **0022–0025** in Phase 5 (containerization).

> These records feed the "Design decisions" section of the README (Phase 0, Step 10 — out
> of scope here). They are the articulable rationale behind the system, kept in writing so
> the project does not contradict itself in later phases.
