# Fraud MLOps Pipeline

![CI](https://github.com/adrianmarchramon/fraud-mlops-pipeline/actions/workflows/ci.yml/badge.svg)

**An end-to-end MLOps system for credit-card fraud detection that retrains itself when the world changes.**

This repository is not a model in a notebook — it is the *production system* around the model:
versioned data, tracked training, a model registry, a typed inference API, containers, CI/CD,
orchestration, and drift monitoring wired into a closed loop that **detects data drift and
triggers automatic retraining**. The fraud model is deliberately the least important part; the
engineering around it is the deliverable.

> **Project status:** 🟢 **Phases 0–5 complete** — foundations, a versioned and validated data
> pipeline (DVC + Pandera), tracked training on top of it, models managed as production
> artifacts rather than loose experiments (packaged with their preprocessor and decision
> threshold, registered in the MLflow Model Registry, promoted to `@production` only when they
> beat the model already there), **a typed REST API serving that model over HTTP** with
> self-generated interactive docs, and **the whole system packaged as containers** — API and
> MLflow brought up together by a single `docker compose up`, so it behaves the same on any
> machine. Phases 6–9 are under active construction — see the [roadmap](#roadmap) below. This is
> a living project, built in gated phases, not an abandoned prototype.

---

## Architecture

The system runs on two clocks over the same infrastructure: a fast **prediction path**
(validate → preprocess → predict → **log every prediction** → respond, in milliseconds) and a
slow **model-lifecycle path** (logs analysed for drift → alert → automatic retrain → evaluate →
promote, over days/weeks). The closed drift → retrain loop is the centrepiece.

![Architecture of the closed-loop fraud MLOps system](docs/images/architecture.png)

The same flow as a quick textual reference:

```
 raw data ─▶ ingest + validate (DVC, Pandera) ─▶ preprocess ─▶ train (MLflow) ─▶ Model Registry
                    ▲                                                                   │
                    │                                                                   ▼
              retrain (Prefect)                                                 FastAPI inference
                    ▲                                                                   │
                    │                                                          log every prediction
                    └────────── drift alert (Evidently) ◀── analyse prediction logs ◀───┘
```

---

## The problem — and why it matters

Card fraud is a **needle-in-a-haystack** problem: in this project's dataset, roughly **0.17% of
transactions are fraudulent** (492 in 284,807). That extreme imbalance makes accuracy a trap — a
model that flags *nothing* as fraud is still ~99.83% accurate and completely useless.

What makes it a *business* problem, not just a statistics problem, is the **cost asymmetry**
between the two ways of being wrong:

- A **false negative** (fraud let through) is direct, often unrecoverable financial loss.
- A **false positive** (a legitimate customer wrongly blocked) is friction, support cost, and
  eroded goodwill.

Because a missed fraud typically costs far more than a false alarm, the system is tuned to
**prioritise recall while controlling precision**, and is measured with **PR-AUC** rather than
the deceptively optimistic ROC-AUC. The decision threshold is treated as part of the model
artifact, tuned to the business cost trade-off rather than left at a naïve 0.5.

The full reasoning — metric choice, cost model, dataset limitations, and stack rationale — lives
in the design-decision records: **[`docs/decisions/`](docs/decisions/)**.

---

## Technology stack

Chosen for a reproducible, production-shaped system at portfolio scale — no Kubernetes, no
over-engineering. The **Phase** column shows when each tool enters the project; "✅ active" means
it is already wired up in the repository today (through Phase 5).

| Concern | Tool | Phase |
|---|---|---|
| Language | Python 3.12 | ✅ active |
| Environment & dependencies | **uv** (single lockfile, no manual venvs) | ✅ active |
| Lint **and** format | **ruff** (one tool — no black) | ✅ active |
| Git hooks | **pre-commit** (ruff, whitespace, large-file guard) | ✅ active |
| Testing | **pytest** | ✅ active |
| Exploration | **pandas · seaborn · matplotlib · Jupyter** | ✅ active |
| Data versioning | **DVC** (data never committed to Git) | ✅ active |
| Data validation | **Pandera** (schema as a quality contract) | ✅ active |
| Modelling | **scikit-learn / XGBoost** + **imbalanced-learn** | ✅ active |
| Experiment tracking | **MLflow** (SQLite backend) | ✅ active |
| Model Registry | **MLflow Model Registry** (versions + aliases) | ✅ active |
| Inference API | **FastAPI** + **Pydantic** + **Uvicorn** | ✅ active |
| Containerization | **Docker** (multi-stage, non-root) + Docker Compose | ✅ active |
| CI/CD | **GitHub Actions** (incl. a model-validation gate) | Phase 6 |
| Orchestration | **Prefect** | Phase 7 |
| Monitoring & drift | **Evidently** | Phase 8 |
| Deployment | **Render / Railway / Fly.io / Modal** (lightweight) | Phase 9 |

---

## Getting started

**Prerequisites:** [`uv`](https://docs.astral.sh/uv/) installed. uv manages the Python
interpreter (pinned to 3.12 in `.python-version`), the virtual environment, and all
dependencies — you do not need to create or activate a venv yourself.

```bash
git clone https://github.com/adrianmarchramon/fraud-mlops-pipeline.git
cd fraud-mlops-pipeline
make setup          # uv sync + install pre-commit hooks
```

`make setup` is the single entry point: it resolves the exact dependency set from `uv.lock` and
installs the git hooks. The rest of the interface is the Makefile:

```bash
make lint           # ruff check
make format         # ruff format
make test           # pytest
make train          # train the model, logging the run to MLflow
make register       # register the best run, promote it if it beats production
make serve          # run the FastAPI inference API on http://localhost:8000
```

### Inspecting the experiments

Every training run — its parameters, metrics, confusion matrix and PR curve — is tracked in
MLflow, backed by a local SQLite store. To browse the run history and compare experiments side
by side:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Runs are best sorted by **PR-AUC**, the primary metric fixed in
[`docs/decisions/0001-business-metric.md`](docs/decisions/0001-business-metric.md). The tracking
store is local and git-ignored, so a fresh clone starts with an empty history until you run
`make train` or `uv run dvc repro`.

### Using the production model

`make register` takes the best run in the experiment, packages it with the fitted preprocessor
and the versioned decision threshold into a single artifact that accepts **raw** transactions,
registers it as a new version of `fraud-detector`, and moves the `@production` alias to it —
but only if its PR-AUC beats the version already holding that alias. An inferior model is
registered and left unpromoted rather than silently shipped.

Consumers never name a version. They ask for the role:

```python
from src.models.register import load_production_model

model = load_production_model()          # models:/fraud-detector@production
predictions = model.predict(raw_transactions)   # fraud_probability + is_fraud
```

Because preprocessing and the threshold travel inside the artifact, the caller feeds raw
transactions straight in — no reimplemented feature engineering, and therefore no
*training-serving skew*. Promoting a new version changes what that call returns without
changing a line of the code that makes it, which is exactly what the inference API relies on.

### Serving the model over HTTP

```bash
make serve          # uvicorn on http://localhost:8000
```

The API loads the `@production` model **once at startup** and holds it in memory, so it never
names a version: promote a new one in the Registry, restart the service, and it serves the new
model with no code change. Three endpoints:

| Endpoint | Purpose |
|---|---|
| `POST /predict` | score one raw transaction |
| `GET /health` | liveness + whether a model is loaded (used by the container health check) |
| `GET /model-info` | which registered model, version and alias are being served |

Open **<http://localhost:8000/docs>** and the interactive Swagger UI lets you send a transaction
and read the prediction straight from the browser — no client code. (A reference-style view
lives at `/redoc`.) Both are generated from the Pydantic schemas, so the documentation cannot
drift from what the API actually accepts.

A request carries the 30 raw dataset columns — `Time`, `Amount` and the anonymised PCA
components `V1`…`V28`, all required:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 0.0, "Amount": 149.62, "V1": -1.359807, "V2": -0.072781, ..., "V28": -0.021053}'
```

```json
{"fraud_probability": 0.000042796866182470694, "is_fraud": 0, "model_version": "1"}
```

Values are illustrative: `model_version` reports whichever version currently holds
`@production`, so it changes as models are promoted.

The response carries the probability alongside the decision: the label drives the action, the
probability supports triage and auditing. Anything malformed — a missing field, a negative
`Amount`, a string where a number belongs — is rejected with a **422** by Pydantic before it
reaches the model.

Every served prediction is appended to `logs/predictions.jsonl` (input, output and a UTC
timestamp, one JSON object per line). That file is git-ignored runtime output, and it is the
raw material Phase 8 consumes to detect drift.

### Running the whole system in containers

From Phase 5 the API and MLflow run as containers, so the system behaves identically on any
machine that has Docker — no Python, no uv, no local dependency set:

```bash
docker compose -f docker/docker-compose.yml up
```

That brings up two services on an internal Compose network:

| Service | Address | What it is |
|---|---|---|
| `api` | <http://localhost:8000> | the FastAPI inference service — `/docs` for the Swagger UI |
| `mlflow` | <http://localhost:5000> | the tracking server and Model Registry |

The API locates the registry by **service name** (`http://mlflow:5000`), resolved by Compose's
internal DNS — no IP address is hardcoded anywhere — and waits for MLflow to report *healthy*
before starting, because it resolves the `@production` alias once at startup and never retries.
Registered models live in a named Docker volume, so they outlive the containers and survive
`docker compose down`.

The image itself is multi-stage: dependencies resolve in a builder stage that never reaches the
final image, which runs as a non-root user and carries a health check that inspects the
*response body* of `/health` — `/health` returns `200` even with no model loaded, so checking
only the status code would report a container that never reached MLflow as healthy.

Once it is up, the API is used exactly as in the previous section:

```bash
curl http://localhost:8000/health          # {"status":"ok"}
curl http://localhost:8000/model-info      # {"model_name":"fraud-detector","version":"1","alias":"production"}
```

> **One-time bootstrap.** The registry volume starts **empty** on a fresh clone, so the first
> `docker compose up` gives you a healthy MLflow and an API honestly reporting
> `{"status":"no_model"}`. Seeding it is the one step that still needs Python, because the model
> is deliberately *not* baked into the image — that is what lets the same image serve whatever
> version currently holds `@production`:
>
> ```bash
> docker compose -f docker/docker-compose.yml up -d mlflow
> MLFLOW_TRACKING_URI=http://localhost:5000 make train
> MLFLOW_TRACKING_URI=http://localhost:5000 make register
> docker compose -f docker/docker-compose.yml up -d
> ```
>
> The same `train.py` and `register.py` populate either registry without a line changing — only
> the environment variable differs. After this, the volume keeps the model and a plain
> `docker compose up` is all that is needed. The reasoning is in
> [`docs/decisions/0025-phase-5-reproducibility-and-key-test-scope.md`](docs/decisions/0025-phase-5-reproducibility-and-key-test-scope.md).

### Getting the data

`make setup` prepares the *environment* but does **not** fetch the dataset — data is never
stored in Git. From Phase 1 the data is **DVC-managed**: `data/raw/creditcard.csv` and the
processed artifacts are tracked by DVC, with only the lightweight `.dvc` / `dvc.lock` pointers
committed to Git.

If you have access to the configured DVC remote, pull the tracked data — raw CSV, processed
`train`/`test` parquet, and the fitted `preprocessor.joblib` — in one step:

```bash
uv run dvc pull
```

> **Note:** the default DVC remote is currently a **local** store on the author's machine — a
> deliberate Phase 1 choice (see
> [`docs/decisions/0005-dvc-local-remote.md`](docs/decisions/0005-dvc-local-remote.md)). A clone
> on a different machine therefore cannot `dvc pull` until a shared/cloud remote is configured.

To reproduce the pipeline from scratch on any machine, fetch the raw CSV from
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (with configured Kaggle API
credentials) and rebuild the versioned pipeline:

```bash
uv run kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
uv run dvc repro
```

This places `creditcard.csv` in `data/raw/`, runs the versioned pipeline
(validate → preprocess → train), regenerates `data/processed/{train,test}.parquet` and
`preprocessor.joblib`, trains the model, and writes `reports/metrics.json`. Since Phase 2 the
pipeline reaches all the way to the model: `dvc repro` re-runs only the stages whose
dependencies actually changed, so editing a hyperparameter in `params.yaml` retrains the model
without recomputing the data. The exploration notebook `notebooks/01_exploration.ipynb` also
runs end to end once the raw CSV is present.

---

## Roadmap

The project is built in nine gated phases; each is finished only when its "Definition of Done"
passes before the next begins.

| Phase | Milestone | Status |
|---|---|---|
| 0 | Repo, environment, data understanding & decision log | ✅ Complete |
| 1 | Versioned data pipeline (DVC + Pandera) | ✅ Complete |
| 2 | Training + experiment tracking (MLflow) | ✅ Complete |
| 3 | Model Registry & packaging | ✅ Complete |
| 4 | Inference API (FastAPI) | ✅ Complete |
| 5 | Containerization (Docker) | ✅ Complete |
| 6 | CI/CD (GitHub Actions) | ⏳ Planned |
| 7 | Orchestration (Prefect) | ⏳ Planned |
| 8 | Monitoring, drift & closed retraining loop (Evidently) | ⏳ Planned |
| 9 | Deployment, final README & demo | ⏳ Planned |

---

## Repository layout

```
src/            # all production code, as an importable package
  data/         # ingest, validate (Pandera), preprocess
  models/       # train, evaluate, register (MLflow)
  api/          # FastAPI app, Pydantic schemas, prediction logic
  monitoring/   # drift detection (Evidently), dashboard
  config.py     # centralized configuration
pipelines/      # Prefect orchestration flows (kept separate from src/)
tests/          # test_data, test_model, test_api
notebooks/      # exploration only — never production code
docs/decisions/ # design-decision records (ADRs)
docker/         # Dockerfile, docker-compose
data/           # DVC-managed, not in Git
```

---

## License

Released under the [MIT License](LICENSE).
