# Decision 20: API structure — inference in `predict.py`, model in `app.state`

- **Date:** 2026-08-23
- **Status:** Accepted

## Context

`project_context/mlops_phase4.md` writes the whole API in `src/api/main.py`: the `lifespan`, the
version resolution, the call to `model.predict()`, and the prediction log. It never uses
`src/api/predict.py`, and it holds the model in a module-level dict, `ml = {}`, which its tests
mutate directly (`mlops_phase4.md:112, 285`).

This repository committed a different structure in Phase 0. `src/api/predict.py` has carried the
docstring *"Load the production model and produce predictions."* since commit `aef7535`, and both
`mlops_generalRoadmap.md:121` and `mlops_fundamentals.md:289` list it as `predict.py # inference
logic`.

## Decision

Three splits, diverging from the reference material on each:

- **Inference lives in `src/api/predict.py`**, which imports no FastAPI symbol at all. It owns
  `resolve_production_version()`, `load_production()` and `predict_transaction()`, and raises
  `PredictionError`. `main.py` is the HTTP surface: app, lifespan, routes, and the translation of
  `PredictionError` into a status code.
- **The `lifespan` function itself stays in `main.py`.** It takes a `FastAPI` argument, so hosting
  it in `predict.py` would drag the framework into the inference module and defeat the split. It
  delegates the actual load to `predict.load_production()`.
- **State lives on `app.state`**, under `MODEL_STATE_KEY = "model"` and
  `VERSION_STATE_KEY = "model_version"`, read through `getattr(request.app.state, ...)`.

`load_production()` calls Phase 3's `load_production_model()` rather than rebuilding the
`models:/fraud-detector@production` URI, so that URI stays written in exactly one place — the
reuse [0015](0015-packaged-model-contract.md) anticipated.

## Alternatives considered

- **Everything in `main.py`, per the reference material** — rejected. It leaves a committed module
  permanently empty and puts Registry access, scoring and HTTP concerns in one file.
- **A module-level `ml = {}` dict** — rejected. `app.state` is the framework-native mechanism and
  is scoped to the application object. The dict is a mutable global that survives between tests:
  the reference material's own three tests assign to it and never clean up, so they are coupled by
  execution order.
- **Importing `register._client()`** for the version lookup — rejected. It is private to that
  module; `predict.py` builds its own `MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)`. The URI is
  the shared contract, not the client object.

## Justification

The separation is what makes each piece testable on its own terms. `predict.py` can be exercised
without HTTP; `main.py` can be exercised with a double injected into `app.state` and no MLflow
present at all ([0021](0021-prediction-log-and-api-tests.md)). It also keeps the dependency arrow
pointing one way — `main` → `predict` → `register` — with no module importing the layer above it.

A startup that cannot load the model **logs the failure and starts anyway**. `/health` then
reports `no_model` and the other two endpoints answer `503`. That is what lets a container
orchestrator observe an unhealthy instance and act, instead of watching a process crash-loop with
no surface to inspect — and it is why `/health` is the one endpoint that must never raise.

`PredictionError(FraudPipelineError)` continues the one-exception-per-module convention
(`ModelTrainingError`, `ModelEvaluationError`, `ModelRegistrationError`). The reference material
uses no custom exception here; adopting one keeps the hierarchy represented in every phase and
means an inference failure surfaces as this project's own type rather than a raw MLflow or pandas
error escaping as an opaque 500.

## Trade-offs / consequences

- **Two modules to keep in sync**, and function signatures the reference material does not
  specify, designed here.
- **The reference material's Step 6 test snippet is not copy-pasteable**: it targets `main.ml`,
  which does not exist. Tests inject into `app.state` instead.
- **A translation layer for few failure modes today.** The alternative — `HTTPException` only —
  would have been defensible had all logic stayed in `main.py`, where the API *is* the boundary.
- **`MLFLOW_TRACKING_URI` remains a hardcoded, CWD-relative literal** (`sqlite:///mlflow.db`), so
  the API resolves the Registry relative to wherever the process starts. Untouched here; it is the
  Phase 5 gap already recorded in [0011](0011-mlflow-sqlite-backend.md)'s trade-offs.
