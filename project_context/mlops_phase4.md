# Phase 4 — Inference API

> So far, you have an excellent, packaged, and registered model, but it is of no use to anyone yet: you can only use it from your own code. To make it a product, it must become a **service** that any system can call. This phase builds that service: a professional REST API that receives a transaction, validates it, passes it through the model, and returns a prediction. This is where you reap the rewards of the packaging from Phase 3: since the production model already contains its preprocessing and threshold, the API remains surprisingly clean, without having to worry about transformations or versions.

**Phase objective:** Serve the model as a professional REST API.
**Duration:** ~2 weeks (weeks 5-6 of the project).
**Upon completion, you will have:** A functional, automatically documented, and tested API that validates incoming transactions, returns predictions with their probabilities, logs each prediction for future monitoring, and can be tested directly from the browser.

---

## The Big Picture: The Model as a Service

The API flow for each incoming transaction is as follows:

```
   Raw transaction (JSON) arrives at POST /predict
            │
            ▼
   ┌──────────────────────────┐
   │  Validation (Pydantic)   │  ← Is the transaction correctly formatted?
   └──────────────────────────┘     If not, automatic 422 error
            │
            ▼
   ┌──────────────────────────┐
   │     Production Model     │  ← Loaded ONCE at API startup
   │    (loaded by alias)     │     from the Model Registry
   └──────────────────────────┘
            │
            ▼
   ┌──────────────────────────┐
   │    Prediction Logging    │  ← input + output + timestamp
   └──────────────────────────┘     (the seed of monitoring)
            │
            ▼
   Response: probability + fraud decision
```

Before starting, add the dependencies for this phase. FastAPI to build the API, Uvicorn as the server to run it, and Pydantic (which FastAPI already pulls in, but we declare it explicitly):

```bash
uv add fastapi "uvicorn[standard]" pydantic
```

---

## Step 1 — Schemas: Validating at the Boundary

The first principle of a robust API, which connects directly to the validation philosophy from the fundamentals, is **never to let malformed data in**. The boundary of your system is where you must validate most rigorously, because that is where data arrives from the outside, over which you have no control. In FastAPI, this validation is provided by **Pydantic schemas**: classes that declare exactly what shape an incoming transaction must have and what shape the response should take. If something arrives that does not fit, FastAPI automatically rejects it before your code even runs.

Complete `src/api/schemas.py`:

```python
from pydantic import BaseModel, ConfigDict, Field, create_model

# The 28 PCA features from the dataset (V1..V28): all float and required.
# We generate them programmatically to avoid writing 28 identical lines.
_pca_features = {f"V{i}": (float, Field(...)) for i in range(1, 29)}

# INPUT model: a raw transaction, just as it arrives from the outside
Transaction = create_model(
    "Transaction",
    Time=(float, Field(ge=0, description="Seconds since the first transaction")),
    Amount=(float, Field(ge=0, description="Transaction amount")),
    **_pca_features,
)


# OUTPUT model: the prediction response
class PredictionResponse(BaseModel):
    # protected_namespaces=() prevents Pydantic warning for the "model_" prefix
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float = Field(
        ..., ge=0, le=1, description="Estimated probability of fraud"
    )
    is_fraud: int = Field(..., description="1 if classified as fraud, 0 if not")
    model_version: str = Field(..., description="Version of the model that made the prediction")
```

There are several decisions worth explaining, as each reflects a best practice in Pydantic v2 (the current version, which differs from v1 still found in many tutorials):

The **programmatically generated input model.** The 28 features `V1` to `V28` are anonymous PCA components, all of the same type and without semantic meaning. Thus, instead of writing 28 identical lines, we use `create_model` to generate them. For `Time` and `Amount`, which are interpretable, we declare them explicitly along with their constraints (`ge=0`, meaning they cannot be negative) and a description that will appear in the documentation. The tuple `(float, Field(...))` indicates the type and that the field is required. This is both clean and fully functional, demonstrating that you know how to handle schemas with many homogeneous fields without repeating yourself.

**Constraints as automatic validation.** By declaring `Amount` with `ge=0`, any request with a negative amount will be automatically rejected by FastAPI with a clear error, without you writing a single line of checking logic. Type validation (making sure `Time` is a number, not text) is also automatic. This is the power of Pydantic: turning implicit assumptions into guarantees enforced by the framework.

The **typed output model.** The response also has a schema: a probability (which we also restrict to the valid range between 0 and 1), a binary decision, and the model version. Including the version in each response is a best practice for traceability: if you ever need to investigate a specific prediction, you will know exactly which model produced it.

> **Pydantic v2 Detail:** The field `model_version` begins with `model_`, a prefix that Pydantic v2 reserves internally, which triggers a warning. We silence this with `model_config = ConfigDict(protected_namespaces=())`. Using `ConfigDict` is also the modern way to configure a model in Pydantic v2 (in v1, a nested `Config` class was used). Knowing this nuance shows you are up-to-date with the library.

---

## Step 2 — Loading the Model on Startup: The Lifespan Pattern

An important design decision is **when** the model is loaded. The naive approach would be to load it on every request, but that would be disastrous for performance: loading a model is an expensive operation, and doing so on every call would add huge latency. The correct approach is to load it **only once, when the API starts up**, keeping it in memory ready to respond. This way, the first request and the millionth request experience the same low latency.

FastAPI manages this with the **lifespan** pattern, and here is an important update. Historically, `@app.on_event("startup")` decorators were used, **but they are deprecated**. The modern approach is a `lifespan` context manager: a function that defines what happens on startup (before the `yield`) and shutdown (after). These will be the foundation of `src/api/main.py`:

```python
import logging
from contextlib import asynccontextmanager

import mlflow.pyfunc
from fastapi import FastAPI
from mlflow import MlflowClient

from src.config import MLFLOW_TRACKING_URI, MODEL_NAME

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud_api")

# Application state: this is where the loaded model lives
ml = {}


def _resolve_production_version() -> str:
    """Finds out which specific version has the @production alias."""
    client = MlflowClient()
    version = client.get_model_version_by_alias(MODEL_NAME, "production")
    return version.version


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup: Load the production model ONCE ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@production"
    ml["model"] = mlflow.pyfunc.load_model(model_uri)
    ml["version"] = _resolve_production_version()
    logger.info("Model loaded: %s (version %s)", MODEL_NAME, ml["version"])
    yield
    # --- Shutdown: Release resources ---
    ml.clear()
    logger.info("Model unloaded")


app = FastAPI(
    title="Fraud Detection API",
    description="Serves the fraud detection model in production.",
    version="1.0.0",
    lifespan=lifespan,
)
```

Notice how the reward of Phase 3 manifests here. Loading the model takes just a single line: `mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")`. This line requests "the model that has the production alias", without the API knowing or needing to know which specific version it is. When you promote a new model in the registry, you simply restart the API to load the new one, **without touching the code**. This decoupling of the code from the active version, which you built in Phase 3, is what makes this so clean. We store the model and its version in an `ml` dictionary that acts as the application state, accessible from the endpoints.

---

## Step 3 — Endpoints

Now you define the three API routes. Each has a clear purpose. Continue `src/api/main.py`:

```python
import pandas as pd
from fastapi import HTTPException

from src.api.schemas import PredictionResponse, Transaction


@app.get("/health")
def health():
    """Health check: indicates whether the API is alive and the model is loaded."""
    return {"status": "ok" if "model" in ml else "no_model"}


@app.get("/model-info")
def model_info():
    """Information about the active model."""
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"model_name": MODEL_NAME, "version": ml["version"], "alias": "production"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    """Receives a raw transaction and returns the fraud prediction."""
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # The transaction is already validated by Pydantic; we convert it to a DataFrame
    df = pd.DataFrame([transaction.model_dump()])
    result = ml["model"].predict(df)

    response = PredictionResponse(
        fraud_probability=float(result["fraud_probability"].iloc[0]),
        is_fraud=int(result["is_fraud"].iloc[0]),
        model_version=ml["version"],
    )
    log_prediction(transaction, response)  # defined in the next step
    return response
```

Each endpoint serves a necessary function in a production service:

The **`/predict`** endpoint is the heart of the API. It receives a transaction (which Pydantic has already validated via the `Transaction` type), converts it to a DataFrame using `model_dump()` (Pydantic v2's method for serializing to a dictionary), and passes it to the model. Here, once again, the elegance of the packaging is apparent: since the production model contains its preprocessing and threshold, we simply pass it the raw transaction, and it returns the already-calculated probability and decision. There is no preprocessing in the API, and no threshold in the API: all of that lives inside the model. The response is constructed using the typed schema and returned.

The **`/health`** endpoint is a *health check*: a simple route that reports whether the API is alive. It might seem trivial, but it is essential: in Phase 9, the deployment service will use this endpoint to know if your application is running properly or if it needs a restart. This is a standard requirement for any production service.

The **`/model-info`** endpoint returns which model is active and its version. This provides operational traceability: at any time, you can ask the running API which model it is serving, which is highly useful for debugging and confirming that a promotion was applied successfully.

Also, note the use of `HTTPException`: if for some reason the model is not loaded, the API responds with a 503 code (Service Unavailable) and a clear message, rather than failing in a confusing way. Handling errors with correct HTTP status codes is part of building a professional API.

---

## Step 4 — Logging Predictions: The Seed of Monitoring

This piece might seem minor, but it is one of the most important in the entire project because it enables the highlight of Phase 8. **Every prediction the API makes must be logged**: the input transaction, the output prediction, and the timestamp of when it occurred. Without this log, it would be impossible to later detect if the model is degrading in production, as you would have no trace of what it has been predicting.

Complete `main.py` with the logging function:

```python
import json
from datetime import datetime, timezone

from src.config import PREDICTIONS_LOG


def log_prediction(transaction: Transaction, response: PredictionResponse) -> None:
    """Logs each prediction as a JSON line, the foundation of monitoring."""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": transaction.model_dump(),
        "fraud_probability": response.fraud_probability,
        "is_fraud": response.is_fraud,
        "model_version": response.model_version,
    }
    PREDICTIONS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")
```

Each prediction is appended as a line to a file in JSONL format (one JSON object per line), a simple and highly practical format for growing logs. Each entry saves the complete input, the output, and a UTC timestamp. Add the log path to your `src/config.py`:

```python
PREDICTIONS_LOG = PROJECT_ROOT / "logs" / "predictions.jsonl"
```

It is worth understanding why this is the **seed of monitoring**. In Phase 8, you will build a system that detects *drift*: it checks if the transactions arriving in production have changed their distribution compared to the training data, or if the predictions are degrading. That system needs data on what the model has been seeing and predicting, and that data is exactly what this log captures. In this phase, you are planting the seed that will grow into the most impressive piece of the project. That is why, even though it now looks like a simple `write` to a file, it is one of the most important design decisions you will make.

> **Production Note:** For a project, a JSONL file is perfect. In a real, large-scale system, these logs would go to a database or an event streaming system. Pointing out that you understand this difference (using a file for the project versus a database for actual production) demonstrates good engineering judgment regarding how solutions scale.

---

## Step 5 — Running the API and Automatic Documentation

With everything written, you can start the API. Remember that in Phase 0 you already added the `serve` command to the Makefile, so you only need to run:

```bash
make serve
# equivalent to: uv run uvicorn src.api.main:app --reload
```

The `--reload` option automatically reloads the API whenever you change the code, which is highly convenient during development. The API will be available at `http://localhost:8000`.

And here comes one of the greatest features of FastAPI, and the success criteria of this phase: **automatic interactive documentation**. Because your endpoints use Pydantic schemas, FastAPI automatically generates complete documentation following the OpenAPI standard and serves it with a visual interface (Swagger UI) at `http://localhost:8000/docs`. Open that address in your browser, and you will see all your endpoints documented with their input and output schemas. Best of all, **you can test them directly from there**: expand the `/predict` endpoint, click "Try it out", fill in a transaction (you can copy values from a real row in the `creditcard.csv` dataset), and run it. You will receive the prediction along with its probability, live.

This ability to test the model from the browser, without writing a single line of client code, is incredibly valuable for a portfolio: it allows a recruiter or an engineer to interact with your model directly. And it is, literally, the definition of "done" for this phase: if you start the API, go to `/docs`, send a test transaction, and receive a prediction with its probability, you have met the objective. (FastAPI also offers a second documentation interface, more reading-oriented, at `/redoc`.)

---

## Step 6 — Testing with TestClient

We close with tests, maintaining the quality discipline of the entire project. FastAPI provides a `TestClient` that allows you to test the API without starting a real server, simulating HTTP requests. Complete `tests/test_api.py`:

```python
import pandas as pd
from fastapi.testclient import TestClient

from src.api import main
from src.api.main import app


class _FakeModel:
    """Mock model to avoid relying on the registry during tests."""
    def predict(self, df):
        return pd.DataFrame({"fraud_probability": [0.9], "is_fraud": [1]})


def _sample_payload() -> dict:
    payload = {"Time": 0.0, "Amount": 100.0}
    payload.update({f"V{i}": 0.0 for i in range(1, 29)})
    return payload


def test_health_reports_ok_when_model_loaded():
    main.ml["model"] = _FakeModel()
    main.ml["version"] = "1"
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_a_prediction():
    main.ml["model"] = _FakeModel()
    main.ml["version"] = "1"
    client = TestClient(app)
    response = client.post("/predict", json=_sample_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["is_fraud"] == 1
    assert 0.0 <= body["fraud_probability"] <= 1.0


def test_predict_rejects_invalid_payload():
    bad_payload = _sample_payload()
    del bad_payload["Amount"]  # missing a required field
    client = TestClient(app)
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422  # Pydantic validation
```

These tests verify the key behaviors of the API. The critical technique here is how we avoid relying on the Model Registry: we inject a **mock model** directly into the `main.ml` state before calling the endpoints, and we use the `TestClient` without its context manager so that the `lifespan` does not execute and no attempt is made to load the real model. This makes the tests fast and removes the need for a running MLflow infrastructure.

The third test is especially revealing: it sends a transaction that is missing a required field and verifies that the API responds with a **422** code (Unprocessable Entity). This 422 is generated automatically by Pydantic without us writing any validation checks; the test confirms that validation at the boundary is working. Testing that your API correctly rejects invalid data, and not just that it accepts valid data, is a sign of professional care. Run them with `make test`.

---

## Verification: Definition of Done

The phase is complete when the following are met:

- [ ] `schemas.py` defines the input schema (the transaction, with validation) and the output schema (the typed prediction) using Pydantic v2.
- [ ] The production model is loaded only once on startup, using the `lifespan` pattern (not the deprecated `on_event`).
- [ ] Loading uses the `@production` alias, without coupling the code to a specific version.
- [ ] The `/predict`, `/health`, and `/model-info` endpoints exist, complete with HTTP error handling.
- [ ] Each prediction is logged (input + output + timestamp) in the JSONL log, laying the foundation for monitoring.
- [ ] Tests with `TestClient` pass, including the validation case (422).
- [ ] **The key test:** You start the API, go to `http://localhost:8000/docs`, send a test transaction, and receive a prediction with its probability.

The key test is the interactive documentation: if you can open `/docs`, send a transaction, and see the live prediction, you have built a real inference service, usable by anyone, not just from your own code. That is the difference between having a model and having a product.

---

## Deliverables and Next Steps

Upon wrapping up Phase 4, you have converted the model into a service: a professional REST API that rigorously validates incoming transactions, serves predictions with their probabilities using the packaged production model, logs each prediction for future monitoring, documents and tests itself, and is fully covered by tests. You have crossed the line separating a model from a product: now any system, or anyone using a browser, can use your model.

The next step, **Phase 5**, solves the classic "it works on my machine" problem: you will package this API with **Docker** so that it runs identically on any machine, independent of your local configuration. You will build an image containing the API and its entire environment, and run it alongside MLflow using Docker Compose. The clean, well-structured API you built here is exactly what you will containerize, and the health check you added will be what allows the container (and, in Phase 9, the deployment service) to know if the application is healthy. You have gone from "I know how to serve a model" to being on the verge of "I know how to package a service to run anywhere".
