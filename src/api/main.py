"""FastAPI application and entry point for the inference API.

Deliberately thin. This module owns the HTTP surface — the app, its lifespan,
the three routes, and the translation of failures into status codes — and
delegates every inference concern to src/api/predict.py.

The model is loaded exactly once, on startup, through the @production alias.
Nothing here names a version: promoting a new one in the Registry changes what
this service answers with after a restart, without a line of this file
changing (docs/decisions/0015-packaged-model-contract.md).
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from mlflow.pyfunc import PyFuncModel

from src.api.predict import load_production, predict_transaction
from src.api.schemas import PredictionResponse, Transaction
from src.config import MODEL_NAME
from src.exceptions import PredictionError
from src.models.register import PRODUCTION_ALIAS

# No basicConfig(): configuring the root logger is the application runner's
# decision, not a library module's. Every module in this project takes a
# named logger and leaves handlers to whoever runs it.
logger = logging.getLogger(__name__)

MODEL_STATE_KEY = "model"
VERSION_STATE_KEY = "model_version"

MODEL_UNAVAILABLE_DETAIL = "Model not loaded"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the production model once at startup and release it at shutdown.

    Loading here rather than per request is the whole point: unpacking the
    artifact and deserializing the preprocessor and booster costs far more
    than scoring does, so paying it once keeps the millionth request as fast
    as the first.

    A failed load is logged and swallowed on purpose. The service still comes
    up, /health reports that no model is loaded, and the other two endpoints
    answer 503 — which is what lets a container orchestrator see an unhealthy
    instance and act on it, instead of watching the process crash-loop with
    no surface to inspect.
    """
    try:
        model, version = load_production()
    except PredictionError:
        # exception() keeps the cause's traceback; the API is deliberately
        # still allowed to start.
        logger.exception("Starting without a model: the production load failed")
    else:
        setattr(app.state, MODEL_STATE_KEY, model)
        setattr(app.state, VERSION_STATE_KEY, version)
        logger.info("Loaded %s version %s", MODEL_NAME, version)

    yield

    for key in (MODEL_STATE_KEY, VERSION_STATE_KEY):
        if hasattr(app.state, key):
            delattr(app.state, key)
    logger.info("Model released")


app = FastAPI(
    title="Fraud Detection API",
    description="Serves the packaged fraud-detection model held by the "
    "@production alias in the MLflow Model Registry.",
    version="1.0.0",
    lifespan=lifespan,
)


def _loaded_model(request: Request) -> tuple[PyFuncModel, str]:
    """Return the loaded model and its version, or fail with a 503.

    Args:
        request: the active request, carrying the application state.

    Returns:
        The production artifact and the version it was resolved from.

    Raises:
        HTTPException: 503 if startup could not load a model.
    """
    model: PyFuncModel | None = getattr(request.app.state, MODEL_STATE_KEY, None)
    version: str | None = getattr(request.app.state, VERSION_STATE_KEY, None)

    if model is None or version is None:
        raise HTTPException(status_code=503, detail=MODEL_UNAVAILABLE_DETAIL)

    return model, version


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    """Report whether the service is up and holding a model.

    The only endpoint that answers even with no model loaded — reporting the
    degraded state is exactly its job, so it must never raise.
    """
    loaded = getattr(request.app.state, MODEL_STATE_KEY, None) is not None
    return {"status": "ok" if loaded else "no_model"}


@app.get("/model-info")
def model_info(request: Request) -> dict[str, str]:
    """Report which registered model, version and alias are being served.

    Operational traceability: it is how a promotion is confirmed to have taken
    effect, and how a past prediction is attributed to a concrete artifact.
    """
    _, version = _loaded_model(request)
    return {
        "model_name": MODEL_NAME,
        "version": version,
        "alias": PRODUCTION_ALIAS,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction, request: Request) -> PredictionResponse:
    """Score one raw transaction.

    By the time this runs, Pydantic has already rejected anything malformed
    with a 422; the body is known to hold all 30 fields with valid types and
    ranges. The transaction then goes to the model untouched — the artifact
    carries its own preprocessing and threshold.

    Raises:
        HTTPException: 503 if no model is loaded, 500 if scoring fails.
    """
    model, version = _loaded_model(request)

    # Annotated because model_dump() is untyped and would otherwise widen.
    features: dict[str, float] = transaction.model_dump()

    try:
        prediction = predict_transaction(model, features)
    except PredictionError as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionResponse(
        fraud_probability=prediction.fraud_probability,
        is_fraud=prediction.is_fraud,
        model_version=version,
    )
