# Phase 3 — Model Registry and Packaging

> This phase marks an important shift in mindset. Up until now, you have *experimented* with models; from here on, you begin to *manage them as production artifacts*. A model heading to production is not a loose file you save in a folder: it is a versioned entity, retrievable by name, with a lifecycle (proposed, validated, deployed, retired) and a history of which version was active at any given moment. In this phase, you will take the best model you identified in Phase 2, package it along with everything it needs to function, and register it so that anyone can retrieve "the production model" with a single line of code, without knowing or needing to know the specific version.

**Phase objective:** Manage model versions as production artifacts.
**Duration:** ~1 week (week 5 of the project).
**Upon completion you will have:** Your best model packaged (with its preprocessing and its threshold) and registered in the MLflow Model Registry, versioned, with automatic promotion to production if it outperforms the current model, and retrievable by alias with a single line of code.

---

## The Big Picture: From Experiment to Managed Artifact

The workflow of this phase takes the best experiment from Phase 2 and converts it into a managed production model:

```
   Best run from Phase 2 (highest PR-AUC in MLflow)
            │
            ▼
   ┌──────────────────────────────────┐
   │  Packaging (register.py)          │
   │  preprocessor + model + threshold │  ← a single artifact that receives
   │  → a unified PyFunc model         │     RAW data and returns predictions
   └──────────────────────────────────┘
            │
            ▼
   ┌──────────────────────────────────┐
   │  MLflow Model Registry           │
   │  "fraud-detector" v1, v2, v3...   │  ← automatic versioning
   └──────────────────────────────────┘
            │
            ▼
   ┌──────────────────────────────────┐
   │  Automatic promotion             │
   │  Does it beat the active model?   │  ← if yes, @production alias
   └──────────────────────────────────┘
            │
            ▼
   mlflow.pyfunc.load_model("models:/fraud-detector@production")
   → loads the production model in ONE line
```

Remember that MLflow has two major functions: experiment tracking (which you used in Phase 2) and the **Model Registry**, which is the focus of this phase. The Model Registry is the system that manages the lifecycle of models: it versions them, assigns them names and references, and allows you to retrieve them reliably. The SQLite backend you configured in Phase 2 is precisely what makes this registry possible (flat-file storage does not support it), so you are already good to go.

---

## Step 1 — Understanding the Model Registry: Aliases Instead of Stages

Before writing any code, it is worth understanding how the registry works, and here is a point where most tutorials are outdated. Paying attention to this will give you an advantage, as you will demonstrate that you know the current approach.

The registry organizes models into two levels. A **registered model** is a namespace, an entity with a unique name like `fraud-detector`. Inside it live the **model versions**: each time you register a model under that name, a new version is created, numbered automatically (v1, v2, v3...). Each version is an immutable snapshot: once created, it does not change. This gives you a complete history of all the models you have registered.

The key question is: how do you indicate which of all those versions is the one in production? Here is the important shift. Historically, MLflow used **stages**: you moved a version through four fixed stages (`None`, `Staging`, `Production`, `Archived`). **But stages are deprecated** and will be removed in a future release. If you see tutorials using `transition_model_version_stage` or URIs like `models:/name/Production`, they are using the old API.

The modern approach uses **aliases** and **tags**. An **alias** is a named reference that you point to a specific version, which you can move whenever you want. Instead of the fixed "Production" stage, you assign an alias `@production` (or `@champion`, as you prefer) to the version you want active. The advantage over stages is huge: aliases are flexible (you can create as many as you need: `@champion`, `@challenger`, `@shadow`...), they are not limited to four fixed values, and they cleanly separate the *identity* of the version from its current *role*. **Tags** complement this by adding arbitrary metadata to each version (for example, the metric it achieved, or who validated it).

In practice, this means that instead of "promoting a version to the Production stage", you **assign the `@production` alias to it**, and to retrieve the active model you request "the version with the `@production` alias". This is the pattern we will use, and knowing it is exactly the kind of up-to-date detail that distinguishes someone who stays current.

---

## Step 2 — The Packaging Problem and Its Solution

Now for the central challenge of this phase. Think about what you have and what you need. Your model, trained in Phase 2, expects to receive **already preprocessed** data: scaled features in the format produced by the Phase 1 pipeline. But in Phase 4, the API will receive **raw transactions** just as they arrive from the real world, without preprocessing. There is a gap between the two, and if you don't solve it properly, the specter of *training-serving skew* (which we saw in the fundamentals) reappears: you would have to reimplement the preprocessing in the API, risking it differing from the training preprocessing.

The solution is to **package the model along with everything it needs to function** into a single artifact that receives raw data and returns predictions, with no intermediate manual steps. This package must contain three things: the trained **preprocessor** (the `preprocessor.joblib` you saved in Phase 1), the trained **model** (the best one from Phase 2), and the **decision threshold** (the one you chose during evaluation). By joining them, you guarantee that the production preprocessing is identical to the training preprocessing—because it is literally the same object—and that the correct threshold always travels with the model.

MLflow allows you to do this using a **custom PyFunc model**: a class that wraps your components and defines how a prediction is made from start to finish. This will be the base for `src/models/register.py`:

```python
import mlflow.pyfunc
import pandas as pd


class FraudModel(mlflow.pyfunc.PythonModel):
    """Packages preprocessor + model + threshold into a single artifact.

    Receives raw transactions and returns fraud probability and decision.
    """

    def __init__(self, preprocessor, model, threshold: float):
        self.preprocessor = preprocessor
        self.model = model
        self.threshold = threshold

    def predict(self, context, model_input, params=None):
        # 1. Preprocess raw data exactly as in training
        X = self.preprocessor.transform(model_input)
        # 2. Get fraud probability
        proba = self.model.predict_proba(X)[:, 1]
        # 3. Apply the decision threshold to produce the label
        prediction = (proba >= self.threshold).astype(int)
        return pd.DataFrame(
            {"fraud_probability": proba, "is_fraud": prediction}
        )
```

It is worth understanding why this solves the problem so cleanly. The class saves the three components as attributes, and its `predict` method defines the complete sequence: preprocess the raw data, calculate the probability, apply the threshold. When MLflow registers this object, it will **serialize the class along with all its attributes** (including the preprocessor and model), meaning the resulting artifact is self-contained: it contains everything needed to transform a raw transaction into a decision. The Phase 4 API won't need to know anything about preprocessing or thresholds; it will simply load this package and pass the transaction to it as-is. We return both the probability and the binary label because both are useful: the label for making decisions, and the probability for ranking cases by risk or for auditing.

---

## Step 3 — The Threshold as Part of the Artifact

It is worth pausing for a moment on the threshold, because how it is handled is one of those subtleties that demonstrates maturity. In Phase 2, you consciously chose a decision threshold through the optimization you performed in `evaluate.py`, linking it to your business metric and cost asymmetry. This threshold **is not an implementation detail; it is part of the model**: without it, the model only produces probabilities and does not know where the boundary between "fraud" and "no fraud" lies.

That is why we treat it with the same care as the model itself. The chosen threshold lives versioned in `params.yaml` (so that which threshold you used and why remains registered in Git), and at the same time, it is **embedded in the package** (so that it always travels with the model, and the API applies it without needing to know it). We will read it from the parameters file during packaging:

```python
import yaml

from src.config import PROJECT_ROOT


def load_threshold() -> float:
    """Loads the chosen decision threshold versioned in params.yaml."""
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)["train"]["threshold"]
```

Make sure that the value of `train.threshold` in `params.yaml` is the optimal threshold you found in Phase 2, not the default 0.5. This is the implementation of the task "version the threshold along with the model": it remains versioned in the file, embedded in the artifact, and, as you will see, also registered as a version tag. A triple guarantee that it is never lost.

---

## Step 4 — Registering the Model

Now you will assemble the complete registration process. The module must find the best experiment from Phase 2, load its pieces, package them, and register the result in the Model Registry. Continue editing `src/models/register.py`:

```python
import joblib
import mlflow
import mlflow.pyfunc
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature

from src.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
    PREPROCESSOR_PATH,
    TARGET,
)
from src.data.ingest import load_raw_data


def find_best_run(client: MlflowClient, metric: str = "pr_auc"):
    """Finds the run with the best metric in the experiment."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No runs found in the experiment. Run Phase 2 first.")
    return runs[0]


def build_packaged_model(best_run) -> FraudModel:
    """Loads components from the best run and packages them."""
    sklearn_model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    threshold = load_threshold()
    return FraudModel(preprocessor, sklearn_model, threshold)
```

With the pieces ready, you register the package. This is what creates a new version in the Model Registry:

```python
def register_model() -> int:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    best_run = find_best_run(client)
    candidate_metric = best_run.data.metrics["pr_auc"]
    packaged = build_packaged_model(best_run)

    # Example of RAW input and signature (unpreprocessed data)
    raw_example = load_raw_data().drop(columns=[TARGET]).iloc[:5]
    sample_output = packaged.predict(None, raw_example)
    signature = infer_signature(raw_example, sample_output)

    with mlflow.start_run(run_name="register"):
        mlflow.log_metric("pr_auc", candidate_metric)
        result = mlflow.pyfunc.log_model(
            name="model",
            python_model=packaged,
            signature=signature,
            input_example=raw_example,
            registered_model_name=MODEL_NAME,
        )

    new_version = result.registered_model_version
    # We save the metric as a version tag to compare it later
    client.set_model_version_tag(
        MODEL_NAME, new_version, "pr_auc", str(candidate_metric)
    )
    print(f"Registered version {new_version} of '{MODEL_NAME}'")
    return new_version
```

Several decisions are worth noting here. The `find_best_run` function queries MLflow, sorting runs by PR-AUC in descending order and taking the best one, automating the selection you did manually in the Phase 2 UI. The **signature is constructed using raw data** (`raw_example`), not preprocessed data, since that is what the package receives; this documents that the registered model expects transactions exactly as they are and returns the probability and decision. By passing `registered_model_name=MODEL_NAME` to `log_model`, MLflow automatically registers a new version under that name. And we save the candidate's metric as a **version tag**, which will allow us to compare it later against the model currently in production.

> **API Note (2026):** We use `name="model"` in `log_model`, which is the current parameter in MLflow 3. In earlier versions it was called `artifact_path`; if you are working with MLflow 2.x you will see that name. Also, add `MODEL_NAME = "fraud-detector"` to your `src/config.py`.

---

## Step 5 — Automatic Promotion

Here you will implement one of the most valuable components of this phase: ensuring that a new model is promoted to production **only if it is better than the one already there**. This acts as a *quality gate*: an automatic check that prevents an inferior model from accidentally reaching production. This is the same concept that will reappear in Phase 6 with model validation in CI/CD, demonstrating you understand that deploying blindly is dangerous.

The logic compares the candidate's metric with that of the model currently holding the `@production` alias. If the candidate is better (or if there is no model in production yet), it assigns the alias to it; otherwise, it leaves it registered but unpromoted. Complete `register.py`:

```python
def get_production_metric(client: MlflowClient) -> float | None:
    """Reads the metric of the version holding the @production alias, if it exists."""
    try:
        version = client.get_model_version_by_alias(MODEL_NAME, "production")
        return float(version.tags["pr_auc"])
    except Exception:
        return None  # there is no model in production yet


def promote_if_better(client: MlflowClient, new_version: int, candidate_metric: float):
    production_metric = get_production_metric(client)

    if production_metric is None or candidate_metric > production_metric:
        client.set_registered_model_alias(MODEL_NAME, "production", new_version)
        print(
            f"✅ Version {new_version} promoted to @production "
            f"(PR-AUC {candidate_metric:.4f})"
        )
    else:
        print(
            f"⏸️  Version {new_version} registered but NOT promoted: "
            f"{candidate_metric:.4f} does not outperform current model ({production_metric:.4f})"
        )


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    best_run = find_best_run(client)
    candidate_metric = best_run.data.metrics["pr_auc"]
    new_version = register_model()
    promote_if_better(client, new_version, candidate_metric)


if __name__ == "__main__":
    main()
```

The key function is `set_registered_model_alias`, which assigns the `@production` alias to the winning version. Notice how `get_production_metric` reads the metric from the **tag** of the version currently in production, retrieving it by its alias: this is the modern approach (aliases + tags) in action, without a single deprecated "stage". The check gracefully handles the initial case when there is no model in production yet, promoting the first one automatically. From then on, each new registration only shifts the alias if it actually improves performance.

---

## Step 6 — Loading the Production Model with a Single Line

Here is the payoff for all the work in this phase, and your criterion for success. Once the model is registered and promoted, retrieving it is trivial—and most importantly, **you do not need to know which version it is**. You simply request "the production model":

```python
import mlflow.pyfunc

from src.config import MODEL_NAME

# Load the production model by its alias, without knowing the version
model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")

# And predict directly on raw data
predictions = model.predict(raw_transactions)
```

That single line, `mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@production")`, is the goal of this entire phase. The URI `models:/fraud-detector@production` means "the version of the fraud-detector model that has the production alias". When you promote a new version, this same line will automatically load the new one without touching the code that consumes it. This is exactly what makes the alias pattern so powerful: it decouples the code consuming the model from the specific active version. The Phase 4 API will use precisely this line, and it will never have to change even if you promote new models every week.

And because the model is packaged, `model.predict(raw_transactions)` receives raw transactions and directly returns the probability and decision of fraud, with the preprocessing and threshold applied internally. All the packaging work from the previous steps manifests here, in this simplicity.

---

## Step 7 — Exploring the Registry in the UI

Take a moment to see what you have built in the MLflow UI (`http://localhost:5000`). In the models section, you will see your registered `fraud-detector` model with all of its versions listed. For each version, you can view its tags (including the PR-AUC metric you saved) and its assigned alias. The version with the `@production` alias is clearly marked. If you register multiple models (for instance, by retraining with different data or hyperparameters), you will see the versions stack up and the `@production` alias move automatically to the best one. This visibility of the model lifecycle is exactly what a team needs to know, at any given moment, which model is deployed and why.

---

## Step 8 — Tests

We close with tests that verify the two core logics of this phase: packaging and the promotion decision. Complete `tests/test_model.py` by adding:

```python
import numpy as np
import pandas as pd

from src.models.register import FraudModel


class _FakePreprocessor:
    """Mock preprocessor for testing: returns data as-is."""
    def transform(self, X):
        return X.values if hasattr(X, "values") else X


class _FakeModel:
    """Mock model: probability is the first column."""
    def predict_proba(self, X):
        p = np.asarray(X)[:, 0]
        return np.column_stack([1 - p, p])


def test_packaged_model_outputs_expected_columns():
    packaged = FraudModel(_FakePreprocessor(), _FakeModel(), threshold=0.5)
    raw = pd.DataFrame({"feat": [0.2, 0.9]})
    result = packaged.predict(None, raw)
    assert list(result.columns) == ["fraud_probability", "is_fraud"]
    assert len(result) == 2


def test_packaged_model_applies_threshold():
    packaged = FraudModel(_FakePreprocessor(), _FakeModel(), threshold=0.5)
    raw = pd.DataFrame({"feat": [0.3, 0.8]})  # probas 0.3 and 0.8
    result = packaged.predict(None, raw)
    # With threshold 0.5: 0.3 -> not fraud (0), 0.8 -> fraud (1)
    assert result["is_fraud"].tolist() == [0, 1]
```

These tests use test doubles (a mock preprocessor and model) to quickly and in isolation verify that the packaging produces the expected columns and that the threshold is correctly applied. Testing the packaging logic without depending on a real trained model is a good practice that keeps your tests fast and focused. Run them using `make test`.

---

## Verification: The Definition of "Done"

The phase is complete when the following conditions are met:

- [ ] You understand and use the modern approach of aliases and tags, rather than deprecated stages.
- [ ] `register.py` automatically finds the best run from Phase 2 by its metric.
- [ ] The model is packaged with its preprocessor and threshold in a PyFunc that receives raw data and returns predictions.
- [ ] The decision threshold is versioned in `params.yaml` and embedded in the artifact.
- [ ] The model is registered in the Model Registry, automatically creating numbered versions.
- [ ] Automatic promotion assigns the `@production` alias only if the candidate outperforms the current model.
- [ ] Tests pass using `make test`.
- [ ] **The key test:** You can load the production model with a single line (`mlflow.pyfunc.load_model("models:/fraud-detector@production")`), without knowing the specific version, and predict on raw data.

The key test is the single-line check: if you can retrieve "the production model" by its alias without knowing the version and use it directly on raw transactions, you have built a production-grade model management setup. This decoupling between the code and the active version is the essence of what the Model Registry provides.

---

## Deliverables and What Comes Next

By completing Phase 3, you have leaped from experimenting to managing models as production artifacts: your best model is packaged autonomously (preprocessor and threshold included), registered and versioned in the Model Registry, protected by a quality gate that only promotes superior models, and retrievable by alias with a single line of code. Beyond the mechanics, you have shown that you understand the lifecycle of a production model and the modern pattern (aliases and tags) to manage it, avoiding the deprecated stages that still appear in many tutorials.

The next step, **Phase 4**, builds directly on this and is where the model begins to be useful to the world: you will wrap the production model in a **REST API** using FastAPI. Here is where the reward of this phase's packaging shows: the API will only need to load the model with the single line `mlflow.pyfunc.load_model("models:/fraud-detector@production")` and pass it the transactions it receives, without worrying about preprocessing, thresholds, or versions. All the complexity you have encapsulated here is what will make the next phase's API clean and robust. You have gone from "knowing how to manage models" to being on the verge of "knowing how to serve them as a product".
