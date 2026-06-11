# Phase 2 — Training and Experiment Tracking

> This is the phase where Machine Learning finally comes into play. However, it is best to start with the right mindset, which might feel counterintuitive: the goal of this phase **is not to obtain the absolute best possible model**. The goal is to train models rigorously and, above all, to record every single experiment with enough discipline that you can always answer the question: "what exactly produced this result?". Remember what we discussed in the fundamentals: companies are not impressed by your F1-score, but by your ability to build a reliable system. A mediocre, well-traced model is worth more for this project than an excellent one you trained once without really knowing how. Traceability is the skill being demonstrated here.

**Objective of the phase:** train models while rigorously tracking every experiment.  
**Duration:** ~2 weeks (weeks 3-4 of the project).  
**Upon completion you will have:** several experiments comparable side-by-side in the MLflow UI, each with its parameters, metrics, and artifacts; the best model identified with clear criteria; and a rigorous evaluation that includes cross-validation and the conscious choice of the decision threshold based on your business metric.

---

## The Big Picture: Experimenting with Traceability

The workflow of this phase directly consumes what you produced in Phase 1 and adds the tracked experimentation layer:

```
   data/processed/train.parquet  ──┐
   data/processed/test.parquet   ──┤
                                   ▼
                      ┌─────────────────────────┐
                      │  train.py               │
                      │  - trains a model       │
                      │  - handles imbalance    │
                      │  - calculates metrics   │
                      └─────────────────────────┘
                                   │
                                   ▼
                      ┌─────────────────────────┐
                      │  MLflow                 │  ← logs params, metrics,
                      │  (tracking)             │    model, and artifacts
                      └─────────────────────────┘
                                   │
                                   ▼
                  MLflow UI: compare all experiments
                  side-by-side
```

As we discussed in the fundamentals, MLflow solves two distinct problems: **experiment tracking** and the **model registry** (Model Registry). In this phase, we use only the first; the Model Registry will come in Phase 3. Here we focus on recording every training run with all its context so we can compare them.

Before starting, add the dependencies you will need. MLflow for tracking, XGBoost for a more powerful model, and imbalanced-learn for resampling techniques to handle the imbalance:

```bash
uv add mlflow xgboost imbalanced-learn
```

---

## Step 1 — Spinning Up MLflow

MLflow needs a place to store experiment data. You have two options, and the choice matters for a future reason that is worth anticipating.

The simplest option is **flat-file storage**, where MLflow saves everything inside an `mlruns/` folder. It works out of the box with zero configuration, but has a critical limitation: **it does not support the Model Registry** which you will need in Phase 3. Therefore, to avoid refactoring your work later, the recommendation is to use a **SQLite backend** from the very beginning. It is just as simple (a single-file database with no server to maintain) but fully supports the model registry.

Configure the SQLite backend by adding these constants to `src/config.py`:

```python
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "fraud-detection"
MODELS_DIR = PROJECT_ROOT / "models"
```

Remember to add `mlflow.db` and the MLflow artifacts folder to your `.gitignore`, just as you did with `mlruns/` in Phase 0, so you don't version these files managed by MLflow.

To open the MLflow web interface and visualize the experiments, run the following command in a separate terminal (leave it running while you work):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

This spins up the UI at `http://localhost:5000`. It will be empty at first; it will populate as you train.

---

## Step 2 — Loading Features and Preparing Training

The training code consumes the datasets produced by the Phase 1 data pipeline. Let's first create a small, reusable loading function. These will be the foundation of `src/models/train.py`:

```python
import pandas as pd

from src.config import TARGET, TEST_PATH, TRAIN_PATH


def load_split(path):
    """Loads a processed dataset and splits it into features (X) and target (y)."""
    df = pd.read_parquet(path)
    return df.drop(columns=[TARGET]), df[TARGET]
```

Notice that there is no preprocessing here: the data arrives clean, validated, and scaled from the previous phase, with the train and test split already completed. This separation of concerns is exactly what we aimed for: preprocessing lives in Phase 1, training lives here, and each part does one thing.

---

## Step 3 — The First Model: An Honest Baseline

Good experimental discipline starts with a **baseline**: a simple model that serves as a reference point. Without a baseline, you cannot tell if your complex models add any value. For classification, a **logistic regression** is the ideal baseline: it is simple, fast, interpretable, and surprisingly competitive in many problems.

What matters for fraud is how it handles the class imbalance. Logistic regression (and many other scikit-learn models) accepts the `class_weight="balanced"` parameter, which automatically forces the model to pay more attention to the minority class, compensating for the imbalance without needing to modify the data. It is the simplest of the three techniques we will see, and an excellent starting point.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    class_weight="balanced",   # compensates for imbalance
    max_iter=1000,
    random_state=42,
)
model.fit(X_train, y_train)
```

---

## Step 4 — The Right Metrics for Fraud

Before training more models, we must measure performance correctly, and this is critical. As emphasized in the fundamentals, in a highly imbalanced problem, accuracy is useless (a model that says "nothing is fraud" would have over 99% accuracy and be completely useless). The metrics that actually matter are different, and it is best to calculate all of them to get a complete picture:

```python
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
```

Each metric measures something distinct and necessary. **Precision** tells you, of all the cases you marked as fraud, what fraction actually was: it controls false positives, meaning how many legitimate clients you disturb. **Recall** tells you, of all the actual fraud, what fraction you managed to detect: it controls false negatives, the fraud that slips past you. **F1** is the harmonic mean of both, a single number that balances them. And **PR-AUC** (the area under the precision-recall curve, computed in scikit-learn using `average_precision_score`) summarizes the model's performance across all possible thresholds.

Notice a deliberate detail: `average_precision_score` receives the **probabilities** (`y_proba`), not the binary predictions, because it measures the quality of the model's ranking independently of the threshold. And remember why we use PR-AUC instead of the more common ROC-AUC: in highly imbalanced problems, the ROC curve gives a misleadingly optimistic impression, whereas the precision-recall curve reflects the actual performance much better when the positive class is rare. Being able to explain this choice is exactly the kind of detail that distinguishes someone who understands the problem.

---

## Step 5 — Handling Imbalance: Three Strategies

Extreme imbalance is the central technical challenge in fraud detection, and there are three families of techniques to address it. It is good to understand all three, as each has its advantages and being able to reason about them demonstrates domain expertise.

**Strategy 1: Class weighting.** This is what you already used with `class_weight="balanced"`. The model penalizes mistakes on the minority class more heavily, compensating for the imbalance without touching the data. It is the simplest and often sufficient. In XGBoost, the equivalent parameter is `scale_pos_weight`, which is typically set to the ratio of negative to positive cases.

**Strategy 2: Resampling (SMOTE and similar).** This involves rebalancing the training data, either by generating synthetic examples of the minority class (SMOTE, from the `imbalanced-learn` library) or downsampling the majority class. Here lies a **critical data leakage trap** that you must avoid and that many people make: resampling should only be applied to the **training** data, never to the test set, and when used with cross-validation, it must happen **inside** each fold, not before. If you generate synthetic examples across the entire dataset and then split, synthetic information derived from the test set leaks into training. The correct way to do this is with the `imbalanced-learn` `Pipeline`, which applies resampling only during the training phase of each fold:

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression

# Resampling goes INSIDE the pipeline: it only affects training,
# never the evaluation set
pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=1000, random_state=42)),
])
```

It is the same lesson you learned with scaling in Phase 1, now applied to resampling: nothing that depends on the data can be fit using the set you are going to evaluate on.

**Strategy 3: Adjusting the decision threshold.** This is the most important of the three and, paradoxically, the one most often ignored. By default, a classifier decides "fraud" if the estimated probability exceeds 0.5. **But 0.5 is almost always the wrong threshold in an imbalanced problem.** The optimal threshold depends on your business metric and the cost asymmetry you documented in Phase 0, and it is rarely 0.5. Adjusting the threshold is often what improves real-world performance the most, without changing the model at all. We will see how to find it rigorously in the evaluation step; for now, it is enough to know that the threshold is a parameter you choose deliberately, not a default value you accept without thinking.

The reasonable approach is to combine strategies: start with class weighting (simple and effective), consider resampling if needed, and always adjust the threshold at the end.

---

## Step 6 — Logging the Experiment in MLflow

Now you put everything together in the complete training script, which is the core of this phase. The key is that every training run is **fully logged** in MLflow: its parameters, its metrics, the resulting model, and visual artifacts that help understand it. Complete `src/models/train.py`:

```python
import json

import matplotlib
matplotlib.use("Agg")  # headless backend to save plots in scripts
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

from src.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    PROJECT_ROOT,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)


def load_params() -> dict:
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)["train"]


def load_split(path):
    df = pd.read_parquet(path)
    return df.drop(columns=[TARGET]), df[TARGET]


def build_model(params: dict):
    """Builds the model specified in the parameters."""
    name = params["model"]
    if name == "logistic_regression":
        return LogisticRegression(
            class_weight="balanced",
            max_iter=params["max_iter"],
            random_state=params["random_state"],
        )
    if name == "xgboost":
        return XGBClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            scale_pos_weight=params["scale_pos_weight"],
            random_state=params["random_state"],
            eval_metric="aucpr",
        )
    raise ValueError(f"Unsupported model: {name}")


def save_confusion_matrix(y_true, y_pred) -> str:
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    path = "confusion_matrix.png"
    disp.figure_.savefig(path, bbox_inches="tight")
    plt.close(disp.figure_)
    return path


def save_pr_curve(y_true, y_proba) -> str:
    disp = PrecisionRecallDisplay.from_predictions(y_true, y_proba)
    path = "pr_curve.png"
    disp.figure_.savefig(path, bbox_inches="tight")
    plt.close(disp.figure_)
    return path


def train() -> None:
    params = load_params()
    X_train, y_train = load_split(TRAIN_PATH)
    X_test, y_test = load_split(TEST_PATH)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=params["model"]):
        model = build_model(params)
        model.fit(X_train, y_train)

        # Prediction with the chosen threshold
        threshold = params["threshold"]
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "pr_auc": average_precision_score(y_test, y_proba),
        }

        # --- MLflow Logging ---
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(save_confusion_matrix(y_test, y_pred))
        mlflow.log_artifact(save_pr_curve(y_test, y_proba))

        # The model is logged with its signature and an input example,
        # which documents the expected format and simplifies deployment
        signature = infer_signature(X_test, y_pred)
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_test.iloc[:5],
        )

        # Metrics also saved to file so DVC can track them
        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"Training completed. Metrics: {metrics}")


if __name__ == "__main__":
    train()
```

It is worth understanding what each part of the logging does, as every call captures a piece of traceability:

`mlflow.log_params` registers all experiment parameters (which model, what hyperparameters, what threshold). This is what will allow you later to know exactly what configuration produced each result.

`mlflow.log_metrics` records the calculated metrics, which are what you will compare across experiments in the UI.

`mlflow.log_artifact` saves visual files: the **confusion matrix** (which displays hits and misses by class at a glance) and the **precision-recall curve** (showing the balance between precision and recall across thresholds). These artifacts make each experiment visually inspectable, not just a bunch of numbers.

`mlflow.sklearn.log_model` saves the trained model itself, and here there are two important details. We pass a **signature** (`signature`), inferred with `infer_signature`, which documents what columns and types the model expects and what it returns; and an **input example** (`input_example`), a few real rows. Both are fundamental for Phase 4: when you deploy the model as an API, the signature guarantees that it receives data in the correct format, and the example serves as live documentation of the expected format. Registering the model with its signature is a production practice that shows you are thinking about deployment right from the training phase.

> **Useful shortcut:** MLflow offers `mlflow.sklearn.autolog()`, a single line that automatically records standard parameters, metrics, and the model when calling `fit()`. It is highly convenient, but in fraud detection, the explicit logging shown above is preferred because the metrics that truly matter (PR-AUC, the chosen threshold) are specific and autolog does not capture them on its own. You can use autolog as a complement so you don't miss anything standard, combining it with manual logging of your custom metrics.

---

## Step 7 — A More Powerful Model: XGBoost

With the baseline recorded, you train a more powerful model to see if it improves performance. **XGBoost** is a gradient boosting algorithm that typically performs very well on tabular data like this and is one of the most widely used in the industry for these types of problems. Thanks to how we structured `build_model`, you don't need to write new code: you only need to change the parameters.

To handle the imbalance, XGBoost uses `scale_pos_weight`, which is best set to the ratio of negative to positive cases. For this fraud dataset, where fraud is around 0.17%, this ratio is roughly 577 (meaning there are about 577 legitimate transactions for every fraudulent one). Ideally, you would calculate this from the data instead of setting it blindly, but as an initial parameter, it works well.

The beauty of having everything parameterized is that each configuration you train is registered as a separate, comparable experiment in MLflow. You simply change the model in `params.yaml`, run it again, and MLflow records a new run next to the previous ones.

---

## Step 8 — Versioned Training Parameters

Just like in Phase 1, the training parameters live in `params.yaml`, versioned, so that each experiment is reproducible and DVC can track them. Add a `train` section to the file:

```yaml
train:
  model: logistic_regression   # change to "xgboost" for the other model
  threshold: 0.5
  random_state: 42
  # Logistic Regression parameters
  max_iter: 1000
  # XGBoost parameters
  n_estimators: 300
  max_depth: 5
  learning_rate: 0.1
  scale_pos_weight: 577
```

Changing the value of `model` between `logistic_regression` and `xgboost`, or adjusting any hyperparameter, and running the training again, produces a new experiment in MLflow. This is how you build, run by run, the comparable history that serves as this phase's deliverable.

---

## Step 9 — Comparing Experiments in the MLflow Interface

After training several models (logistic regression, XGBoost, perhaps variants with different hyperparameters or thresholds), you open the MLflow UI at `http://localhost:5000`, and this is where the value of rigorous tracking materializes.

In the interface, you will see your `fraud-detection` experiment with all its runs listed. You can select and compare them side-by-side: a table will show parameters and metrics for each in parallel columns, letting you see at a glance which configuration yielded the best PR-AUC, which had the best recall, and which struck the best balance. You can **sort** the runs by any metric to find the best one, and click on any run to see its artifacts: the confusion matrix and the PR curve you saved. 

This ability to compare experiments side-by-side, with all their context, is exactly what makes MLflow a traceability tool rather than a simple notepad. And it is, literally, the **success criterion** for this phase: when you open the UI and see a history of comparable experiments with their metrics and artifacts, you have met the objective. Take your time here to identify, with clear criteria, which is your best model, as it will be the one you register and promote in Phase 3.

---

## Step 10 — Rigorous Evaluation: Cross-Validation and Optimal Threshold

Identifying the best model in the UI is a good start, but a truly rigorous evaluation requires two more things that demonstrate a higher level of care. Complete `src/models/evaluate.py`:

```python
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedKFold, cross_val_score


def cross_validate_pr_auc(model, X, y, n_splits: int = 5):
    """Estimates the PR-AUC with stratified cross-validation and its variability."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="average_precision")
    return scores.mean(), scores.std()


def optimal_threshold_f1(y_true, y_proba):
    """Finds the threshold that maximizes F1 on the precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    # precision/recall have one more element than thresholds; we align them
    best_idx = int(np.argmax(f1[:-1]))
    return float(thresholds[best_idx]), float(f1[best_idx])


def cost_optimal_threshold(y_true, y_proba, cost_fp: float, cost_fn: float):
    """Finds the threshold that minimizes total expected business cost."""
    candidates = np.linspace(0.01, 0.99, 99)
    costs = []
    for t in candidates:
        y_pred = (y_proba >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        costs.append(fp * cost_fp + fn * cost_fn)
    best_idx = int(np.argmin(costs))
    return float(candidates[best_idx]), float(costs[best_idx])
```

The first technique is **stratified cross-validation**. Evaluating the model only once on a single test set can be misleading: that result might depend on the luck of that specific split. Cross-validation trains and evaluates the model multiple times on different splits of the data and provides not just an average but also a **standard deviation** that measures how much performance varies. A model with a good average but high variability is less reliable than one with a slightly lower but stable average. We use `StratifiedKFold` (rather than regular K-Fold) specifically because of the class imbalance: it guarantees that each split maintains the proportion of fraud, just as we did with the stratified split in Phase 1.

The second, and highly valuable, technique is **threshold optimization**. As mentioned, the default threshold of 0.5 is almost never optimal in an imbalanced problem. The `optimal_threshold_f1` function traverses the precision-recall curve and finds the threshold that maximizes the F1-score. But there is a version even closer to business reality: `cost_optimal_threshold` finds the threshold that **minimizes the total expected cost**, given the cost of a false positive and a false negative. This is where the decision you documented in Phase 0 about cost asymmetry comes full circle: if you estimated that letting fraud pass costs, say, much more than disturbing a legitimate client, this function translates that estimate into the specific threshold that minimizes your loss. Framing threshold selection in terms of business cost, rather than an abstract metric, is what elevates the project from a technical exercise to a solution designed for the real world.

The threshold you select here is an important decision and **forms part of the model artifact**: it will be versioned along with it, because without the correct threshold, the model does not know where to draw the decision boundary. In Phase 3, you will register it next to the model in the Model Registry.

---

## Step 11 — Connecting Training to the DVC Pipeline

To maintain end-to-end reproducibility, training should also be a stage in the DVC pipeline, extending the data graph into the model graph. Add a `train` stage to `dvc.yaml`:

```yaml
  train:
    cmd: uv run python -m src.models.train
    deps:
      - src/models/train.py
      - data/processed/train.parquet
      - data/processed/test.parquet
    params:
      - train.model
      - train.threshold
      - train.max_iter
      - train.n_estimators
      - train.max_depth
      - train.learning_rate
      - train.scale_pos_weight
      - train.random_state
    metrics:
      - metrics.json:
          cache: false
```

Notice that this stage uses `metrics` instead of `outs` for the `metrics.json` file. This tells DVC to treat that file as trackable metrics: you can run `dvc metrics show` to view them and, very usefully, `dvc metrics diff` to compare how metrics change between commits. It is a way to have metrics versioned and comparable at the pipeline level, complementing MLflow's tracking.

A reasonable question arises here: isn't tracking metrics with **both** MLflow and DVC redundant? It is not, because they serve different purposes. MLflow is for **interactive experimentation**: comparing multiple runs in a visual interface while searching for the best model. DVC is for **pipeline reproducibility**: ensuring that training is reconstructed deterministically as part of the workflow, and that metrics remain tied to each code and data version in Git. Using both demonstrates that you understand the difference between experimenting and producing.

With this stage added, `dvc repro` now reconstructs the entire data and training pipeline, re-running only what has changed.

---

## Step 12 — Tests

We close with tests, maintaining quality discipline. Complete `tests/test_model.py`:

```python
import numpy as np

from src.models.evaluate import cost_optimal_threshold, optimal_threshold_f1
from src.models.train import build_model


def test_build_model_logistic():
    model = build_model(
        {"model": "logistic_regression", "max_iter": 100, "random_state": 42}
    )
    assert model.__class__.__name__ == "LogisticRegression"


def test_optimal_threshold_in_valid_range():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.6])
    threshold, f1 = optimal_threshold_f1(y_true, y_proba)
    assert 0.0 <= threshold <= 1.0
    assert 0.0 <= f1 <= 1.0


def test_cost_threshold_favors_recall_when_fn_is_expensive():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.3, 0.4, 0.6, 0.7])
    # If a false negative is extremely expensive, the optimal threshold must be low
    threshold, _ = cost_optimal_threshold(y_true, y_proba, cost_fp=1, cost_fn=1000)
    assert threshold <= 0.6
```

These tests verify that building the model works, that the optimal threshold falls within a valid range, and, more interestingly, that the cost logic behaves as expected: when a false negative is very costly, the optimal threshold drops to catch more fraud. Testing business logic, and not just the mechanics, is a sign of exceptional care. Run them with `make test`.

---

## Verification: The "Definition of Done"

The phase is complete when the following are met:

- [ ] MLflow is configured with a SQLite backend (prepared for the Model Registry in Phase 3).
- [ ] `train.py` trains a model, handles class imbalance, and logs parameters, metrics, the model (with signature and input example), and visual artifacts to MLflow.
- [ ] You have trained at least two different models (the logistic regression baseline and XGBoost), generating comparable runs.
- [ ] `evaluate.py` provides stratified cross-validation and threshold optimization, including the business-cost-based version.
- [ ] You have made a conscious choice of the decision threshold based on your business metric.
- [ ] Training is a stage in `dvc.yaml`, and `dvc repro` reconstructs the entire pipeline.
- [ ] Tests pass with `make test`.
- [ ] **The key test:** you open the MLflow UI and see a history of side-by-side comparable experiments, with their metrics and artifacts, and you can identify the best one with clear criteria.

The key test is the interface: if you can open MLflow, compare your runs, sort them by PR-AUC, and inspect their artifacts, you have built a truly traceable experimentation process, which is exactly the skill this phase aims to demonstrate.

---

## Deliverables and What's Next

Upon closing Phase 2, you have a rigorous and, above all, fully traced training process: several comparable models in MLflow, each with its complete context; class imbalance handled thoughtfully through a combination of techniques; a robust evaluation with cross-validation; and a conscious choice of decision threshold tied to your business metric. Beyond the models themselves, what you have demonstrated is traceability: the ability to always answer "what produced this result?", which is one of the most highly valued skills in MLOps.

The next step, **Phase 3**, is built directly on this: you will take the best model identified here and manage it as a production artifact using MLflow's **Model Registry**. You will learn to register the model, version it, transition it between `Staging` and `Production` states, and package it alongside its preprocessing and decision threshold so Phase 4 can serve it as an API. The SQLite backend configured in this phase is precisely what makes that registry possible. You have moved from "I know how to train and compare models" to being on the verge of "I know how to manage models as versioned production artifacts."
