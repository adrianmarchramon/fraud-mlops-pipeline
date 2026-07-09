# Phase 8 — Monitoring and Drift Detection

> **This is the crown jewel of the project.** Everything built up to this point (versioned data, registered models, API, containers, CI/CD, orchestration) supports a system that works today. But a Machine Learning model has a characteristic that distinguishes it from traditional software: **it degrades over time, silently**. Reality changes, the learned patterns lose their validity, and nothing crashes or throws an error; its predictions simply get worse without anyone noticing. This phase builds the system that detects this degradation and, by connecting it to the retraining loop you wired up in Phase 7, closes the entire loop: your model stops being a static artifact and becomes a **self-maintaining organism**. This is the piece that, when shown in a video, leaves anyone amazed, because it is exactly what courses almost never cover and companies struggle with the most.

**Phase objective:** Detect when the model degrades in production.
**Duration:** ~2 weeks (weeks 9-10 of the project).
**By the end, you will have:** A system that detects drift by comparing current reality with training data, interactive visual reports, threshold-based alerts, a closed retraining loop, and a demonstration where you inject drift and watch the system react on its own.

---

## The Big Picture: Closing the Loop

The workflow of this phase is what turns all the previous components into a living cycle:

```
   REFERENCE                           CURRENT
   (training data)                     (API prediction logs,
   the distribution that               recorded since Phase 4)
   the model learned
        │                                   │
        └─────────────┬─────────────────────┘
                      ▼
        ┌──────────────────────────────┐
        │  Evidently (drift.py)         │
        │  has reality changed?        │  → interactive HTML report
        └──────────────────────────────┘
                      │
                      ▼
             Is drift > threshold?
                 │         │
                YES        NO
                 │         └──► do nothing
                 ▼
        ┌──────────────────────────────┐
        │  Alert + trigger (Phase 7)   │  → automatic retraining
        └──────────────────────────────┘
                      │
                      └──► the loop goes back to the beginning
```

This is where two decisions from previous phases converge and pay off. On one hand, the **prediction logging** that the API has been recording since Phase 4: those logs are the "current data" that we will compare against the training data. The seed you planted back then germinates right now. On the other hand, the **loop wiring** you set up in Phase 7: the monitoring pipeline already knows how to trigger retraining; it was only missing the detection logic that you build in this phase. You will implement this missing piece and switch on the entire system.

Before starting, add Evidently and, for the demonstration, the HTTP requests library:

```bash
uv add evidently
uv add --dev requests
```

---

## Step 1 — Understanding Drift and Its Types

Before diving into code, it is worth understanding what we are detecting, because there are several types of drift, and knowing how to distinguish them demonstrates mastery. The general concept is that the model learned from a past reality (the training data, which we call **reference**) and operates on a present reality (the data arriving in production, which we call **current**); drift is the divergence between the two.

This manifests in three ways:

**Data drift** is the shift in the distribution of the input features. For example, if transactions suddenly start coming in with much higher amounts, or with feature values different from those in training, the model is operating in an environment it does not recognize. This is the easiest type of drift to detect because it only requires comparing the input distributions without needing to know if the predictions were correct.

**Concept drift** or **target drift** is more insidious: it is the shift in the *relationship* between the features and the outcome. In fraud detection, this is constant because fraudsters adapt their techniques specifically so that patterns that previously indicated fraud stop doing so. Detecting it requires observing how the distribution of the target or predictions changes.

**Prediction quality** is the direct measurement of performance: comparing the model's predictions with the ground truth. The problem, as we saw in the fundamentals, is the **label delay**: in fraud detection, you often do not know if a transaction was fraudulent until days or weeks later when a chargeback or claim arrives. That is why this measurement is delayed, and why data drift on the inputs is so valuable: it gives you an **early warning signal** that something is wrong before the true labels arrive. Mentioning this nuance demonstrates that you understand the problem thoroughly.

In this phase, we will focus primarily on data drift, which provides the earliest and most useful signal, while keeping the structure ready to add other types of drift later.

---

## Step 2 — The Data: Reference and Current

Detection compares two datasets. The **reference** is the distribution the model was trained on; the **current** represents the recent transactions that the API has recorded. These will be the foundation of `src/monitoring/drift.py`:

```python
import json

import pandas as pd

from src.config import PREDICTIONS_LOG, RAW_DATA, TARGET


def load_reference() -> pd.DataFrame:
    """Reference distribution: the features the model was trained on."""
    df = pd.read_csv(RAW_DATA)
    return df.drop(columns=[TARGET]).sample(n=5000, random_state=42)


def load_current() -> pd.DataFrame:
    """Recent production data: transactions that the API has logged."""
    records = []
    with open(PREDICTIONS_LOG) as f:
        for line in f:
            records.append(json.loads(line)["input"])
    return pd.DataFrame(records)
```

Here you can see the direct connection to Phase 4. The `load_current` function reads the `predictions.jsonl` file where the API has been saving every processed transaction, extracts the inputs, and converts them into a DataFrame. Without that logging, this phase would be impossible: you would have no trace of what the model has seen in production. The reference, on the other hand, is a sample of the training data, representing the reality the model knows. Comparing the two means comparing "what the model learned" with "what the model is seeing now."

---

## Step 3 — Detection with Evidently

Now for the heart of this phase: using Evidently to compare both distributions and decide if drift is present. There is a **critical** up-to-date detail you must know here, because this is where almost all tutorials fail: Evidently has two very different generations of APIs. The old one (versions 0.6.7 or earlier) uses `from evidently.report import Report` and `from evidently.metric_preset import DataDriftPreset`. The **current API** is different: it is imported from `evidently` and `evidently.presets`, and, note that the order of the arguments in `run` changed to (current, reference). Using the correct API is exactly what shows you are up to date. Complete `drift.py`:

```python
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import REPORTS_DIR

DRIFT_SHARE_THRESHOLD = 0.5  # if more than 50% of features drift, drift is detected
DRIFT_REPORT_PATH = REPORTS_DIR / "drift_report.html"


def _drifted_share(eval_result) -> float:
    """Extract the proportion of drifted columns from the Evidently result.

    Note: the exact structure may vary depending on the version; inspect
    eval_result.dict() in your installation to confirm the path.
    """
    result = eval_result.dict()
    for metric in result.get("metrics", []):
        value = metric.get("value")
        if isinstance(value, dict) and "share" in value:
            return float(value["share"])
    return 0.0


def drift_share_between(reference, current) -> float:
    """Generates the Evidently report, saves it as HTML, and returns the proportion
    of drifted features. (Current API: run receives current then reference.)"""
    report = Report([DataDriftPreset()], include_tests=True)
    my_eval = report.run(current, reference)

    REPORTS_DIR.mkdir(exist_ok=True)
    my_eval.save_html(str(DRIFT_REPORT_PATH))
    return _drifted_share(my_eval)


def detect_drift() -> bool:
    """Checks if there is significant drift. This function is called by the
    monitoring pipeline from Phase 7."""
    from src.monitoring.drift import load_current, load_reference

    reference = load_reference()
    current = load_current()
    if current.empty:
        return False
    share = drift_share_between(reference, current)
    return share >= DRIFT_SHARE_THRESHOLD
```

It is worth understanding the components, as each serves a specific purpose:

**The data drift preset.** `Report([DataDriftPreset()])` tells Evidently to evaluate the drift of each feature between the two datasets. Evidently automatically chooses the appropriate statistical test for each column based on its type and size, and summarizes the result. Setting `include_tests=True` adds pass/fail checks.

**Report and HTML generation.** `report.run(current, reference)` runs the analysis (remember the new API order: current first, reference second), and `save_html` saves an interactive report to disk. This report is the visual centerpiece of the phase, as you will see in the next step.

**Threshold-based decision.** The `detect_drift` function extracts the proportion of drifted features and compares it to a threshold (`DRIFT_SHARE_THRESHOLD`). If the proportion exceeds the threshold, it declares that drift is present. The threshold is a design decision: lower is more sensitive (retraining on small shifts); higher is more conservative. You choose it based on how much change you can tolerate before retraining.

**The `detect_drift` function**, finally, is what fills the gap you left in Phase 7. The monitoring pipeline called it, but it did not exist yet; now it does. This means that the loop, which was wired but turned off, is now **switched on**.

> **Robustness note:** The exact way to extract the drift proportion from the result may vary across Evidently versions. That is why, when implementing it, you should inspect `my_eval.dict()` in your installation to confirm the exact path, and ensure your tests (Step 8) verify that detection works. The underlying design (reference vs current, threshold, decision) is what matters and remains the same.

---

## Step 4 — The Interactive Report

Take a moment to open the `drift_report.html` file that Evidently generated. This report is one of the most impressive assets of the project. It is an interactive page that, for each feature, shows the overlaid distributions of reference and current data, the result of the statistical drift test, and a summary of how many and which features have drifted. At a glance, you can see exactly what changed and by how much.

The value of this report for your portfolio is twofold. On one hand, it helps you understand and debug the behavior of the model in production. On the other, and very importantly, it is prime visual material for your README and, above all, for the demonstration video: seeing those distributions shift, with features marked in red as "drift detected," immediately and visually communicates that your system is keeping watch over reality. Few things in a junior portfolio convey as much sophistication as a well-presented drift monitoring report.

---

## Step 5 — Threshold-Based Alerts

Detecting drift is useless if no one is alerted. That is why we are adding an **alert** system: when drift exceeds the threshold, a notification is triggered. Add an alert function to `drift.py`:

```python
import logging
import os

import requests

logger = logging.getLogger("monitoring")


def send_alert(message: str) -> None:
    """Triggers an alert: always through logs, and via webhook if configured."""
    logger.warning("DRIFT ALERT: %s", message)

    webhook_url = os.getenv("DRIFT_WEBHOOK_URL")
    if webhook_url:
        try:
            requests.post(webhook_url, json={"text": f"⚠️ {message}"}, timeout=10)
        except requests.RequestException as exc:
            logger.error("Could not send alert to webhook: %s", exc)
```

The alert operates on two levels. It always logs a warning, leaving a trace. Additionally, if you have configured a `DRIFT_WEBHOOK_URL` environment variable (for instance, pointing to a Slack or Discord channel), it also sends a notification to that webhook, allowing you to receive real-time messages. Keeping the webhook configurable via environment variables, just as you did with the MLflow address in Phase 5, aligns with best practices: the alert works out of the box without setup (logging only) and is enhanced if you configure it. This flexibly covers the "alert via log, email, or webhook" requirement.

---

## Step 6 — Closing the Loop

Now you bring everything together and close the loop. The monitoring pipeline from Phase 7 already had the wiring; it only needed `detect_drift` to be real (which it is now) and the alert to be sent. Refine the monitoring flow so that upon detecting drift, it alerts and triggers retraining. Update `pipelines/monitoring_pipeline.py`:

```python
from prefect import flow, get_run_logger, task
from prefect.deployments import run_deployment

from src.monitoring.drift import detect_drift, send_alert


@task(retries=2, retry_delay_seconds=30, name="Check drift")
def check_drift_task() -> bool:
    logger = get_run_logger()
    drift_detected = detect_drift()
    logger.info("Drift detected? %s", drift_detected)
    return drift_detected


@flow(name="monitoring-pipeline", log_prints=True)
def monitoring_pipeline() -> None:
    logger = get_run_logger()
    logger.info("Starting monitoring check")

    if check_drift_task():
        send_alert("Significant drift detected. Triggering retraining.")
        run_deployment(name="training-pipeline/on-demand", timeout=0)
        logger.warning("Retraining triggered")
    else:
        logger.info("No significant drift. Retraining not required.")


if __name__ == "__main__":
    monitoring_pipeline()
```

Here is the project's climax. When this flow runs (scheduled daily, as you configured in Phase 7) and detects drift, three things happen in sequence: an alert is triggered, the retraining pipeline is launched (which validates, preprocesses, trains, registers, and promotes a new model using recent data), and the system returns to its starting point with an updated model. **The loop is closed.** Your system detects its own degradation and cures itself without human intervention. This is literally what a real, production ML system is, and it is something almost no junior portfolio demonstrates.

---

## Step 7 — Simulating Drift: The Demo

You have the system; now you need to **demonstrate it**, and this is the part that makes the biggest impact. You are going to deliberately inject drifted data to prompt your system to detect it and react live. Create a script called `scripts/simulate_drift.py`:

```python
import numpy as np
import requests

API_URL = "http://localhost:8000/predict"


def make_drifted_transactions(n: int = 300, seed: int = 0) -> list[dict]:
    """Generates transactions with a distribution shifted relative to training."""
    rng = np.random.default_rng(seed)
    transactions = []
    for _ in range(n):
        tx = {"Time": float(rng.uniform(0, 200_000))}
        # V features with a shifted mean → drift relative to training
        for i in range(1, 29):
            tx[f"V{i}"] = float(rng.normal(loc=3.0, scale=2.0))
        # Anomalously high amounts
        tx["Amount"] = float(rng.uniform(1_000, 5_000))
        transactions.append(tx)
    return transactions


def main() -> None:
    transactions = make_drifted_transactions()
    for tx in transactions:
        requests.post(API_URL, json=tx, timeout=10)
    print(f"Sent {len(transactions)} drifted transactions to the API")


if __name__ == "__main__":
    main()
```

The full demonstration, which you will capture on video, follows this sequence. With the API and Prefect running, you execute the simulation script, which sends hundreds of transactions with a clearly shifted distribution (different feature means, anomalously high amounts). The API processes and **logs** them, exactly as it would do with real traffic. You then run the monitoring pipeline, which reads these recent data records, compares them to the training reference using Evidently, and **detects the drift**. The alert is triggered, retraining is kicked off, and you see everything happen live in the Prefect UI and the Evidently report.

That moment (anomalous data enters → system detects it → alert sounds → self-retraining triggers) recorded in a two-minute clip is likely the most impressive asset in your entire portfolio. It shows, without you having to explain a thing, that you understand how a living ML system operates. Accompany it with the Evidently report showing the shifted distributions, and you will have a demo that few junior candidates can match.

---

## Step 8 — Tests

We wrap up with tests that verify the drift detection works properly: that it detects drift when it is present and avoids false alarms when it is not. Complete `tests/test_monitoring.py`:

```python
import numpy as np
import pandas as pd

from src.monitoring.drift import drift_share_between


def test_detects_drift_when_distribution_shifts():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {"f1": rng.normal(0, 1, 500), "f2": rng.normal(0, 1, 500)}
    )
    # Clearly shifted distribution
    current = pd.DataFrame(
        {"f1": rng.normal(5, 1, 500), "f2": rng.normal(5, 1, 500)}
    )
    assert drift_share_between(reference, current) > 0.5


def test_no_drift_when_distribution_is_stable():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {"f1": rng.normal(0, 1, 500), "f2": rng.normal(0, 1, 500)}
    )
    # Same distribution
    current = pd.DataFrame(
        {"f1": rng.normal(0, 1, 500), "f2": rng.normal(0, 1, 500)}
    )
    assert drift_share_between(reference, current) < 0.5
```

These tests verify the essential behavior: when the reference and current distributions differ clearly, the drift proportion is high; when they are identical, it is low. They also serve to confirm that your extraction from the Evidently result works correctly: if the extraction path were incorrect, these tests would fail and alert you. Run them with `make test`.

---

## Verification: Definition of "Done"

This phase is complete when the following criteria are met:

- [ ] `drift.py` compares the reference (training) and current data (API logs) using the current Evidently API.
- [ ] An interactive HTML drift report is generated.
- [ ] There is a threshold-based alert system (logs and, optionally, a webhook).
- [ ] The `detect_drift` function fills the gap from Phase 7, and the monitoring pipeline alerts and triggers retraining upon detecting drift.
- [ ] A script exists to simulate drift by injecting shifted data.
- [ ] Tests verify that detection works correctly (detecting drift when present, and not when absent).
- [ ] **The key test:** You inject drifted data, the system detects it, the alert is triggered, and retraining is initiated.

The key test is the culmination of the entire project: if you run the simulation, see how the system detects the drift, how the alert sounds, and how retraining is triggered on its own, you have built a living Machine Learning system. That closed loop (the system's ability to detect its own degradation and self-heal) is precisely what separates an exceptional portfolio project from an ordinary one.

---

## Deliverables and Next Steps

By finishing Phase 8, you have built the crown jewel of the project: a monitoring system that detects drift by comparing current reality with training data, interactive reports that visually show what has changed, threshold-based alerts, and, above all, the closed automatic retraining loop. Your model is no longer an artifact that is trained once and forgotten: it is an organism that watches and maintains itself. This is the piece that demonstrates, better than any other, that you understand how production ML actually works.

The next and final step, **Phase 9**, showcases the value of everything: you will deploy the complete system to the cloud to make it publicly accessible, and, crucially, package it to make an impact. You will write the definitive README, record the demo video (with the drift loop you just built as the absolute centerpiece), and prepare the project presentation. All the technical work from these eight phases now needs a showcase, because, as we saw in the fundamentals, most people will judge your project by the README and the video before they look at the code. You have gone from "I know how to build a self-maintaining ML system" to being just steps away from "I know how to present it to leave companies highly impressed."
