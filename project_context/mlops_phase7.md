# Phase 7 — Orchestration

> Until now, your training and validation scripts have been isolated pieces: you run them manually, one by one, and if something fails, you only find out when you check. A production system needs something more: chaining these pieces into **flows** that run in order, automatically retry if a transient failure occurs, can be scheduled to run on their own, and are observable (allowing you to see what is happening at any given moment on a dashboard). This is called **orchestration**. In this phase, you will convert your scripts into orchestrated flows using Prefect, and, most importantly, you will build the skeleton of the project's **closed loop**: a scheduled monitoring pipeline that, when it detects issues, will automatically trigger retraining. This piece prepares the ground for the highlight of Phase 8.

**Phase objective:** Convert your scripts into orchestrated and schedulable flows.
**Duration:** ~1 week (week 8 of the project).
**By the end, you will have:** A training pipeline as a chained flow with retries and logging, a scheduled monitoring pipeline, both visible on the Prefect dashboard, and the mechanism to trigger retraining manually or in response to an event.

---

## The Big Picture: Two Flows and a Loop

In this phase, you will build two flows that, together, form the skeleton of the active system:

```
   ┌─────────────────────────────────────────────────────┐
   │  TRAINING PIPELINE (training_pipeline)              │
   │                                                      │
   │  validate → preprocess → train → register            │
   │  (with retries and logging at each stage)            │
   │                                                      │
   │  triggered: manually  OR  by event (drift)           │
   └─────────────────────────────────────────────────────┘
              ▲
              │ triggers retraining
              │
   ┌─────────────────────────────────────────────────────┐
   │  MONITORING PIPELINE (monitoring_pipeline)          │
   │                                                      │
   │  checks for drift on recent data                     │
   │  ┌─ drift detected? ─ YES ─► triggers training ──────┼──┘
   │  └─ drift detected? ─ NO  ─► does nothing            │
   │                                                      │
   │  runs: scheduled (e.g., daily)                       │
   └─────────────────────────────────────────────────────┘
```

The **training pipeline** chains the stages you already built in previous phases (validating data, preprocessing, training the model, registering and promoting it). The **monitoring pipeline** runs on a schedule, checks if the data has drifted, and if so, triggers the training pipeline. Together, they close the loop: the system monitors and retrains itself. In this phase, you will set up all this orchestration machinery; the actual drift detection (using Evidently) will be implemented in Phase 8, fitting into the placeholder we prepare here.

### Prefect vs. ZenML, and the Relationship with DVC

As we saw in the fundamentals, you have two good choices for an orchestrator. **Prefect** is simpler and more general-purpose; **ZenML** is specifically geared toward ML. We will use **Prefect** for its simplicity and gentle learning curve, which fits a solo project better. Either one would be a defensible choice.

It is worth clarifying a potential question, as it shows you understand what each tool is for: doesn't DVC already handle orchestration? Not exactly, and the distinction is important. **DVC** (which you set up in Phases 1 and 2) is concerned with **reproducibility**: it ensures that the data pipeline is regenerated in a deterministic and cached way, answering "what runs and with what result." **Prefect** is concerned with **orchestration**: the "when and with what reliability" it runs, along with scheduling, retries, observability, and event-driven triggers. They are complementary layers, just as MLflow and DVC complemented each other in Phase 2. In fact, you have two valid ways to combine them: having Prefect tasks call your Python functions directly (which we will cover for its fine-grained visibility), or having a Prefect task invoke `dvc repro` to leverage DVC's cache. Mentioning that you know both options and their trade-offs demonstrates good judgment.

Before starting, install Prefect:

```bash
uv add prefect
```

---

## Step 1 — Set Up Prefect

Prefect needs a server to manage the state of the flows and serve the control panel. For development, you can spin up a local server. In a separate terminal (leave it running):

```bash
uv run prefect server start
```

This starts the Prefect server and dashboard at `http://localhost:4200`. If at any point Prefect cannot find the server, you can point to it explicitly with `prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api`.

It is helpful to remember Prefect's terminology, which is simple. A **task** is a unit of work, a function decorated with `@task`. A **flow** is a process that orchestrates several tasks, a function decorated with `@flow`. And a **deployment** is a flow registered on the server that can be scheduled or triggered remotely. With this, the code you write will read naturally.

---

## Step 2 — From Functions to Tasks: The Training Pipeline

Here is the core idea of the phase, and also where you see the reward of having structured your code well in previous phases: since your logic is already encapsulated in clean functions (validate, preprocess, train, register), converting them into an orchestrated flow is just a matter of wrapping them as Prefect tasks. Complete `pipelines/training_pipeline.py`:

```python
from prefect import flow, get_run_logger, task

from src.data.preprocess import preprocess as run_preprocess
from src.data.validate import validate_raw_data
from src.models.register import main as run_register
from src.models.train import train as run_train


@task(retries=3, retry_delay_seconds=10, name="Validate data")
def validate_task() -> dict:
    logger = get_run_logger()
    report = validate_raw_data()
    logger.info(
        "Validation OK: %s rows, fraud rate %.4f",
        report["n_rows"], report["fraud_rate"],
    )
    return report


@task(retries=2, retry_delay_seconds=10, name="Preprocess data")
def preprocess_task() -> None:
    logger = get_run_logger()
    run_preprocess()
    logger.info("Preprocessing completed")


@task(retries=2, retry_delay_seconds=30, name="Train model")
def train_task() -> None:
    logger = get_run_logger()
    run_train()
    logger.info("Training completed")


@task(retries=2, retry_delay_seconds=10, name="Register and promote")
def register_task() -> None:
    logger = get_run_logger()
    run_register()
    logger.info("Registration and promotion completed")


@flow(name="training-pipeline", log_prints=True)
def training_pipeline() -> None:
    logger = get_run_logger()
    logger.info("Starting the training pipeline")
    validate_task()
    preprocess_task()
    train_task()
    register_task()
    logger.info("Training pipeline finished")


if __name__ == "__main__":
    training_pipeline()
```

It is worth understanding what this wrapper provides, because every detail solves a real production problem:

**Automatic retries.** Notice `@task(retries=3, retry_delay_seconds=10, ...)`. This tells Prefect that if a task fails, it should retry it up to three times, waiting ten seconds between attempts. This is incredibly valuable for transient failures: a network connection that drops for a split second, a service that is slow to respond. Without orchestration, a single temporary failure breaks the entire process; with it, the system self-heals. We configure more retries on tasks prone to transient failures and fewer on computationally expensive ones.

**Structured logging.** Each task obtains a logger using `get_run_logger()` and logs its progress. These messages appear in the Prefect dashboard associated with each run, so you have a clear audit trail of what happened and when, without having to dive into console outputs.

**Order orchestration.** Within the flow, tasks are called sequentially, and Prefect runs them in that order, one after the other. Since each stage depends on the output of the previous one (preprocessing produces the data that training consumes), this sequential execution ensures everything happens in the correct order. Validation comes first, acting as a gatekeeper: if the data does not pass the contract, the flow stops before training on bad data.

**Clean reuse.** Notice that we haven't rewritten any logic: the tasks simply invoke the functions you already had (`validate_raw_data`, `run_preprocess`, `run_train`, `run_register`). This is possible thanks to the separation of concerns you maintained since Phase 1, and it is exactly what makes adding orchestration so cheap. If the logic had been tangled up in notebooks, this would be a nightmare.

> **For parallel execution:** If you have independent tasks in the future that can run at the same time, Prefect supports this using `.submit()`, which launches tasks concurrently. For a sequential pipeline like this, calling them in order is the correct and simplest approach.

---

## Step 3 — The Scheduled Monitoring Pipeline

The second flow is the one that closes the loop. It runs on a schedule, checks for drift in recent data, and if drift is found, triggers retraining. Complete `pipelines/monitoring_pipeline.py`:

```python
from prefect import flow, get_run_logger, task
from prefect.deployments import run_deployment


@task(retries=2, retry_delay_seconds=30, name="Check drift")
def check_drift_task() -> bool:
    logger = get_run_logger()
    # The real detection with Evidently is implemented in Phase 8.
    # For now, create a minimal version of detect_drift that returns False.
    from src.monitoring.drift import detect_drift
    drift_detected = detect_drift()
    logger.info("Drift detected? %s", drift_detected)
    return drift_detected


@flow(name="monitoring-pipeline", log_prints=True)
def monitoring_pipeline() -> None:
    logger = get_run_logger()
    logger.info("Starting monitoring check")

    if check_drift_task():
        logger.warning("Drift detected: triggering retraining")
        run_deployment(name="training-pipeline/on-demand", timeout=0)
    else:
        logger.info("No significant drift. Retraining skipped.")


if __name__ == "__main__":
    monitoring_pipeline()
```

This is where the magic of the closed loop happens, so it is worth understanding it well. The task `check_drift_task` checks for drift; its actual logic, using Evidently, will be built in Phase 8, but the orchestration machinery is set up here. The key piece is `run_deployment(name="training-pipeline/on-demand", timeout=0)`: this line **triggers the training pipeline from the monitoring pipeline**. Setting `timeout=0` means it does not wait for it to finish (it fires and forgets). This way, when monitoring detects drift, the system triggers retraining on its own, without human intervention.

This is exactly what turns your project into a **living ML system** rather than a static model: the "monitor → detect drift → retrain → deploy" loop is closed. In this phase, you have built the wiring for that loop; in Phase 8, you will turn it on by implementing the actual drift detection. As we noted earlier, for now, create a minimal version of `detect_drift` in `src/monitoring/drift.py` that simply returns `False` so the orchestration is runnable; Phase 8 will replace it with the real detection.

---

## Step 4 — Deploy: Schedule and Trigger

Having the flows written is one thing; making them run on a schedule or allowing them to be triggered is another. For that, we create **deployments**. There is an important up-to-date detail here: Prefect's deployment model changed with version 3. If you see tutorials using `Deployment.build_from_flow`, they are using the old Prefect 2 API, which is deprecated. The modern approach uses `to_deployment` and `serve`. Create a file, for example `pipelines/serve.py`, to deploy both flows:

```python
from prefect import serve

from pipelines.monitoring_pipeline import monitoring_pipeline
from pipelines.training_pipeline import training_pipeline

if __name__ == "__main__":
    # Training: available on-demand (manual or event-driven)
    training_deploy = training_pipeline.to_deployment(name="on-demand")

    # Monitoring: scheduled daily at 6:00 AM (cron)
    monitoring_deploy = monitoring_pipeline.to_deployment(
        name="daily",
        cron="0 6 * * *",
    )

    serve(training_deploy, monitoring_deploy)
```

It is helpful to understand each part:

The **training deployment** is created with `to_deployment(name="on-demand")`, without a schedule: it exists to be triggered whenever needed, either manually or by the drift event (it is precisely the deployment that `run_deployment` invokes from the monitoring flow).

The **monitoring deployment** adds `cron="0 6 * * *"`, a cron expression that means "every day at 6:00 AM". Thus, Prefect will run the drift check automatically every day, without anyone having to trigger it. For other frequencies, simply adjust the cron expression.

The **`serve`** function runs both deployments in a single process that remains listening: it runs the monitoring when scheduled, and the training when triggered. As long as this process is running (alongside the Prefect server), your system is active and monitoring itself.

With this, you have the **three ways of triggering** a pipeline required for this phase. **Manually** from the command line (`prefect deployment run "training-pipeline/on-demand"`) or from the dashboard with a single click. **Scheduled**, like the daily monitoring. And **event-driven**, when the monitoring detects drift and triggers retraining. This flexibility is the hallmark of a well-orchestrated system.

---

## Step 5 — The Prefect Dashboard

Take a moment to look at what you have built on the Prefect dashboard at `http://localhost:4200`. It is the visible deliverable of this phase and where the orchestration comes to life before your eyes. In the dashboard, you will see your flows and deployments, and for each execution, you can inspect the status of each task (running, completed, failed, retrying), the logs you recorded, the timings of each stage, and the schedule of future runs. If a task fails and retries, you will see it reflected in real time. This observability (knowing at all times what is happening, what failed, and what was recovered) is precisely what distinguishes an orchestrated system from a blind set of scripts.

---

## Step 6 — Run and Verify

Verify that everything is working. With the Prefect server running, launch the training pipeline directly:

```bash
uv run python -m pipelines.training_pipeline
```

You will see in the console, and especially in the dashboard, how the flow starts and each task runs in order: validate, preprocess, train, register, each with its own logging. That is the success criterion of this phase: you launch the pipeline with a command and watch each stage execute.

To test retries, which are one of the greatest benefits of orchestration, you can intentionally trigger a failure (for example, by temporarily renaming the data file so that validation fails) and watch on the dashboard how Prefect automatically retries the task the configured number of times before giving up. Seeing this self-healing behavior is proof that you have built a truly resilient flow, not a fragile sequence.

To run the complete system with its deployments and schedule, run the serving process (in a separate terminal, alongside the server):

```bash
uv run python -m pipelines.serve
```

From there, the monitoring will run on its own daily, and you will be able to trigger the training whenever you want from the dashboard or the command line.

---

## Verification: The "Definition of Done"

The phase is complete when the following criteria are met:

- [ ] The training pipeline is written as a Prefect flow that chains validate → preprocess → train → register, with retries and logging on each task.
- [ ] The monitoring pipeline is written as a flow that checks for drift and, if detected, triggers retraining (with actual detection pending for Phase 8).
- [ ] Deployments are defined using the modern Prefect 3 model (`to_deployment` + `serve`), not the deprecated API.
- [ ] Monitoring is scheduled (cron) and training can be triggered manually or via an event.
- [ ] Flows and their runs are visible on the Prefect dashboard.
- [ ] **The key test:** You run the training pipeline with a command and see each stage execute in the dashboard, with automatic retries if a task fails.

The key test is observable execution: if you launch the pipeline and see in the dashboard how each stage executes in order, and confirm that a failing task retries on its own, you have built real orchestration. This combination of ordered execution, resilience to failure, and visibility is the essence of what orchestration brings to an ML system.

---

## Deliverables and What's Next

By wrapping up Phase 7, your scripts have evolved into an orchestrated system: a resilient training pipeline with retries and logging, a scheduled monitoring pipeline, both observable on a dashboard, and the mechanism to trigger retraining manually or in response to an event. More importantly, you have built the **wiring of the closed loop**: the machinery that allows the system to monitor and retrain itself. You only need to turn on the piece that detects when to retrain.

The next step, **Phase 8**, is that very piece, and it is the highlight of the project: you will implement **monitoring and drift detection using Evidently**. You will build the `detect_drift` function that we left as a placeholder here, using the prediction logs that the API has been recording since Phase 4 to compare the current real-world data with the training data. Once you connect that detection to the retraining trigger you have wired in this phase, you will close the entire loop: drift detected → automatic retraining. That is the moment that, captured on video, leaves anyone impressed. You have moved from "knowing how to orchestrate flows" to being on the verge of "knowing how to build an ML system that maintains itself."
