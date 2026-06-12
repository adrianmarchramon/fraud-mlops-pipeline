# Phase 6 — Automated CI/CD

> This is the phase where the project makes the definitive leap from "student" to "professional." Until now, you ran tests manually, checked formatting when you remembered, and built the image whenever you needed it. **CI/CD** automates all of that: it defines processes that automatically run quality checks on every change you push, and if everything passes, build and deploy your application. CI/CD applied to Machine Learning is exactly what companies want to see and rarely find in junior profiles; it is probably the most significant piece when it comes to making an impression, because it demonstrates that you understand that the work doesn't end when the code works on your machine, but when it is delivered in a reliable and repeatable manner.

**Phase objective:** Automate testing, building, and deployment on every change.  
**Duration:** ~1 week (week 7 of the project).  
**Upon completion, you will have:** A CI/CD pipeline with GitHub Actions that protects the main branch (no change that breaks tests or formatting can enter), a gate that prevents deploying a model that does not meet minimum quality standards, and the automatic building and publishing of your Docker image on every approved change.

---

## The Big Picture: CI, CD, and the Model Gate

It is useful to distinguish between two concepts that are often mentioned together. **CI** (Continuous Integration) is the automation of quality checks: on every change, linting, formatting, and tests are executed, ensuring that nothing broken enters the project. **CD** (Continuous Deployment) is the automation of delivery: when an approved change reaches the main branch, the application is automatically built and deployed.

Here is the complete workflow you will set up:

```
   You push / open a Pull Request
            │
            ▼
   ┌──────────────────────────────────┐
   │  CI (ci.yml)                     │
   │  - linting (ruff check)          │  ← on EVERY change
   │  - formatting (ruff format --check)│
   │  - tests (pytest)                │
   │  - model quality gate            │
   └──────────────────────────────────┘
            │  if everything passes
            ▼
   ┌──────────────────────────────────┐
   │  Branch Protection               │  ← cannot merge to main
   │  requires CI to be green         │     if CI fails
   └──────────────────────────────────┘
            │  PR is merged to main
            ▼
   ┌──────────────────────────────────┐
   │  CD (cd.yml)                     │
   │  - builds the Docker image       │  ← only on main
   │  - publishes it to GHCR          │
   │  - (deploys → Phase 9)           │
   └──────────────────────────────────┘
```

There is also a specific MLOps component that distinguishes this pipeline from a standard software pipeline: a **model validation gate**, a check that prevents a model that does not reach a minimum quality threshold from being deployed. This is the same "quality gate" concept you saw in Phase 3's automatic promotion, now integrated into the delivery pipeline.

---

## Step 1 — GitHub Actions Basics

GitHub Actions is the CI/CD tool integrated into GitHub, free for projects like this. Before writing the pipelines, it is helpful to understand its terminology, which is straightforward. A **workflow** is an automated process defined in a YAML file inside the `.github/workflows/` directory. Each workflow is triggered by an **event** (such as a `push`, the opening of a *pull request*, etc.). A workflow contains one or more **jobs**, which run on clean virtual machines called **runners**. Each job contains a sequence of **steps**, and each step either runs a command or uses an **action**—a reusable component that someone has already built (for example, to install uv or build a Docker image). With this terminology, the files you will write are easy to read.

---

## Step 2 — The CI Workflow

The first pipeline runs quality checks on every change. Create the file `.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Set up uv
        uses: astral-sh/setup-uv@v8.1.0
        with:
          enable-cache: true

      - name: Install dependencies
        run: uv sync --locked --dev

      - name: Linting
        run: uv run ruff check .

      - name: Check formatting
        run: uv run ruff format --check .

      - name: Run tests
        run: uv run pytest
```

It is worth understanding each part, as each reflects current best practices:

The **trigger** (`on: [push, pull_request]`) causes the workflow to run on every push and pull request, which is exactly when you want to verify quality.

The **checkout action** (`actions/checkout@v6`) clones your repository onto the runner, so that subsequent steps have access to the code.

The **uv installation** uses Astral's official action. Pay attention to an important, up-to-date detail: we use the full version `@v8.1.0`, not `@v8`. This is because `setup-uv` stopped publishing floating tags like `@v8`, so you must pin the exact version. Knowing this nuance demonstrates that you are up to date. The `enable-cache: true` option caches the uv package store between runs, so subsequent builds do not re-download dependencies and run much faster.

The **dependency installation** uses `uv sync --locked --dev`. The `--locked` flag is key in CI: it makes the pipeline **fail if `uv.lock` is out of sync** with `pyproject.toml`, guaranteeing that the versioned lockfile is consistent. The `--dev` flag includes development dependencies (pytest, ruff) required for the checks.

There are three **quality checks**. `ruff check .` runs the linter. `ruff format --check .` verifies that the code is correctly formatted **without modifying it** (the `--check` flag is important: in CI you want to verify, not change). Remember that, as discussed in Phase 0, ruff also acts as a formatter, replacing black; this is why there is no separate step for black here. And `pytest` runs your entire test suite, which you have built phase by phase.

If any of these steps fail, the entire job fails, which will be shown on GitHub with a red mark on the commit or pull request. If they all pass, you get a green checkmark.

> **Optional — version matrix:** You could run the tests on multiple Python versions simultaneously (`strategy: matrix: python-version: ["3.11", "3.12"]`), which verifies compatibility. For this project, which targets Python 3.11, a single version is sufficient, but mentioning that you are familiar with build matrices demonstrates depth.

---

## Step 3 — Protect the Main Branch

The CI workflow on its own reports whether the quality is correct, but it does not prevent a flawed change from being merged. For that, you need to enable **branch protection**, which is a GitHub setting (not a YAML configuration). In the repository settings, under the branches section, add a protection rule for `main` that **requires the CI workflow to pass** before allowing a pull request to be merged.

With this, you establish a professional workflow pattern: no one writes directly to `main`; changes arrive through pull requests, and a pull request cannot be merged until CI is green. This is precisely what the task "fail the merge if something doesn't pass" means. The effect is powerful: the main branch remains **permanently healthy** because it is impossible for code that fails checks to enter it. And this, as you will see, is what allows the deployment process to trust `main`.

---

## Step 4 — The Model Validation Gate

Here you add the specific MLOps component: a check that prevents deploying a model that does not meet minimum quality standards. The idea is simple yet important: in standard software, "passing tests" means the code functions; in Machine Learning, you also need to guarantee that the **model** is good enough. A model whose performance has degraded should never reach production, just as you would not deploy failing code.

We implement this gate as another test, which checks that the model's metric exceeds a minimum threshold. Create `tests/test_model_quality.py`:

```python
import json

import pytest

from src.config import PROJECT_ROOT

# Minimum quality threshold: below this, it is not deployed
MIN_PR_AUC = 0.75
METRICS_FILE = PROJECT_ROOT / "metrics.json"


@pytest.mark.skipif(
    not METRICS_FILE.exists(),
    reason="metrics.json not yet available (train the model first)",
)
def test_model_meets_minimum_quality():
    metrics = json.loads(METRICS_FILE.read_text())
    pr_auc = metrics["pr_auc"]
    assert pr_auc >= MIN_PR_AUC, (
        f"PR-AUC {pr_auc:.4f} is below the required minimum ({MIN_PR_AUC}). "
        "Deployment blocked."
    )
```

Here again, we see the value of a decision you made earlier. In Phase 2, you configured `metrics.json` as a DVC metric with `cache: false`, meaning this file lives in Git and is versioned. Thanks to this, the gate is **self-contained**: the CI can read the metric directly from the cloned repository, without needing to download data or connect to MLflow. The gate reads the PR-AUC of the last registered training run and checks that it exceeds the minimum; if it does not, the test fails, CI turns red, and due to branch protection, the change cannot be merged or deployed.

You should set the `MIN_PR_AUC` threshold sensibly: below your model's current performance (so it does not block good models), but high enough to detect a regression (a model that suddenly performs worse). The `skipif` is a safety net for the initial runs, before any training run exists. Since this gate is just another test, it runs within `pytest` inside the CI workflow, thereby automatically protecting every change. Having a quality gate specifically for the model, and not just for the code, shows that you understand what distinguishes MLOps from standard software development.

---

## Step 5 — The CD Workflow

The second pipeline builds and publishes your Docker image when an approved change reaches `main`. Create the file `.github/workflows/cd.yml`:

```yaml
name: CD

on:
  push:
    branches: [main]

permissions:
  contents: read
  packages: write

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v6

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract metadata (tags, labels)
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

Each part deserves an explanation, as this is where you see how delivery is automated:

The **trigger** (`on: push: branches: [main]`) ensures this workflow only runs when a change reaches `main`—that is, after a pull request has been merged. This is where the elegance of the design lies: since branch protection guarantees that only changes passing the CI (including the model gate) are merged to `main`, this pipeline can **trust** that the code and model are valid. The validation gate governs what reaches `main`, while deployment builds from an already validated `main`. There is no need to repeat the checks here.

The **permissions** (`packages: write`) are required for the workflow to publish the image to the GitHub Container Registry.

The **GHCR login** (GitHub Container Registry) uses the `GITHUB_TOKEN`, a token that GitHub automatically provides to each workflow, without requiring you to configure credentials. It authenticates you so you can publish images.

The **metadata generation** automatically creates the appropriate tags and labels for the image (for example, a tag with the commit hash), following best practices.

The **build and push** step uses the official Docker action to build the image from your `Dockerfile` (the one you wrote in Phase 5) and publish it to GHCR. This is where the payoff of the previous phase comes in: the image you defined is now built and published completely automatically on every approved change.

> **Cloud Deployment (Phase 9):** This workflow publishes the image, which is the first step of deployment. The final step—launching that image on a cloud service (Render, Railway, Fly.io, or Modal)—will be addressed in Phase 9, where an additional step here or connecting the service directly to the repository is often sufficient. For now, having the image automatically built and published represents the bulk of the CD work.

---

## Step 6 — README Badges

Finally, a low-effort, high-impact detail: adding **badges** to your README to show the status of your pipeline. GitHub Actions automatically generates a badge showing the status of each workflow. Add something like this to the beginning of your README:

```markdown
![CI](https://github.com/YOUR_USERNAME/mlops-fraud-pipeline/actions/workflows/ci.yml/badge.svg)
```

This badge will show a green "passing" status when your CI is healthy, and a red one if it fails. Remember what we discussed in the fundamentals about who the project is meant to impress: a technical recruiter often does not dive into the code, but instead judges by surface signals, and a green CI badge is one of those signals that communicates professionalism at a glance. If you want to go further, you can add a **test coverage** badge by integrating a service like Codecov, which measures what percentage of your code is covered by tests, though this is optional.

---

## Verification: The "Definition of Done"

The phase is complete when the following requirements are met:

- [ ] The `ci.yml` workflow runs linting, formatting checks, and tests on every push and pull request.
- [ ] The installation uses the full version of `setup-uv` and `uv sync --locked`.
- [ ] Branch protection requires the CI to pass before merging to `main`.
- [ ] The model validation gate (`test_model_quality.py`) checks for a minimum metric threshold and runs within the CI.
- [ ] The `cd.yml` workflow builds and publishes the Docker image to GHCR when a change reaches `main`.
- [ ] The README displays a badge with the CI status.
- [ ] **The key test:** You push, see the checks running on GitHub, and if everything passes, the image builds and publishes on its own.

The key test is the full cycle: if you push (or open a pull request), see the checks running in the Actions tab, and verify that an approved change to `main` automatically triggers the building and publishing of the image, you have built a real CI/CD pipeline. That automation—where quality is verified and delivery occurs without your manual intervention—is the essence of what separates a professional project from a student project.

---

## Deliverables and What's Next

Upon completing Phase 6, your project features professional-grade quality and delivery automation: a pipeline that verifies linting, formatting, and tests on every change; a protected main branch that is impossible to break; a specific gate that prevents deploying a model that does not meet the quality bar; and the automatic building and publishing of your Docker image. You have demonstrated, perhaps more than in any other phase, that you understand how to deliver software reliably and repeatably, which is exactly what companies value most and find least in junior profiles.

The next step, **Phase 7**, focuses on **orchestration**: up to this point, your training and monitoring scripts are standalone pieces that you execute with commands, but a production system needs to chain them into workflows that run in order, with automatic retries if something fails, scheduling, and visibility into what is happening. Using Prefect or ZenML, you will convert those scripts into orchestrated, programmable workflows, laying the groundwork for automatic retraining. The automation you built here (the code lifecycle) will be complemented by the orchestration in the next phase (the data and model lifecycle). You have progressed from "knowing how to automate quality and delivery" to being on the verge of "knowing how to orchestrate ML workflows."
