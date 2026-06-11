# Phase 5 — Containerization

> There is a problem so classic in software engineering that it has its own idiom: "it works on my machine." A project that depends on the exact configuration of your computer (your Python version, your system libraries, your environment variables) is fragile and impossible to share with guarantees. This phase solves this problem at its root with **Docker**: you will package your API and its entire environment into an image that runs identically on any machine that has Docker, without relying on anything from your local system. And you will spin it up alongside MLflow using Docker Compose, so that a single command starts the entire system. The ultimate goal is compelling: that someone who does not even have Python installed can run your entire project.

**Goal of this phase:** package everything so it runs identically on any machine.
**Duration:** ~1 week (week 6 of the project).
**By the end, you will have:** an optimized Docker image of your API, a `docker-compose.yml` that spins up the API and MLflow together and connected, and the certainty that anyone can run your system using only Docker.

---

## The Big Picture: Images, Containers, and Services

Before diving into the steps, it is useful to establish two concepts. An **image** is like a frozen recipe: an immutable package that contains your application, its dependencies, and its environment. A **container** is a running instance of that image: what actually runs. You can launch many identical containers from a single image. You will build an image of your API and run it as a container.

The architecture you are going to set up consists of two collaborating services:

```
   ┌──────────────────────────────────────────────────┐
   │                docker-compose                      │
   │                                                    │
   │   ┌─────────────────┐      ┌────────────────────┐ │
   │   │  API Container  │─────▶│  MLflow Container  │ │
   │   │  (FastAPI)      │ http │  (server + data)   │ │
   │   │  port 8000      │      │  port 5000         │ │
   │   └─────────────────┘      └────────────────────┘ │
   │                                    │               │
   │                                    ▼               │
   │                            persistent volume       │
   │                            (registered models)     │
   └──────────────────────────────────────────────────┘
```

The API will no longer use a local file-based MLflow; instead, it will connect to an **MLflow service** running in its own container through Docker's internal network. This is much closer to a real production architecture, where the model registry is an independent service. To make this work, we will make a small change: making the MLflow address configurable via an environment variable, so that the same code works both locally (with SQLite) and in containers (pointing to the MLflow service).

---

## Step 1 — The Multi-Stage Dockerfile

The heart of this phase is the Dockerfile: the recipe that describes how to build your API image. We will write it following current best practices to produce a small, secure, and fast-building image. Create the file `docker/Dockerfile`:

```dockerfile
# syntax=docker/dockerfile:1

# ===== Stage 1: builder =====
FROM python:3.11-slim AS builder

# Copy the uv binary from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# 1) Install ONLY the dependencies (this layer is cached as long as they don't change)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# 2) Copy the project code and install it
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ===== Stage 2: runtime =====
FROM python:3.11-slim

# curl, required for the HEALTHCHECK
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN useradd --create-home appuser

WORKDIR /app

# Copy the built environment and code from the builder
COPY --from=builder --chown=appuser:appuser /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Every decision in this file has a reason, and knowing them is precisely what distinguishes someone who understands Docker from someone who simply copies a Dockerfile from the internet:

**Multi-stage builds.** The file has two stages. The `builder` stage installs dependencies and builds the environment, which requires tools (uv, caches). The final stage (`runtime`) starts from a clean image and only copies the built assets, without carrying over uv or build caches. The result is a smaller final image containing only what is necessary to run the API. Separating build-time dependencies from runtime is the key technique for keeping images lightweight.

**uv inside Docker, done right.** We copy the uv binary from its official image (`COPY --from=ghcr.io/astral-sh/uv:latest`), which is a clean way to make it available on top of the familiar `python:slim` base. The environment variables `UV_COMPILE_BYTECODE=1` (precompiles bytecode, which speeds up startup) and `UV_LINK_MODE=copy` (copies instead of linking, avoiding filesystem warnings) are the recommended settings. Additionally, `--frozen` forces the use of the exact versions locked in your `uv.lock`, guaranteeing that the image has the same dependencies as your local environment.

**Layer caching, the most important optimization.** Pay attention to the order: first, we install **only the dependencies** (using mounts for `uv.lock` and `pyproject.toml`), and only after that do we copy the application code. This is deliberate and very important. Docker caches each layer and reuses the cache if nothing has changed. Since your code changes constantly but your dependencies rarely do, separating these steps means that when you only change code (the common case), Docker reuses the heavy, cached dependency layer and only rebuilds the quick part. This reduces build times from minutes to seconds. The `--mount=type=cache` flags also add a persistent cache for uv downloads between builds.

**The lightweight base image.** We use `python:3.11-slim`, a stripped-down variant of the official Python image that omits unnecessary tools, keeping the image small without sacrificing compatibility.

**The non-root user.** By default, containers run as root, which is a security risk: if someone were to compromise the application, they would have administrative privileges inside the container. We create an unprivileged user named `appuser` and run the API as this user. Running as a non-root user is a standard security practice expected in any production-ready image.

**The integrated health check.** The `HEALTHCHECK` instructs Docker on how to check if the application is healthy by calling the `/health` endpoint you built in Phase 4. Here you can see the payoff of that decision: that endpoint, which seemed trivial, now automatically allows Docker (and, in Phase 9, the deployment service) to know whether your API is alive or needs to be restarted.

**The startup command.** The `CMD` launches the API with Uvicorn, listening on all interfaces (`0.0.0.0`, required inside a container) on port 8000. Note that we do not use `--reload`: that option is for development, not production.

---

## Step 2 — The .dockerignore File

Just as `.gitignore` controls what Git ignores, the `.dockerignore` file controls what is excluded from the Docker build context. This matters for three reasons: it makes the build faster (fewer files to process), the image smaller (unnecessary files are not copied), and more secure (sensitive data does not end up inside the image). Create the `.dockerignore` file in the root directory:

```
# Environment and caches
.venv/
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/

# Version control
.git/

# Data and artifacts (do NOT go in the image)
data/
mlruns/
mlartifacts/
mlflow.db
logs/

# Development
notebooks/
tests/
```

The most important part is the exclusion of **data and artifacts**. The `data/` folder (with the dataset), `mlruns/`, `mlflow.db` (the local MLflow registry), and `logs/` must never enter the image: they are potentially large and are not part of the application. In fact, remember that the model will not reside inside the API image; instead, the API will load it from the MLflow service at runtime, so there is no reason to include the local registry. We also exclude notebooks and tests, which are not needed to run the API in production (tests are run separately, as you will see in Phase 6 with CI). The result is a clean build context and an image containing only the essentials.

---

## Step 3 — Making the MLflow Address Configurable

In previous phases, the MLflow address was hardcoded in the code (`sqlite:///mlflow.db`). To ensure the same code works both locally and inside containers, we will make it configurable via an environment variable, reading it from the environment with a default value. Modify the corresponding line in `src/config.py`:

```python
import os

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
```

This small but important change follows a best practice principle known as environment-based configuration (part of the "12-factor app" methodology): configuration that changes between environments is not hardcoded but injected from the outside. Now, when you run the API locally, it will use SQLite by default; and when you run it in a container, we will pass the `MLFLOW_TRACKING_URI` variable pointing to the MLflow service without changing a single line of code. The same binary, different behavior depending on the environment. This is exactly what you need to make the code portable.

---

## Step 4 — The docker-compose.yml File

Now you will orchestrate the two services. Docker Compose allows you to define a multi-container system and spin it up with a single command. Create the file `docker/docker-compose.yml`:

```yaml
services:
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest   # you can pin a specific version
    command: >
      mlflow server
      --host 0.0.0.0
      --port 5000
      --backend-store-uri sqlite:////mlflow/mlflow.db
      --artifacts-destination /mlflow/artifacts
      --serve-artifacts
    ports:
      - "5000:5000"
    volumes:
      - mlflow-data:/mlflow

  api:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    depends_on:
      - mlflow

volumes:
  mlflow-data:
```

It is useful to understand each part, as this shows how containers fit together into a system:

The **`mlflow` service** spins up an MLflow server in its own container. It uses the official MLflow image and starts the server with a SQLite backend store and an artifact store, both inside the `/mlflow` directory. The `--serve-artifacts` flag enables the server to serve model artifacts as well, meaning the API only needs to communicate with it without accessing files directly. It exposes port 5000.

The **`api` service** builds your image using the Dockerfile you wrote (`build`) and runs it. We pass the environment variable `MLFLOW_TRACKING_URI: http://mlflow:5000`, which your code will read thanks to the change in the previous step. It exposes port 8000.

The **connection between both** is one of the most elegant aspects of Docker Compose. Notice that the MLflow address is `http://mlflow:5000`, using `mlflow` (the service name) as the hostname. Docker Compose creates an internal network where each service is accessible by its name, automatically resolving it to the corresponding container's IP address. This way, the API finds MLflow without you having to know any IP address. The `depends_on` directive indicates that MLflow must start before the API.

The **`mlflow-data` volume** is crucial: containers are ephemeral (if they are deleted, their contents are lost), but registered models must persist. A volume is a storage mechanism that outlives the containers. It is mounted here at `/mlflow`, where MLflow stores its database and artifacts. Thanks to it, your registered models do not disappear upon restart.

> **For added robustness:** `depends_on` only controls the startup order; it does not guarantee that MLflow is ready to accept connections when the API starts. In a more demanding system, you would add a `healthcheck` to the `mlflow` service and use `depends_on` with `condition: service_healthy`, so the API waits for MLflow to be operational. For this project, the basic ordering is usually sufficient.

---

## Step 5 — Build, Populate, and Run

With everything written, you can now start the system. The process consists of three main steps. First, build the API image (from the root of the project):

```bash
docker compose -f docker/docker-compose.yml build
```

Second, because the containerized MLflow registry starts empty, it needs to be **populated** with your model. Start only the MLflow service first, and register your model pointing to it (reusing the scripts from the previous phases, but directing them to the containerized MLflow using the environment variable):

```bash
docker compose -f docker/docker-compose.yml up -d mlflow

# Train and register against the containerized MLflow
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m src.models.train
MLFLOW_TRACKING_URI=http://localhost:5000 uv run python -m src.models.register
```

Here you can see, once again, the value of making the MLflow address configurable: the exact same training and registration scripts you ran locally now populate the containerized registry without any changes. The model remains saved in the persistent volume, ready for the API to load it.

Third, spin up the entire system:

```bash
docker compose -f docker/docker-compose.yml up
```

This starts both containers. You will see in the logs how MLflow initializes and, subsequently, how the API starts up, runs its `lifespan` process, connects to the MLflow service through the internal network, loads the production model, and becomes ready to receive requests.

---

## Step 6 — Verifying the Complete System

Now you will check that everything is working inside the containers. The API is accessible on port 8000 of your host machine, mapped to the container's port. First, test the health check:

```bash
curl http://localhost:8000/health
# Expected response: {"status":"ok"}
```

If it responds with `ok`, it means the API successfully started, connected to the MLflow service, and loaded the model, all inside containers. Also check which model it is serving:

```bash
curl http://localhost:8000/model-info
```

And, most satisfyingly, open `http://localhost:8000/docs` in your browser: you will see the same interactive documentation from Phase 4, but now served from a container, and you will be able to send a test transaction and receive a live prediction. The difference from Phase 4 is profound, even if it is not visible on the surface: this no longer depends on your Python installation or your local environment—it runs in an isolated, reproducible container.

---

## Step 7 — Verifying the Image Size

A well-built image should be reasonably small. Check the size of the image you have created:

```bash
docker images
```

Find your API image in the list. Thanks to the multi-stage build, the `slim` base image, and the `.dockerignore` file, it should have a contained size. It is important to be realistic about expectations: a Machine Learning image will never be tiny because it brings along heavy libraries like numpy, scikit-learn, XGBoost, and MLflow, which take up considerable space. It is normal for it to be around several hundred megabytes or slightly more. What matters is that the techniques you applied (multi-stage build, lightweight base image, and the exclusion of data and artifacts) prevent it from bloating unnecessarily. If you want to investigate what is taking up space, running `docker history` will show you the weight of each layer, which is useful for identifying optimization opportunities. Being able to reason about image size, and understanding that ML dependencies establish a baseline, demonstrates good engineering judgment.

---

## Verification: The "Definition of Done"

The phase is complete when the following requirements are met:

- [ ] The `Dockerfile` is multi-stage, uses uv correctly, starts from a lightweight image, runs as a non-root user, and includes a health check.
- [ ] Layer caching is correctly ordered (dependencies before application code).
- [ ] The `.dockerignore` file excludes data, artifacts, caches, and development assets.
- [ ] The MLflow address is configurable via an environment variable.
- [ ] The `docker-compose.yml` file spins up the API and MLflow connected via the internal network, with a persistent volume for the models.
- [ ] You can build the image, populate the registry, and spin up the system using Docker Compose.
- [ ] The API responds correctly from inside the container (health, model-info, and predictions via `/docs`).
- [ ] **The key test:** someone without Python installed can clone your project and run it entirely using only Docker (`docker compose up`).

The key test is the most revealing: if someone who does not have Python (or any of your libraries) can clone the repository and, with a single `docker compose up`, have your entire system working, you have achieved total portability. This independence from individual environments is exactly what containerization delivers, and what makes your project truly executable by anyone, anywhere.

---

## Deliverables and What Comes Next

By closing Phase 5, you have a packaged and portable system: an optimized Docker image of your API (multi-stage, lightweight, secure, with a health check), a multi-container setup that spins up the connected API and MLflow with a single command, and the guarantee that it runs identically on any machine, eliminating the "it works on my machine" problem. You have resolved the challenge of environment reproducibility, which is one of the core skills expected of any engineer today.

The next step, **Phase 6**, automates everything you have been doing manually so far and is where the project makes the definitive leap from "student" to "professional": you will set up **CI/CD with GitHub Actions**. For every change you push, a pipeline will automatically run linting and tests, build the Docker image you just defined, and deploy it if everything passes. The image you created in this phase is exactly what the pipeline in the next phase will build and publish automatically, and you will add a validation gate that prevents a worse model from reaching production. You have gone from "I know how to package the system" to being on the verge of "I know how to automate its quality and delivery."
