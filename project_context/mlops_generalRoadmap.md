# Roadmap: End-to-End MLOps Pipeline

> Portfolio project designed to demonstrate that you understand a Machine Learning model is not just a notebook, but a **complete production system**. This is the type of project most lacking in junior profiles and the one companies seek the most.

**Estimated duration:** 10–12 weeks (part-time, ~10-15h/week)  
**Level:** Intermediate-Advanced  
**Final result:** A system that ingests data, trains versioned models, deploys them as an API, monitors them in production, and automatically retrains when degradation is detected.

---

## 1. Exactly What You Will Build

An end-to-end system that covers the complete lifecycle of a Machine Learning model:

```
       Raw data
            │
            ▼
      [ Ingestion + validation ]  ──►  versioned with DVC
            │
            ▼
      [ Training ]  ──►  experiments tracked with MLflow
            │
            ▼
      [ Model Registry ]  ──►  Model Registry (MLflow)
            │
            ▼
      [ Inference API ]  ──►  FastAPI + Docker
            │
            ▼
      [ Deployment ]  ──►  Cloud (Render / Railway / Fly.io / Modal)
            │
            ▼
      [ Monitoring ]  ──►  Evidently (data & concept drift)
            │
            ▼
      [ Automatic retraining ]  ──►  orchestrated with Prefect/ZenML
            │
            └──────────► back to start (closed loop)
```

All orchestrated, versioned, automated with CI/CD, and equipped with real-time monitoring dashboards.

---

## 2. The ML Problem You Will Solve

**Main recommendation: Fraud detection in financial transactions.**

Why this is a suitable choice for your profile:

- **It connects with an interest in cybersecurity** → fraud is applied security. This makes your portfolio coherent.
- **Drift is natural and demonstrable** → fraud patterns constantly evolve, so drift monitoring makes real sense rather than being artificial.
- **It is a business problem that companies understand** → in interviews, you can discuss the cost trade-offs of false positives vs. false negatives.
- **High-quality public dataset available** → see the resources section.

**Equally valid alternatives** (choose based on what motivates you):

| Problem | Suggested Dataset | Why it works |
|----------|------------------|--------------|
| Customer churn prediction | Telco Customer Churn (Kaggle) | Classic, natural drift in customer behavior |
| Demand / sales forecasting | Store Sales (Kaggle) | Time series, seasonal drift |
| Sentiment classification | Amazon Reviews | NLP, vocabulary drift over time |

> **Key advice:** The specific problem matters less than the quality of the *infrastructure* you build around it. Companies won't be impressed by your F1-score; they will be impressed by your pipeline. Do not waste weeks optimizing the model; invest them in the system.

---

## 3. Complete Tech Stack

An industry-standard stack optimized for an individual project (without the overhead of enterprise platforms like SageMaker/Vertex, which can be counterproductive for a personal project).

| Layer | Tool | Why this one and not another |
|------|-------------|------------------------|
| Language | Python 3.11+ | The absolute standard in ML |
| Code versioning | Git + GitHub | Standard, but requires clean practices (see Git section) |
| Data versioning | **DVC** | Versions large datasets without committing them to Git |
| Experiment tracking | **MLflow** | Industry standard, free, and self-hosted |
| Model Registry | **MLflow** | Included out of the box, manages model versions |
| Data validation | **Pandera** or Great Expectations | Data quality schemas and tests |
| Inference API | **FastAPI** | Fast, typed, automatic documentation |
| Containers | **Docker** + Docker Compose | Full reproducibility |
| CI/CD | **GitHub Actions** | Free, integrated, and highly capable |
| Orchestration | **Prefect** or **ZenML** | Simpler than Airflow to get started |
| Monitoring / drift | **Evidently AI** | Open-source, does not require dedicated infrastructure |
| System metrics | Prometheus + Grafana (optional) | For the stretch goal |
| Cloud deployment | **Render / Railway / Fly.io / Modal** | Free tiers, no Kubernetes |
| Testing | **pytest** | Standard |
| Code quality | **ruff** + **black** + **pre-commit** | Automatic linting and formatting |

> **About Kubernetes:** You do NOT need it for this project. Mentioning Modal/Render/Fly.io demonstrates that you know how to choose the right tool for the scale of the problem, which is exactly what a good engineer values. Save Kubernetes for a robot project or as a documented stretch goal.

---

## 4. Repository Structure

A clean structure communicates professionalism before anyone even reads a line of code. Use this as a base (inspired by `cookiecutter-data-science`):

```
mlops-fraud-pipeline/
├── .github/
│   └── workflows/
│       ├── ci.yml                # tests + linting on every push
│       └── cd.yml                # image build + deploy
├── data/
│   ├── raw/                      # raw data (managed by DVC, not Git)
│   ├── processed/                # processed data (DVC)
│   └── .gitignore
├── src/
│   ├── data/
│   │   ├── ingest.py             # data loading
│   │   ├── validate.py           # validation with Pandera
│   │   └── preprocess.py         # feature engineering
│   ├── models/
│   │   ├── train.py              # training + MLflow tracking
│   │   ├── evaluate.py           # metrics and validation
│   │   └── register.py           # register in Model Registry
│   ├── api/
│   │   ├── main.py               # FastAPI app
│   │   ├── schemas.py            # Pydantic models
│   │   └── predict.py            # inference logic
│   ├── monitoring/
│   │   ├── drift.py              # Evidently reports
│   │   └── dashboard.py          # monitoring dashboard
│   └── config.py                 # centralized configuration
├── pipelines/
│   ├── training_pipeline.py      # Prefect/ZenML training workflow
│   └── monitoring_pipeline.py    # scheduled monitoring workflow
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── notebooks/
│   └── 01_exploration.ipynb      # ONLY initial exploration, not production
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── dvc.yaml                      # DVC pipeline definition
├── params.yaml                   # versioned hyperparameters
├── pyproject.toml                # dependencies (use uv or poetry)
├── .pre-commit-config.yaml
├── Makefile                      # common commands (make train, make test...)
└── README.md                     # ⭐ the most important document in the project
```

---

## 5. Phase-by-Phase Roadmap

Each phase has an **objective**, **concrete tasks**, a **deliverable**, and a **"definition of done"** (criterio de hecho). Do not move on to the next phase without meeting the criteria.

---

### 🔹 Phase 0 — Setup and Definition (Week 1)

**Objective:** Have the environment and the problem clearly defined before touching ML.

**Tasks:**
- [ ] Create a GitHub repository with the folder structure above (empty but complete).
- [ ] Configure the environment with `uv` (recommended, much faster than pip/poetry) or `poetry`.
- [ ] Install and configure `pre-commit` with `ruff` + `black`.
- [ ] Write an initial README with: problem, objective, planned architecture (diagram), and stack.
- [ ] Download the dataset and perform an initial exploration in a notebook (distributions, nulls, class balance).
- [ ] Document decisions: which business metric matters to you? For fraud, likely **high recall** (you don't want to miss fraud) while controlling the false positive rate.

**Deliverable:** Structured repository + exploration notebook + README with architecture.  
**Definition of Done:** A clean `git clone` + installation works, and the README clearly explains what you are going to build.

---

### 🔹 Phase 1 — Data Pipeline and Versioning (Weeks 2–3)

**Objective:** Convert raw data into features in a reproducible and versioned manner.

**Tasks:**
- [ ] Initialize DVC in the repository (`dvc init`) and configure a remote (can be free Google Drive, S3, or local).
- [ ] Move raw data to `data/raw/` and track it with DVC (`dvc add`).
- [ ] Write `src/data/validate.py` using **Pandera**: define a schema that validates types, ranges, and nulls. This must **fail loudly** if the data does not meet the contract.
- [ ] Write `src/data/preprocess.py` with your feature engineering (encoding, scaling, derived features).
- [ ] Define the pipeline in `dvc.yaml`: each stage (ingest → validate → preprocess) with its dependencies and outputs.
- [ ] Run `dvc repro` and verify that it reconstructs the entire data pipeline.
- [ ] Version `params.yaml` with the preprocessing parameters.

**Deliverable:** Executable DVC pipeline going from raw data to ready features, completely versioned.  
**Definition of Done:** `dvc repro` reconstructs the processed dataset and `dvc.lock` captures the hashes. If you change `params.yaml`, DVC detects what to re-run.

> **Key concept demonstrated here:** data reproducibility. Anyone can regenerate your exact dataset starting from raw data.

---

### 🔹 Phase 2 — Training and Experiment Tracking (Weeks 3–4)

**Objective:** Train models while rigorously logging every experiment.

**Tasks:**
- [ ] Spin up a local MLflow server (`mlflow server`) or use file-based tracking to start.
- [ ] Write `src/models/train.py` which:
  - Loads the versioned features.
  - Trains a model (start simple: LogisticRegression or XGBoost).
  - Logs to MLflow: parameters, metrics (precision, recall, F1, AUC-PR), the model, and artifacts (confusion matrix, PR curve).
  - Handles class imbalance (crucial in fraud): `class_weight`, SMOTE, or threshold tuning.
- [ ] Run several experiments changing hyperparameters and compare them in the MLflow UI.
- [ ] Write `src/models/evaluate.py` with rigorous evaluation: cross-validation, and analysis of the optimal threshold according to your business metric.

**Deliverable:** Multiple comparable experiments in MLflow, with the best model identified.  
**Definition of Done:** You open the MLflow UI and see a history of comparable experiments side-by-side, complete with metrics and artifacts.

> **Key concept:** traceability. You know exactly which data + which code + which parameters produced each model and its outcome.

---

### 🔹 Phase 3 — Model Registry and Packaging (Week 5)

**Objective:** Manage model versions as production-ready artifacts.

**Tasks:**
- [ ] Use the **MLflow Model Registry**: register your best model and learn the lifecycle stages (`Staging`, `Production`, `Archived`).
- [ ] Write `src/models/register.py` which automatically promotes a model to `Production` if it passes a metric threshold.
- [ ] Package the model with its preprocessing (scikit-learn pipeline or custom inference function) so that the API can receive raw data and return predictions without manual steps.
- [ ] Version the decision threshold alongside the model (it is part of the artifact).

**Deliverable:** A registered and versioned model, retrievable by name/version/stage.  
**Definition of Done:** You can load the "Production model" with a single line of code, without needing to know the specific version number.

---

### 🔹 Phase 4 — Inference API (Weeks 5–6)

**Objective:** Serve the model as a professional REST API.

**Tasks:**
- [ ] Write `src/api/schemas.py` with **Pydantic** models that validate the input (a transaction) and type the output (prediction + probability).
- [ ] Write `src/api/main.py` using FastAPI:
  - Endpoint `POST /predict` that receives a transaction and returns a prediction + score.
  - Endpoint `GET /health` for health checks.
  - Endpoint `GET /model-info` that returns the active model's version.
  - Model loading from the Registry on startup.
- [ ] Add logging for each prediction (input + output + timestamp) → this will be the foundation for monitoring.
- [ ] Write API tests using `pytest` + `TestClient`.
- [ ] Leverage FastAPI's automatic documentation (`/docs` with Swagger).

**Deliverable:** A functional, documented, and tested API serving predictions.  
**Definition of Done:** You spin up the API, go to `/docs`, send a test transaction, and receive a prediction with its probability.

---

### 🔹 Phase 5 — Containerization (Week 6)

**Objective:** Package everything so that it runs identically on any machine.

**Tasks:**
- [ ] Write an optimized multi-stage `Dockerfile` (build + runtime): lightweight image (`python:3.11-slim`), properly cached layers, and a non-root user.
- [ ] Write a `docker-compose.yml` that spins up the API + MLflow + (optional) the logging database together.
- [ ] Verify that the image is reasonably small (use `.dockerignore`).
- [ ] Test the entire cycle inside containers.

**Deliverable:** `docker-compose up` spins up the entire system.  
**Definition of Done:** Someone without Python installed can run your project using only Docker.

---

### 🔹 Phase 6 — Automated CI/CD (Week 7)

**Objective:** Automate testing, building, and deployment on every change.

**Tasks:**
- [ ] `ci.yml` (GitHub Actions): run linting (ruff), formatting (black), and tests (pytest) on every push/PR. Prevent merging if any check fails.
- [ ] Add a test that validates the model meets a minimum metric threshold before deployment (model validation gate).
- [ ] `cd.yml`: on push to `main`, build the Docker image and publish it (e.g., to GitHub Container Registry) and/or deploy it to the cloud.
- [ ] Add badges to the README (build passing, coverage).

**Deliverable:** A green CI/CD pipeline that protects the main branch and deploys automatically.  
**Definition of Done:** You push code, see the checks running on GitHub, and if everything passes, it deploys automatically.

> **Key concept:** This is what separates a student project from a professional one. CI/CD for ML is exactly what companies want to see.

---

### 🔹 Phase 7 — Orchestration (Week 8)

**Objective:** Convert your scripts into orchestrated and scheduled workflows.

**Tasks:**
- [ ] Choose **Prefect** (simpler) or **ZenML** (more ML-focused, stack-agnostic).
- [ ] Write `pipelines/training_pipeline.py`: a workflow that chains ingest → validate → preprocess → train → evaluate → register, featuring retries and logging.
- [ ] Write a scheduled (e.g., daily) `pipelines/monitoring_pipeline.py` that evaluates drift on recent data.
- [ ] Configure the training pipeline to be triggered manually or by an event (e.g., when drift is detected).

**Deliverable:** Orchestrated workflows visible in the Prefect/ZenML dashboard.  
**Definition of Done:** You trigger the training pipeline with a single command and watch each stage execute, with automatic retries if anything fails.

---

### 🔹 Phase 8 — Monitoring and Drift (Weeks 9–10)

**Objective:** Detect when the model degrades in production. **This is a key highlight of the project.**

**Tasks:**
- [ ] Write `src/monitoring/drift.py` using **Evidently**:
  - **Data drift**: Has the input feature distribution changed compared to the training dataset?
  - **Concept drift / target drift**: Has the relationship between features and the target changed?
  - **Prediction quality**: If you have delayed ground truth labels, compare predictions against reality.
- [ ] Generate Evidently reports (interactive HTML) and a monitoring dashboard.
- [ ] Define **threshold alerts**: If drift exceeds X, trigger an alert (log, email, or webhook).
- [ ] Close the loop: connect the drift alert to trigger the retraining pipeline (Phase 7).
- [ ] Simulate drift by injecting modified data to **demonstrate** that your system detects it (this is excellent for the demo).

**Deliverable:** Monitoring dashboard + alerts + closed retraining loop.  
**Definition of Done:** You inject drifted data, the system detects it, the alert fires, and retraining is triggered. **Showing this on video will stand out significantly to recruiters.**

---

### 🔹 Phase 9 — Deployment, Documentation, and Presentation (Weeks 11–12)

**Objective:** Put the system online and package it professionally.

**Tasks:**
- [ ] Deploy the API to a free service: **Render**, **Railway**, **Fly.io**, or **Modal** (serverless, ideal for ML without Kubernetes).
- [ ] Ensure the public URL works and the `/docs` documentation is accessible.
- [ ] Write the **final README** (see checklist below).
- [ ] Record a 2-3 minute video showing the entire system in action, focusing on the drift-to-retraining loop.
- [ ] Write a LinkedIn post explaining what you built, what you learned, and the decisions you made (not just "I made a project," but the *reasoning* behind it).
- [ ] Create a clean architecture diagram (using Excalidraw or draw.io) for the README.

**Deliverable:** Online system + excellent README + video + post.  
**Definition of Done:** A recruiter can understand the entire project in 2 minutes by reading the README and watching the video, without having to clone anything.

---

## 6. The Complete README Checklist

The README has a substantial impact on your portfolio. Most people will never clone your repository; they will judge it based on the README.

- [ ] Catchy title + one-line description.
- [ ] Visual architecture diagram.
- [ ] Short GIF or video of the system running.
- [ ] "Why This Project" section (the business problem).
- [ ] Tech stack with badges.
- [ ] Setup instructions that actually work (`make setup`, `docker-compose up`).
- [ ] **Design decisions** section: why you chose each tool. This demonstrates engineering judgment, not just execution.
- [ ] Results: model metrics + dashboard screenshots.
- [ ] Link to the live demo.
- [ ] What you learned / what you would do differently (showing maturity).

---

## 7. Common Mistakes to Avoid

| Mistake | Why it is bad | What to do |
|---------|---------------|------------|
| Obsessing over the F1-score | The model is not the main point of the project | Use a "good enough" model, focus on excellent infrastructure |
| Committing data to Git | Heavy repositories, bad practice | Always use DVC for data |
| Notebooks as production code | Not reproducible, not testable | Use notebooks for exploration only; production code goes in `src/` |
| Poor README | Nobody will understand your work | Invest proper time in the README |
| Not closing the drift loop | You miss out on the most impactful part | Connect monitoring → retraining |
| Skipping tests | Looks amateur/student-level | Use pytest from the start |
| Hardcoding configuration | Fragile and inflexible | Use `config.py` + `params.yaml` |
| Trying to use Kubernetes "to impress" | Unnecessary overhead, bad signal | Modal/Render demonstrate better judgment for this scale |

---

## 8. Stretch Goals (To Go Further)

Once you have the core running, these extras can take it to the next level:

- **Model A/B testing** (champion/challenger): serve two versions and compare real-world performance.
- **Feature Store** (Feast): centralize features for both training and inference.
- **Prometheus + Grafana**: system metrics (latency, throughput, errors) in addition to drift.
- **Shadow deployment**: the new model predicts in parallel without affecting production, to validate it with real traffic.
- **Infrastructure as Code** (Terraform): provision your cloud resources declaratively.
- **Explainability** (SHAP in the API): each prediction comes with its explanation, linking with your other projects.

---

## 9. Learning Resources

**Courses / Guides:**
- *Made With ML* (Goku Mohandas) — the free reference for end-to-end MLOps.
- *MLOps Zoomcamp* (DataTalksClub) — a highly practical, free course on GitHub.
- Official documentation for DVC, MLflow, Evidently, and Prefect (all highly recommended).

**Datasets (Fraud Problem):**
- Credit Card Fraud Detection (Kaggle) — the classic, highly imbalanced dataset.
- IEEE-CIS Fraud Detection (Kaggle) — richer and more realistic.
- Synthetic Financial Datasets (PaySim, Kaggle) — great for simulating drift.

**Concepts to Master Along the Way:**
- MLOps maturity levels (0: manual → 1: training automation → 2: complete CI/CD). Your project should aim to reach Level 2.
- Difference between data drift and concept drift.
- Precision/recall trade-offs and how to choose the threshold based on business needs.

---

## 10. Milestone Summary

| Week | Milestone | Status |
|--------|------|--------|
| 1 | Repo + environment + defined problem | ⬜ |
| 2–3 | Versioned data pipeline (DVC) | ⬜ |
| 3–4 | Tracked experiments (MLflow) | ⬜ |
| 5 | Model Registry + packaging | ⬜ |
| 5–6 | Inference API (FastAPI) | ⬜ |
| 6 | Containerization (Docker) | ⬜ |
| 7 | CI/CD (GitHub Actions) | ⬜ |
| 8 | Orchestration (Prefect/ZenML) | ⬜ |
| 9–10 | Monitoring + drift + closed loop | ⬜ |
| 11–12 | Deployment + README + video + post | ⬜ |

---

**The final goal:** when a recruiter or senior engineer looks at this project, they should think, *"this person truly understands how ML works in production."* It is not just a notebook with a model; it is a living system that trains, deploys, monitors, and retrains itself. That is exactly what is rare to find in junior profiles.
