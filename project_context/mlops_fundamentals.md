# End-to-End MLOps Pipeline — Fundamentals and Design

> This document is the conceptual work that comes **before** writing a single line of code. It covers what you are going to build and why, the problem you are going to solve, the technology stack justified tool by tool, and the repository architecture. Having this clear is what separates an improvised project from one that is designed.

---

## Introduction: what this project actually is

When someone learns Machine Learning, they almost always do it inside a notebook: you load a CSV, train a model, measure accuracy, and celebrate. That is fine for learning the algorithms. The problem is that this environment looks nothing like how ML works in a real company. In production, a model is only a small piece of a much larger system, and the vast majority of the effort, risk, and value lies in everything surrounding the model: where the data comes from, how its correctness is guaranteed, how everything is versioned to reproduce results, how the model is deployed for others to use, how its performance is monitored over time, and what happens when it begins to degrade.

This discipline is called **MLOps**: the combination of Machine Learning, data engineering, and DevOps practices to bring models to production in a reliable, reproducible, and maintainable way. It is, in essence, the engineering that turns an experiment into a product.

This project takes you through that entire path. You are not going to build the best fraud detection model in the world; you are going to build the **system** that trains that model, deploys it as a service that anyone can call, monitors it while it is active, and automatically retrains it when it begins to degrade. The difference is huge, and it is exactly the kind of difference a senior engineer detects in thirty seconds when looking at your portfolio.

### MLOps maturity levels

To frame the ambition of the project, it is useful to understand a framework used in the industry regarding the three levels of maturity:

*   **Level 0 — Manual process.** Everything is done by hand in notebooks. A model is trained, exported, and someone uploads it to a server. There is no automation, no serious versioning, and no monitoring. This is where most people who "know ML" reside.
*   **Level 1 — Training automation.** The training pipeline is automated and can be retrained reproducibly. There is data and experiment versioning, along with basic monitoring.
*   **Level 2 — Full CI/CD.** The entire system is automated end-to-end: any change in code or data triggers automated tests, validation, build, and deployment. Production monitoring closes the loop by triggering retraining when necessary.

The goal of this project is to **reach Level 2**. This is what makes it stand out for a junior profile: almost no one at this level has a project demonstrating Level 2 maturity, because it requires integrating many pieces and understanding how they fit together. It is not difficult due to the complexity of each individual tool, but because of the coordination of the overall system. Precisely that ability to coordinate a system is what companies cannot easily teach and why they pay a premium.

### Who it impresses and why

It is worth being explicit about the audience for this project, as it conditions many design decisions. Your portfolio will be viewed, in order of importance, by engineers conducting technical interviews, technical recruiters filtering candidates, and hiring managers making final hiring decisions. What each values is different:

The **engineer** interviewing you will not be impressed by your F1-score; they will be impressed by architectural decisions, seeing that you understand trade-offs, and a README explaining the *why* behind every choice. They want to see judgment, not just execution.

The **technical recruiter** often does not dive into the code. They judge by surface signals: Is the CI/CD pipeline green? Is it deployed live? Is the README professional? Is there a video? This is why the README and the demo matter as much as the code.

The **hiring manager** thinks in terms of risk: Does this person know how not to break production? Do they understand that the work does not end when the model trains well? A project that includes monitoring and automated retraining answers that question before they even have to ask it.

With this in mind, the entire project is designed to answer a single implicit question: *does this person truly understand how ML works in production?* Every phase, tool, and decision in this document exists to make the answer a resounding yes.

---

## 1. What you are going to build

The system you are going to build takes the form of a closed loop. It is helpful to view it at a glance first, and then walk through it slowly.

```
          Raw data
             │
             ▼
  ┌─────────────────────────┐
  │ Ingestion + validation  │ ──►  versioning with DVC
  └─────────────────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │ Training                │ ──►  experiments tracked with MLflow
  └─────────────────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │ Model Registry          │ ──►  Model Registry (MLflow)
  └─────────────────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │ Inference API           │ ──►  FastAPI + Docker
  └─────────────────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │ Deployment              │ ──►  Cloud (Render / Railway / Fly.io / Modal)
  └─────────────────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │ Monitoring              │ ──►  Evidently (data & concept drift)
  └─────────────────────────┘
             │
             ▼
  ┌─────────────────────────┐
  │ Auto. retraining        │ ──►  orchestrated with Prefect / ZenML
  └─────────────────────────┘
             │
             └──────────► returns to start (closed loop)
```

The best way to understand this system is to look at it from two different perspectives that coexist within it: the **path of a prediction** (what happens when a new transaction arrives, in milliseconds) and the **model lifecycle** (what happens to the system over days and weeks). These are two clocks ticking at very different speeds over the same infrastructure, and understanding both is understanding MLOps.

### The path of a prediction

Imagine the system is already deployed and running. A request arrives: a transaction that needs to be classified as fraud or non-fraud. This is what happens, step by step:

The transaction enters through the **inference API**. The first thing that happens is not the prediction, but **input validation**: the system checks that the transaction has all expected fields, with the correct types, and within reasonable ranges. If something does not align, it is rejected with a clear error before reaching the model. This detail is important because in production, malformed data is the number one cause of silent failures: a model receiving garbage will not complain; it will simply return meaningless predictions.

Once validated, the transaction goes through **preprocessing**, exactly the same as used during training. Here lies a critical MLOps subtlety: the preprocessing for inference and training must be identical. If they differ even slightly, the model receives data in production with a different shape than what it learned, causing its performance to plummet without anything appearing broken. This problem is called *training-serving skew*, and the way to avoid it is to package the preprocessing alongside the model as a single unit.

With the features already calculated, the **model** produces a prediction: not just a label (fraud / non-fraud), but a **probability**. This probability is compared against a **decision threshold** that you have deliberately chosen based on business logic, which is versioned alongside the model because it is part of the artifact, not an implementation detail.

Before returning the response, the system **logs the prediction**: it stores the input transaction, the prediction, the probability, and the timestamp. This might seem like a minor detail, but it is the raw material for monitoring: without a log of what the model has been predicting, it is impossible to detect later if it is degrading.

Finally, the API returns the response. All of this happens in milliseconds, for every request, invisibly to the caller.

### The model lifecycle

The second clock runs much slower and is where the most sophisticated part of the project lives. While the API handles predictions, in the background, the system monitors its own health over time.

The prediction logs that the API accumulates are periodically analyzed for **drift**—the phenomenon where reality changes and the model, having learned from a past reality, ceases to be valid. There are two main ways this happens. The first is **data drift**: the distribution of incoming transactions changes relative to what the model saw in training; for example, a new payment method appears, or the typical range of transaction amounts shifts. The second, more insidious type is **concept drift**: the relationship between the features and the target changes. In fraud, this is constant because fraudsters adapt their techniques precisely to evade the models detecting them.

When the monitoring system detects that drift exceeds a certain threshold, an **alert** is triggered. And here is the piece that closes the loop and is most impressive: that alert does not just sit in an inbox to be ignored; it **automatically triggers the retraining pipeline**. The system returns to the beginning: it ingests the most recent data, validates it, retrains the model on the new reality, evaluates whether the new model is better than the current one, and, if so, promotes it to production in a controlled manner. The old model is archived, the new one enters service, and the cycle continues.

That is a **living ML system**: not a model that is trained once and forgotten, but an organism that maintains itself. Demonstrating that you understand and know how to build this closed loop is probably what will differentiate you most from other junior candidates, because it is precisely the part that courses and tutorials almost never cover and what companies struggle with most on a daily basis.

---

## 2. The problem you will solve: fraud detection

The choice of the problem is not an aesthetic detail. A good problem ensures that every piece of the infrastructure has an obvious reason to exist, allowing you to speak naturally in an interview about real decisions. Transaction fraud detection is an excellent choice for your specific profile, and it is worth understanding exactly why.

### Why fraud is the ideal problem for this project

**It connects with your interest in cybersecurity.** Fraud is, at its core, applied security: there is an intelligent adversary trying to evade your system, and your system has to adapt. This allows your portfolio to tell a coherent story instead of being a collection of disjointed projects. When an interviewer sees fraud alongside your projects on honeypots or intrusion detection, they will understand there is a consistent line of thought behind it.

**Drift is not theoretical; it is the essence of the problem.** In many ML problems, monitoring for drift feels like an artificial addition thrown in just to show you can do it. In fraud, it is the opposite: drift is the central reality. Fraudsters constantly change their techniques to evade the models that detect them. This means your monitoring and retraining system, the crown jewel of the project, has an impeccable business justification. You are not monitoring drift "to practice"; you are monitoring it because without it, the model would become obsolete in weeks.

**It is a problem that any company understands.** You do not need to explain why detecting fraud matters. The cost is obvious and quantifiable. This allows you to speak the language of business, not just metrics, in an interview—a skill that distinguishes good ML engineers.

### What makes fraud technically interesting

There are four characteristics of fraud that make it a rich problem, and understanding them deeply will give you plenty to discuss:

**Extreme class imbalance.** Fraud is rare: in a realistic dataset, fraudulent transactions might account for 0.1% or less of the total. This breaks naive metrics: a model that simply predicts "no fraud" would have 99.9% accuracy and be completely useless. Dealing with this forces you to understand techniques for handling imbalance (class weighting, resampling, threshold adjustment) and to use appropriate metrics, as discussed below.

**Cost asymmetry.** Making a mistake in one direction does not cost the same as in the other. A **false negative** (allowing a fraudulent transaction to slip through) means direct financial loss. A **false positive** (blocking a legitimate transaction) means an angry and potentially lost customer. These costs are rarely equal, and deciding where to set the model's threshold is, in reality, a business decision regarding which error you prefer to make. Being able to reason about this demonstrates a maturity that goes beyond the technical aspect.

**Label delay.** This is subtle and highly realistic. In fraud, you often do not know if a transaction was fraudulent at the moment it occurs; you discover it days or weeks later when a claim or chargeback arrives. This complicates real-time performance monitoring (since ground truth labels arrive late) and makes input drift monitoring even more valuable, as it provides an early signal before the labels arrive. Mentioning this nuance in an interview shows you have truly thought about the problem.

**The adversarial and shifting nature.** Because there is an adapting adversary, the problem is never "solved." This naturally justifies the entire continuous retraining setup and connects directly with a cybersecurity mindset: defense against an evolving threat.

### The metrics that actually matter

For all these reasons, the metrics you use must be the correct ones, and choosing them well is in itself a sign of competence:

**Accuracy is useless** here due to the imbalance, as we have seen. Forget it as a primary metric.

**Recall** (the fraction of actual fraud you manage to detect) is usually the most important metric in fraud, because letting fraud pass is expensive. But high recall at any cost is useless, because you could simply flag everything as fraud.

**Precision** (of what you flag as fraud, what fraction is actually fraud) controls false positives, meaning how many legitimate customers you inconvenience.

The balance between the two is managed using the **decision threshold**, and the visual tool to reason about this trade-off is the **precision-recall curve** and its area under the curve (**PR-AUC**). It is important to use PR-AUC rather than the more common ROC-AUC, because in highly imbalanced problems, the ROC curve gives a deceptively optimistic impression; the precision-recall curve reflects actual performance much better when the positive class is rare. Knowing this and being able to explain it is exactly the type of detail that distinguishes someone who understands the problem from someone who just applied a recipe.

Ultimately, the metric that matters most is the **business cost**: if you can estimate the cost of a false positive and a false negative, you can select the threshold that minimizes the total expected cost. Framing the evaluation in these terms, even with estimated costs, elevates the project from a technical exercise to a business solution.

### Alternatives, in case you are more motivated by something else

Fraud is the recommendation, but the infrastructure you will build is almost identical regardless of the problem. If another domain motivates you more, any of these will work just as well, provided it has the key property of natural and demonstrable drift:

| Problem | Suggested Dataset | Why it works | Type of natural drift |
|:---|:---|:---|:---|
| Customer churn prediction | Telco Customer Churn (Kaggle) | Classic and highly understandable business problem | Customer behavior shifts over time |
| Demand / sales prediction | Store Sales (Kaggle) | Time series, highly in demand in retail | Seasonality and trend shifts |
| Sentiment classification | Amazon Reviews (Kaggle) | Shows you command NLP | Vocabulary and context evolve |

The only things that would change among these options are the specific model and features; the entire MLOps framework, which is the real point of the project, remains identical. That is why the recommendation is to choose the problem that appeals to you most and move forward: the value lies in the infrastructure, not the domain.

---

## 3. The technology stack, justified tool by tool

This is the section where it is most valuable to pause, because choosing the right tools and, above all, **knowing how to explain why you chose them**, is one of the clearest demonstrations of engineering judgment. Anyone can follow a tutorial and use the tools they are told to; a good engineer understands what problem each tool solves and why they prefer it over alternatives.

Before diving into each tool, it is useful to establish the **philosophy of choice** that runs through the entire stack, as it is a mark of maturity in itself. The beginner's temptation is to use the most powerful and heavily marketed tools to impress: Kubernetes, giant cloud platforms, complex orchestrators. This is almost always a mistake, and a senior engineer views it as a sign of immaturity. The right tool is not the most powerful, but the **appropriate one for the scale of the problem**. For an individual project, that means lightweight, free, and low-maintenance tools that let you focus on the system instead of fighting the infrastructure. This is also supported by industry insights: for small teams, it is explicitly recommended to avoid enterprise platforms like SageMaker or Vertex AI, whose configuration and maintenance costs exceed their benefits until you scale. Choosing lightweight is not a limitation; it is the correct decision, and knowing how to argue this puts you ahead of someone who integrated Kubernetes without needing it.

Let's look at the stack layer by layer.

### Language and Environment Management

**Python 3.11 or higher** is the absolute standard in Machine Learning; there is no debate here. The modern version matters for performance and compatibility with current libraries.

To manage dependencies and the virtual environment, the recommendation is **uv**, a tool written in Rust that is orders of magnitude faster than traditional alternatives like pip or even poetry. Using uv instead of pip is a small but revealing detail: it shows you follow the state of the art in tooling rather than sticking to what you learned years ago. If you prefer something more established, **poetry** remains a perfectly valid and widely used option. The important thing, in any case, is that the environment is **reproducible**: that the exact versions of every dependency are pinned, so the project runs identically on your machine and anyone else's.

### Versioning: Code and Data Separately

**Git and GitHub** for code is obvious, but do it right: clear commit messages, feature branches, and a history that tells a story. The repository itself is part of your portfolio, and a clean Git history communicates professionalism.

The interesting problem is **data versioning**, and here is one of the first true MLOps decisions. Git is designed for text, not large data files; committing a dataset of hundreds of megabytes to Git makes the repository bloated, slow, and unmanageable. The solution is **DVC (Data Version Control)**. DVC works alongside Git elegantly: instead of saving the dataset in the repository, it saves a small metadata file pointing to the dataset, which lives in separate storage (it can be free Google Drive, S3, or even a local folder). Git versions the pointer; DVC manages the actual data. This allows you to run `git checkout` on an old version of the code, and `dvc checkout` will bring you the exact version of the data corresponding to that moment. This gives you **data reproducibility**: the ability to regenerate exactly any past result, which is a pillar of MLOps.

DVC also has a second capability you will use: it allows you to define declarative **data pipelines** in a `dvc.yaml` file, where each stage (ingestion, validation, preprocessing) declares its dependencies and outputs. When you run `dvc repro`, DVC checks what has changed and only re-executes what is necessary, just like a smart build system. This turns your preprocessing from a script you run by hand into a reproducible, cached pipeline.

### Data Validation: The Contract with Reality

A central principle of MLOps is that **bad data is the number one cause of failures in production**, and these failures are silent: a model receiving data with the wrong shape does not throw an error; it just predicts poorly. That is why you need a layer that validates the data and **fails loudly** when it does not meet expectations.

For this, you will use **Pandera** (or, as a more complete alternative, **Great Expectations**). Pandera lets you define a *schema*: a contract specifying what columns your dataset must have, of what type, in what ranges, and with what constraints (no nulls where they shouldn't be, values within a permitted set, etc.). If the data does not comply with the contract, validation fails with a clear error before those data contaminate training or reach the model in production. Having explicit data validation is a clear mark of maturity; it is something people coming solely from notebooks almost never do, and any engineer with production experience values it immediately.

### Experiment Tracking and Model Registry: MLflow

When you train models, you quickly find yourself in chaos: What hyperparameters did I use in that experiment that got a good result? With which version of the data? What exact metrics did it achieve? Keeping this in your head or in a notepad does not scale. The industry-standard solution is **MLflow**, and it will be one of the core tools of the project.

MLflow solves two different problems that should not be confused. The first is **experiment tracking**: every time you train, MLflow automatically records the parameters used, the metrics obtained, the model itself, and any artifacts you want (plots, confusion matrices, curves). You then have a web interface where you can compare all your experiments side-by-side and see which one won and why. This gives you **traceability**: you know exactly what combination of data, code, and parameters produced each result.

The second problem MLflow solves is the **model registry** (Model Registry). Once you have a good model, you need to manage it as a production artifact: versioning it, marking which version is in testing (`Staging`) and which is in production (`Production`), and being able to retrieve "the model currently in production" without needing to know its exact version. MLflow's Model Registry does exactly this, allowing you to build workflows where a model is automatically promoted to production only if it exceeds a certain metric threshold. This separation between "experimenting" and "managing models as versioned artifacts" is a profound MLOps concept, and MLflow embodies it perfectly. It is free, self-hostable, and the de facto standard, making it exactly what you want to show you know how to use.

### The Inference API: FastAPI and Pydantic

A model that only works in your notebook is of no use to anyone. For it to be a product, it has to be a **service** that other systems can call. That means a REST API, and the modern tool to build it in Python is **FastAPI**.

FastAPI has several virtues that make it the correct choice over older alternatives like Flask. It is incredibly **fast** (hence the name), built on modern Python typing, and generates **interactive documentation automatically**: you spin up the API and get a web page (Swagger UI) where anyone can view endpoints and test them without writing code. For a portfolio, this is gold, because it allows a recruiter to interact with your model right from the browser.

FastAPI relies on **Pydantic** for validation. With Pydantic, you define—using typed Python classes—exactly what shape an incoming transaction must have and what shape the response will take. If a request arrives that does not fit, FastAPI automatically rejects it with a clear error message before it even reaches your logic. This reinforces the principle of validating at the boundary: you never let malformed data enter the system. The API will have, at minimum, a predict endpoint, a health check endpoint (so the deployment system knows it is alive), and an endpoint that reports which model version is active. Crucially, every prediction will be logged, as this log is the raw material for subsequent monitoring.

### Containerization: Docker

There is a classic software problem summarized by the phrase "it works on my machine." A project that depends on the exact configuration of your computer is fragile and impossible to share. The universal solution is **Docker**, which packages your application alongside its entire environment (Python, libraries, configuration) into an *image* that runs identically on any machine with Docker.

For this project, you will write a `Dockerfile` that builds your API image, ideally in a *multi-stage* format (one stage to build and another, lighter stage to execute) to keep the final image small. You will use **Docker Compose** to spin up the entire system with a single command: the API, the MLflow server, and any other parts, all coordinated. The result is that someone who does not even have Python installed can run your entire project with a simple `docker compose up`. This is **total reproducibility**, and it is an expected skill for any engineer today.

### CI/CD: GitHub Actions

This is where the project transitions from "student" to "professional." **CI/CD** stands for Continuous Integration and Continuous Deployment: the automation of tasks that guarantee quality and deliver software. Instead of running tests and deploying manually, you define pipelines that do it automatically on every change.

You will use **GitHub Actions**, which is integrated into GitHub, free for projects like this, and more than sufficient. You will configure two workflows. The **integration (CI)** workflow runs on every push: it runs linting (ensuring code follows conventions), formatting, and tests, and blocks changes from being merged if anything fails. You will also add a key MLOps concept: a *model validation gate*—a test that checks whether the model meets a minimum performance metric before allowing it to be deployed, ensuring an inferior model never reaches production. The **deployment (CD)** workflow builds the Docker image and automatically publishes or deploys it when a change lands on the main branch. CI/CD applied to Machine Learning is exactly what companies want to see and rarely find in junior profiles; it is perhaps the piece that carries the most weight when it comes to impressing.

### Orchestration: Prefect or ZenML

Your training and monitoring scripts are fine as standalone pieces, but a production system needs to **orchestrate** them: chaining them into flows that run in order, with automatic retries if something fails, scheduling, and visibility into what is happening. The traditional tool for this is Apache Airflow, but it is too heavy for starting out; currently, something simpler is recommended for small teams.

You have two excellent options. **Prefect** is a modern and straightforward orchestrator with a gentle learning curve and a generous free tier; you turn a Python function into a flow step with a decorator and get retries, logging, and a dashboard effortlessly. **ZenML** is an alternative specifically oriented towards ML, stack-agnostic (integrating with any cloud and other tools in the stack), and designed precisely for Machine Learning pipelines. Either is a fine choice: Prefect if you prioritize general simplicity, ZenML if you want something centered on ML workflows. With it, you will define the training pipeline as a flow (ingestion → validation → preprocessing → training → evaluation → registration) and the monitoring pipeline as a scheduled flow that checks for drift periodically and, when necessary, triggers retraining.

### Monitoring and Drift Detection: Evidently

This is the tool that breathes life into the crown jewel of the project. **Evidently AI** is an open-source library specialized in monitoring models in production, and its great virtue for an individual project is that **it does not require dedicated infrastructure**: you integrate it, and it generates reports without needing a complex setup.

With Evidently, you will detect the two types of drift we saw earlier: **data drift** (has the distribution of incoming transactions changed relative to the training data?) and **concept/target drift** (has the relationship between features and the target changed?). If you have actual labels, even with a delay, you can also compare predictions against ground truth to measure actual performance. Evidently generates highly visual, interactive HTML reports that look great in your README and video, and you will define **alerts with thresholds** on top of these metrics to trigger retraining. A neat trick for the demo: you can **deliberately inject drifted data** to prompt your system to detect and react in real-time. Capturing that moment (anomalous data enters, the alert fires, retraining is triggered) is likely what will have the most impact on anyone viewing the project.

### Cloud Deployment: Lightweight, Without Kubernetes

To put your API online publicly and for free, you will use one of several services designed for this: **Render**, **Railway**, **Fly.io**, or **Modal**. All have free tiers and, crucially, **none of them require Kubernetes**. Modal is especially interesting for ML because it is serverless and oriented toward inference, practically eliminating infrastructure management.

It is worth emphasizing this because it is a design decision that communicates mature judgment: **you do not need Kubernetes for this project, and using it "to impress" would be counterproductive**. A senior engineer interprets the unnecessary use of Kubernetes as a sign of immaturity—someone who does not know how to scale the tool to the problem. Choosing a lightweight deployment and being able to explain why (unnecessary overhead for the scale of an individual project) demonstrates the exact opposite: that you understand trade-offs. Kubernetes is an excellent tool for its scale, and it will have its place in other complex setups or as a documented stretch goal, but not here.

### Code Quality: pytest, ruff, black, and pre-commit

Finally, the tools that keep your code healthy and, with minimal effort, make the project look professional. **pytest** is the standard for writing tests in Python, and you will have them from the start: tests for data, the model, and the API. **ruff** (an ultra-fast linter also written in Rust) and **black** (a formatter) guarantee that the code follows consistent conventions and has a uniform format, which matters because well-formatted code communicates care. And **pre-commit** ties all this to Git: it configures hooks that run linting and formatting automatically before every commit, making it impossible to push code that does not meet the standards. These tools are cheap to adopt and yield a very high return in the perceived quality of your project.

### Stack Summary

| Layer | Tool | What problem it solves |
|:---|:---|:---|
| Language | Python 3.11+ | ML standard |
| Environment | uv (or poetry) | Reproducible and fast dependencies |
| Code Versioning | Git + GitHub | History and collaboration |
| Data Versioning | DVC | Version large datasets without bloating Git |
| Data Validation | Pandera / Great Expectations | Data contracts that fail loudly |
| Experiment Tracking | MLflow | Traceability of what produced each result |
| Model Registry | MLflow Model Registry | Manage models as versioned artifacts |
| Inference API | FastAPI + Pydantic | Serve the model with validation and auto-docs |
| Containers | Docker + Compose | Full environment reproducibility |
| CI/CD | GitHub Actions | Automate tests, validation, and deployment |
| Orchestration | Prefect / ZenML | Chain and schedule flows with retries |
| Monitoring / Drift | Evidently AI | Detect model degradation in production |
| Deployment | Render / Railway / Fly.io / Modal | Bring the API online without managing infrastructure |
| Testing and Quality | pytest, ruff, black, pre-commit | Keep code healthy and professional |

---

## 4. The repository architecture

The folder structure of a project communicates professionalism **before anyone reads a single line of code**. A well-organized repository tells anyone who opens it that you know how real software is structured; a chaotic one breeds immediate distrust. This is why it is worth designing the structure consciously, following established principles rather than improvising.

The structure you will use is inspired by the widely recognized `cookiecutter-data-science` convention in the community, adapted to include MLOps components. It is as follows:

```
mlops-fraud-pipeline/
├── .github/
│   └── workflows/
│       ├── ci.yml                # tests + linting on every push
│       └── cd.yml                # build + deploy of the image
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
│   │   └── register.py           # registration in Model Registry
│   ├── api/
│   │   ├── main.py               # FastAPI app
│   │   ├── schemas.py            # Pydantic models
│   │   └── predict.py            # inference logic
│   ├── monitoring/
│   │   ├── drift.py              # Evidently reports
│   │   └── dashboard.py          # monitoring dashboard
│   └── config.py                 # centralized configuration
├── pipelines/
│   ├── training_pipeline.py      # training Prefect/ZenML flow
│   └── monitoring_pipeline.py    # scheduled monitoring flow
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
├── pyproject.toml                # dependencies (uv or poetry)
├── .pre-commit-config.yaml
├── Makefile                      # common commands (make train, make test...)
└── README.md                     # ⭐ the most important document in the project
```

Behind this organization lie several principles that are important to understand, as they are what truly matter and what you will be able to explain:

**Separation of concerns.** The core of the project lives in `src/`, divided into four subfolders that correspond to the four major responsibilities of the system: `data/` (everything related to obtaining, validating, and transforming data), `models/` (training, evaluating, and registering models), `api/` (serving predictions), and `monitoring/` (watching production health). This separation ensures that anyone opening the repository instantly understands where each piece of functionality lives. It is the architectural translation of the four lifecycle stages into folders.

**The `src/` layout.** Placing all application code inside a `src/` folder rather than leaving it in the root is a convention that avoids import issues and keeps the repository root clean, reserved for configuration files and documentation. It is a minor detail that experienced engineers recognize immediately.

**Notebooks in quarantine.** The `notebooks/` folder exists, but with a very narrow purpose: **only for initial exploration**, never for production code. This distinction is one of the most critical lessons of MLOps. Notebooks are magnificent for exploring data and testing ideas, but they are terrible for production code: they do not test well, do not version well, encourage hidden state, and are difficult to reproduce. Keeping them separate and marked as "exploration only" shows you understand this difference, which is precisely what separates someone who has only taken courses from someone who has worked in production.

**Pipelines separated from the code they orchestrate.** The `pipelines/` folder contains the orchestration flows (Prefect/ZenML), separated from `src/`. The idea is that `src/` contains the *logic* (how to train, how to validate), while `pipelines/` contains the *coordination* (the order in which those pieces run, their schedule, and their retries). Separating logic from its orchestration is a best practice that keeps each part focused and reusable.

**Centralized and versioned configuration.** There are two key files for configuration. `src/config.py` centralizes the code's configuration so that nothing is hardcoded across different files, which would make the system fragile. `params.yaml` contains the hyperparameters and pipeline parameters, versioned alongside the code; this means every experiment is tied to the exact parameters used, reinforcing reproducibility. Having explicit, centralized, and versioned configuration is another mark of maturity.

**The `Makefile` as an interface.** A `Makefile` in the root defines short commands for common tasks: `make setup` to prepare the environment, `make train` to train, `make test` for running tests, and `make serve` to spin up the API. This gives anyone arriving at the project a clear and simple interface, without having to remember long commands. It is a detail of courtesy toward whoever uses your project (including your future self) that communicates care.

**The `README.md`, the most important document.** Finally, in the root lives the README, which deserves its own emphasis because, counterintuitive as it may seem, **it has the greatest impact on your portfolio**. The vast majority of people who look at your project, especially recruiters, will never clone the repository or read the code; they will judge it by the README. An excellent README, complete with an architecture diagram, a video of the system in action, a clear explanation of the problem, and, above all, a section on **design decisions** explaining the *why* behind each choice, is what turns a good project into an impressive one. The code shows you know how to execute; the README shows you know how to think, and that is what hiring teams are truly looking for.

---

## Summary

Before writing anything, you have already defined the essentials: you are going to build a **living ML system** for fraud detection that trains, deploys, monitors, and retrains itself, reaching Level 2 of MLOps maturity. You have chosen a problem where natural drift justifies the entire infrastructure and connects with your interest in security. You have a stack of lightweight, modern, and free tools, each selected for a reason you know how to defend. And you have a repository architecture that communicates professionalism at first glance.

All of this shares a single common thread: every decision is designed so that when a senior engineer looks at your work, they conclude that you understand how Machine Learning actually works in production. With these clear foundations, you can begin the initial phases with the confidence of knowing not only what you are going to do, but why.
