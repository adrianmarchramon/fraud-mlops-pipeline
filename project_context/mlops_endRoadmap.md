# MLOps — Closing the Roadmap: Pitfalls, Enhancements, and Resources

> With all nine phases complete, you have the system built and presented. This document covers the surrounding aspects that elevate the project: pitfalls to avoid (cross-cutting lessons across the entire workflow), ways to take it further if you want to keep scaling it, resources for deeper learning, and a final summary of the journey. The final README checklist was already thoroughly developed in Phase 9, so it will not be repeated here.

---

## Common Pitfalls to Avoid

These pitfalls are cross-cutting across the entire project, and most are natural temptations for those transitioning from learning ML in notebooks. Understanding them is not just about avoiding errors; it means internalizing the philosophy that distinguishes a production ML engineer. Each one contains a lesson that resurfaces throughout all phases.

**Obsessing over the F1-score.** This is the most natural trap for those coming from academia, where the goal is to maximize a metric. In this project, the model is secondary; what is impressive is the infrastructure surrounding it. A "good enough" model with an excellent pipeline is worth infinitely more, both for your portfolio and for a company, than an extraordinary model trained blindly without a system around it. Do not waste weeks fine-tuning hyperparameters to squeeze out a percentage point of PR-AUC; invest those weeks in the system instead. Companies do not lack people who know how to train models; they lack people who know how to bring them to production.

**Committing data to Git.** Git is designed for text, not datasets. Versioning data in Git makes the repository huge, slow, and unmanageable—an immediate sign of inexperience. That is what DVC is for, which versions data separately and leaves only lightweight pointers in Git. This mistake is so common that avoiding it (with a `.gitignore` and a `.dockerignore` that exclude data, and DVC managing it) is one of the first things an experienced engineer checks.

**Treating notebooks as production code.** Notebooks are great for exploring data and testing ideas, but terrible as production code: they are not easily tested, do not version well, encourage hidden state, and are difficult to reproduce. The distinction between an "exploration notebook" and "code in `src/` for production" is one of the most important lessons in MLOps. Maintaining this boundary (keeping notebooks quarantined, solely for exploration) demonstrates that you understand what separates someone coming from online courses from someone who has worked in production.

**Neglecting the README.** As counterintuitive as it may seem, the README has more impact on the perception of your project than the code itself, because most people will judge it without cloning the repository. A poor README wastes all your technical work; an excellent one multiplies its value. Spending real time on it, especially on the design decisions section, is not optional—it is where the impression you make is won or lost.

**Failing to close the drift loop.** The closed loop (detecting drift → automatic retraining) is the most impressive part of the entire project and what sets it apart the most. Leaving it half-done (detecting drift but not triggering retraining) wastes the very piece that turns your model into a living system. Make sure to close this loop and demonstrate it in your video; it is your most valuable asset.

**Skipping tests.** A project without tests looks amateur, no matter how good the code is. Tests are not an afterthought at the end; they are built from the start, phase by phase: testing data, the model, the API, and monitoring. They demonstrate care and professionalism, and they are what gives CI/CD its purpose. Having them from day one saves you from rewriting the project to add them later.

**Hardcoding configuration.** Scattering fixed values throughout the code (paths, parameters, addresses) makes the system fragile and impossible to adapt across environments. Configuration must be centralized (in `config.py` and `params.yaml`), and variables that change between environments should be injected via environment variables, just as you did with the MLflow address. This is not pedantry; it is what makes the same code work identically on your machine, in a container, and in the cloud.

**Using Kubernetes "to impress."** Paradoxically, this is what least impresses a senior engineer, as it reveals that you do not know how to match the tool to the problem. For an individual project, Kubernetes is unnecessary and counterproductive overhead. Choosing a lightweight deployment option (such as Render, Modal, or Fly.io) and knowing how to explain why demonstrates the exact opposite: sound judgment. The correct tool is not the most powerful one, but the one suited to the scale of the problem, and that maturity is worth more than any poorly applied trendy technology.

The common thread running through all these pitfalls is the same: each one represents the difference between treating ML as an academic exercise and treating it as production engineering. Internalizing these lessons is, ultimately, what this entire project is about.

---

## Stretch Goals: If You Want to Go Further

Once you have the complete system up and running, these extras take it to the next level and demonstrate an even deeper mastery. They are not necessary for the project to impress, but each is a recognizable component of mature ML platforms, and adding one or two (well-executed and well-documented) will distinguish you even further. They are worth knowing about even if you do not implement them, because being able to discuss them in an interview already demonstrates solid judgment.

**Model A/B testing (champion/challenger).** This involves serving two versions of the model at the same time and comparing their performance on real traffic before deciding which one stays. The current production model is the "champion"; a new candidate is the "challenger." You route a fraction of traffic to the challenger, record the results of both, and compare their performance in the real world rather than on a test set. This naturally connects with the Model Registry you set up: you would use aliases like `@champion` and `@challenger` to manage both versions. It is of medium-high difficulty and demonstrates that you understand how to safely introduce new models to production, measuring before committing.

**Feature Store (Feast).** A *feature store* centralizes the definition and serving of features, ensuring that training and inference use the exact same features and allowing features to be reused across models. It is the enterprise-scale solution to the *training-serving skew* problem you addressed by packaging preprocessing. Feast is the industry-standard open-source feature store. You would define "feature views," materialize them, and serve them for both training and prediction. It is highly difficult, but a feature store is one of the hallmarks of a mature ML platform, and very few junior engineers even know what it is.

**Prometheus and Grafana.** This adds system-level observability, complementing drift monitoring. While Evidently watches data and model quality, Prometheus and Grafana monitor the operational health of your API: response latency, requests per second, error rates, and resource usage. You would expose a metrics endpoint in the API (there are libraries that integrate this with FastAPI in just a few lines), Prometheus would scrape them periodically, and Grafana would display them in dashboards. It is of medium difficulty and demonstrates that you understand that monitoring an ML system has two sides: the model and the infrastructure.

**Shadow deployment.** This is a safe deployment technique where a new model runs in parallel on real traffic, making predictions that **do not affect** production decisions, to validate it against real data before activating it. Unlike A/B testing, it has zero impact on users because the shadow model's predictions are only logged and compared, never used. Every request would be sent to both the production and shadow models, logging both for comparison. It is of medium-high difficulty and is an advanced technique that demonstrates a sophisticated understanding of model deployment.

**Infrastructure as Code (Terraform).** Instead of configuring your cloud infrastructure by clicking through a console, you define it declaratively in versioned files so that it is reproducible and auditable. You would write Terraform configurations describing the resources you need (the deployment service, databases, storage), and Terraform spins them up for you. This connects directly with the deployment in Phase 9, elevating it from "I configured it manually" to "I define it as code." It is of medium difficulty, and infrastructure as code is a central practice of DevOps and MLOps that shows you think about reproducible infrastructure.

**Explainability (SHAP in the API).** This means having every prediction accompanied by an explanation of *why*: which features drove it. SHAP values explain individual predictions, attributing to each feature its contribution to the decision. You would integrate this into the API, returning alongside each prediction an explanation of the most significant factors. This is especially relevant in fraud and finance, regulated domains where explainability is often mandatory, and it demonstrates a responsible AI mindset. Furthermore, it connects brilliantly with the ninth project in your portfolio plan (the explainability dashboard), allowing you to repurpose the work. It is of medium difficulty.

Beyond these six, there are other natural directions if you want to keep going: more thorough data validation with Great Expectations, retraining triggered by performance drops (not just drift), cost monitoring, or serving an ensemble of models. But with one or two of the above, well-implemented and documented, you will have a project that clearly stands out.

---

## Learning Resources

To dive deeper into MLOps beyond this project, these are the most valuable resources, along with a note on what each provides so you know where to turn.

**Made With ML, by Goku Mohandas.** This is probably the most complete free reference for learning end-to-end MLOps. It covers the entire lifecycle with a very practical, well-structured approach, making it an excellent complement to this project for understanding the "why" behind each piece in greater depth.

**MLOps Zoomcamp, by DataTalksClub.** A free, highly practical course available on GitHub that walks through the lifecycle of an ML system with concrete exercises. It is ideal if you learn best by doing, and many of its components (tracking, orchestration, deployment, monitoring) overlap with those in your project, reinforcing what you have learned from another angle.

**"Designing Machine Learning Systems", by Chip Huyen.** This is the definitive book on designing ML systems in production. It is not a tool tutorial, but rather a guide to the concepts, trade-offs, and architectural decisions that underpin everything you have built. Reading it will give you the mental framework to reason about any ML system, not just this one, and it is exactly the kind of understanding that shines in an interview.

**Official tool documentation.** The documentation for DVC, MLflow, Evidently, Prefect, FastAPI, and uv are all high quality and serve as the primary, most up-to-date sources. When you have a specific question about a tool, going to its official documentation is almost always better than a third-party tutorial, which may be outdated—as you have seen throughout this project with APIs that have changed.

Regarding **datasets** for practicing with the fraud problem (or variations), Kaggle remains the best source: the classic Credit Card Fraud Detection to start, IEEE-CIS Fraud Detection for something richer and more realistic, and PaySim (synthetic data) if you want to simulate drift with more control.

There is also a set of **concepts** worth mastering, as they underpin the project and are most frequently asked about: MLOps maturity levels (from level 0 manual to level 2 with full CI/CD, which is where your project lands), the difference between data drift and concept drift, the trade-off between precision and recall and how to choose the threshold based on business needs, and the training-serving skew problem and how to avoid it. If you can explain these four fluently, you demonstrate an understanding that goes far beyond simply having followed a recipe.

---

## Summary of Milestones

As a recap of the entire journey, here is the complete phase-by-phase roadmap of the project:

| Phase | Milestone | What It Demonstrates |
|-------|-----------|----------------------|
| 0 | Repo, environment, and problem defined | Solid, reproducible foundations |
| 1 | Versioned data pipeline (DVC) | Data reproducibility and validation |
| 2 | Tracked experiments (MLflow) | Traceability and experimental rigor |
| 3 | Model Registry and packaging | Model management as artifacts |
| 4 | Inference API (FastAPI) | The model converted into a product |
| 5 | Containerization (Docker) | Full system portability |
| 6 | CI/CD (GitHub Actions) | Automated quality and delivery |
| 7 | Orchestration (Prefect) | Resilient, programmable workflows |
| 8 | Monitoring and drift (Evidently) | Closed-loop: a living system |
| 9 | Deployment and presentation | Project online and ready to impress |

---

## A Final Reflection

You have walked a path that very few junior-level engineers complete, and its value is worth recognizing. You have not just built a model: you have built a **production Machine Learning system**, from start to finish, with all the pieces that matter and that almost no one demonstrates. More important than any specific tool, what you have internalized is a way of thinking: that production ML is engineering, that the model is only a small part of it, and that reproducibility, automation, observability, and presentation are what separate an experiment from a product.

This learning transcends this project. The structure, best practices, and concepts you have applied here are the same ones you will use in any ML system you build in the future, and the same ones you will reuse in the other projects of your portfolio plan, accelerating your progress. You have laid a foundation that already makes you a strong candidate. When an experienced engineer looks at this work, the conclusion will be exactly what we set out to achieve from the very first document: this person understands how production Machine Learning actually works. Which is precisely what companies are looking for.
