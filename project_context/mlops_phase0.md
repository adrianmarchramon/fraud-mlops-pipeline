# Phase 0 — Setup and Definition

> This is the first week of the project and the only one in which you will not work directly with Machine Learning. Its purpose is to build the foundation: setting up the environment, structuring the repository properly, getting the quality tools running, and above all, deeply understanding the data to make the design decisions that will shape everything else. It is tempting to skip this phase to "get to the interesting part," but rushing this setup is a common cause of projects turning into chaos halfway through. The foundation is not visible in the final picture, but it supports the entire building.

**Phase objective:** To have the environment and the problem thoroughly defined before writing any ML code.  
**Duration:** ~1 week (10-15 hours).  
**When finished, you will have:** A repository that anyone can clone and install cleanly, with the complete structure, active quality tools, the dataset downloaded and explored, and a decision document justifying the approach.

---

## Before Starting: What You Need Installed

Before the first step, ensure you have this base set of tools available. These are the tools you will need throughout the entire project, not just in this phase.

**Git.** The version control system. Check that you have it with `git --version`. If not, install it from [git-scm.com](https://git-scm.com) or using your system's package manager.

**uv.** The Python environment and dependency manager, which will be your primary tool for everything related to your environment. One of its main benefits is that you do not even need Python pre-installed: uv handles downloading and managing the Python version you request. Install it with a single command:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify with `uv --version`. uv replaces pip, pyenv, virtualenv, and poetry on its own, meaning a single tool covers Python version management, virtual environments, dependencies, and command execution.

**Docker Desktop.** You will need this from Phase 5 onward for containerization, but it is best to install it now to avoid surprises. Download it from [docker.com](https://www.docker.com/products/docker-desktop). Verify with `docker --version`.

**A GitHub account.** Where the repository will live, which is a central part of your portfolio. If you do not have one, create it at [github.com](https://github.com).

**A Kaggle account.** Where you will download the dataset. Create it at [kaggle.com](https://www.kaggle.com); you will configure its API later to download the data via the command line.

**A code editor.** VS Code is recommended because it integrates well with the tools in this stack (uv, ruff) and with notebooks, but any modern editor works. If you use VS Code, install the Python and Ruff extensions.

With this ready, let's begin.

---

## Step 1 — Create and Clone the Repository

The first step is to create the repository on GitHub, as it will host the project and we want everything version-controlled from day one.

Go to GitHub and create a new repository. Give it a clear, descriptive name, such as `mlops-fraud-pipeline`. Initialize it with a README, add a `.gitignore` template for Python, and choose a license (the MIT license is a common, straightforward option for portfolio projects). These three elements (README, .gitignore, and license) are minimal indicators of a well-maintained repository and should be present from the start.

Once created, clone it to your machine:

```bash
git clone https://github.com/YOUR_USERNAME/mlops-fraud-pipeline.git
cd mlops-fraud-pipeline
```

From this point on, you will work inside this folder. A good habit to build early is making small, frequent commits with clear messages: Git history is part of what an engineer will review, and a clean history that shows the project's evolution communicates professionalism.

---

## Step 2 — Initialize the Project with uv

Now, convert the folder into a Python project managed by uv. Inside the repository, run:

```bash
uv init
```

This creates the basic structure of a modern Python project: a `pyproject.toml` file (the central file where dependencies and tool configurations are declared), a `.python-version` file (which pins the project's Python version), and a startup file. uv also ensures that the `.gitignore` has sensible entries for Python.

Next, pin the Python version the project will use. For this project, Python 3.11 or higher is appropriate:

```bash
uv python pin 3.11
```

This writes `3.11` to the `.python-version` file. If you do not have it installed, uv downloads that Python version automatically. The value of pinning the version is reproducibility: anyone who clones the project will use the exact same version of Python as you, eliminating the classic "it works on my machine" issue.

---

## Step 3 — Add Dependencies

Now, declare the libraries the project needs. uv distinguishes between two types of dependencies, and understanding this distinction is important: production dependencies (needed for the project to run in production, such as ML libraries) and development dependencies (only needed while developing, such as testing frameworks or linters, which should not end up in the production image).

Start with production dependencies. You do not need to add them all now; you can introduce them as you progress through the phases. However, having the core data and ML libraries from the start is helpful for exploration:

```bash
uv add pandas scikit-learn matplotlib seaborn
```

`pandas` for data manipulation, `scikit-learn` for ML models and utilities, and `matplotlib` alongside `seaborn` for exploratory visualizations.

> **Note for 2026:** A modern alternative to pandas is **Polars**, which uses lazy execution and is notably faster and more memory-efficient, especially with large datasets. To get started and maintain compatibility with most tutorials, pandas is highly suitable; if you want to demonstrate familiarity with current tools, Polars is an excellent choice. You can start with pandas and migrate later if you wish.

Now, add the development dependencies using the `--dev` flag:

```bash
uv add --dev pytest ruff pre-commit jupyter kaggle
```

Here, you have `pytest` (for tests), `ruff` (the linter and formatter), `pre-commit` (for quality hooks), `jupyter` (for the exploration notebook), and `kaggle` (the CLI to download the dataset).

Every time you run `uv add`, uv resolves the dependencies, installs them in the project's virtual environment (in a `.venv` folder it creates automatically), and updates two files: `pyproject.toml`, where the dependencies are declared, and `uv.lock`, a lockfile that records the exact versions of everything. This `uv.lock` file is key to reproducibility: it guarantees that when someone else (or your CI) runs `uv sync`, they get the exact same environment as you, down to the last sub-dependency. Both files must be versioned in Git.

> **Important note:** With uv, you do not need to manually "activate" the virtual environment. To run any command within the project's environment, use the `uv run` prefix. For example, `uv run pytest` runs pytest inside the environment, and `uv run python script.py` runs a script. This avoids the common issue of forgetting to activate the environment.

---

## Step 4 — Create the Folder Structure

With the project initialized, set up the complete folder structure designed in the fundamentals. Although many folders will be empty for now, creating them from the start makes the project organization clear and prevents you from improvising locations later.

You can create the entire structure at once with these commands (on macOS/Linux; on Windows, use PowerShell or create them manually):

```bash
# Folders
mkdir -p data/raw data/processed
mkdir -p src/data src/models src/api src/monitoring
mkdir -p pipelines tests notebooks docker

# Module files (empty for now)
touch src/__init__.py src/config.py
touch src/data/__init__.py src/data/ingest.py src/data/validate.py src/data/preprocess.py
touch src/models/__init__.py src/models/train.py src/models/evaluate.py src/models/register.py
touch src/api/__init__.py src/api/main.py src/api/schemas.py src/api/predict.py
touch src/monitoring/__init__.py src/monitoring/drift.py src/monitoring/dashboard.py
touch pipelines/training_pipeline.py pipelines/monitoring_pipeline.py
touch tests/test_data.py tests/test_model.py tests/test_api.py
touch docker/Dockerfile docker/docker-compose.yml
touch params.yaml
```

The `__init__.py` files are important: they convert the folders into Python packages, allowing you to import code cleanly between modules (for example, `from src.data.preprocess import ...`). This detail makes your `src/` directory behave like a well-structured package rather than a collection of loose scripts.

It is also useful to set up the `.gitignore` file (uv will have already added Python entries, but ensure it includes the following) to avoid versioning data or heavy artifacts:

```gitignore
# Data (managed by DVC starting from Phase 1)
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Environment and artifacts
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
mlruns/
*.pyc

# Notebooks (checkpoints)
.ipynb_checkpoints/
```

The `data/` directory deserves an explanation: we want the data folders to exist in the repository (so the structure is complete), but we do not want the actual data to be uploaded to Git. That is why we ignore their content but preserve the folders using empty `.gitkeep` files (`touch data/raw/.gitkeep data/processed/.gitkeep`). Starting in Phase 1, DVC will take care of versioning the actual data.

---

## Step 5 — Configure Code Quality

Now, activate the tools that automatically keep the code clean and professional. This configuration is straightforward to implement and offers a significant return in how the project is perceived.

First, configure **ruff** by adding its setup to the end of your `pyproject.toml`. Ruff serves as both a linter (detecting issues and bad practices) and a formatter (providing consistent code formatting), allowing you to cover both with a single tool:

```toml
[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "B"]
```

The `select` section activates several rule sets: style errors (`E`), logical errors detected by pyflakes (`F`), automatic import sorting (`I`), naming conventions (`N`), syntax modernization (`UP`), and common bug detection (`B`). This is a sensible and thorough configuration without being overly restrictive.

> **Note for 2026:** The original roadmap mentioned `ruff` and `black`, but ruff now includes its own formatter (`ruff format`) which is compatible with black's style. This means you no longer need black: ruff acts as both linter and formatter, leaving you with a single tool and a single configuration file. This is the current recommended practice and simplifies the stack.

Next, configure **pre-commit**, which will run these checks automatically before each commit, making it very difficult to commit code that does not meet the standards. Create a `.pre-commit-config.yaml` file in the root directory:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.0   # use the latest version; then run 'pre-commit autoupdate'
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

The first two hooks run ruff (automatically fixing what it can) and the formatter. The following are highly useful generic checks: they remove trailing whitespace, ensure files end with a newline, validate YAML syntax, and, particularly useful for this project, prevent large files from being added (which stops you from accidentally committing a dataset to Git).

Once the file is created, install the hooks in your repository:

```bash
uv run pre-commit install
```

From now on, every time you run `git commit`, pre-commit will execute these checks, and if anything fails, it will block the commit until it is fixed. To run the checks manually across the entire project at any time, run `uv run pre-commit run --all-files`. To update the hook versions to the latest available, run `uv run pre-commit autoupdate`.

---

## Step 6 — The Makefile as the Project Interface

To ensure that anyone (including your future self) can use the project without remembering long commands, create a `Makefile` in the root directory with shortcuts for common tasks:

```makefile
.PHONY: setup lint format test train serve clean

setup:        ## Install dependencies and configure pre-commit
	uv sync
	uv run pre-commit install

lint:         ## Check code with ruff
	uv run ruff check .

format:       ## Format code with ruff
	uv run ruff format .

test:         ## Run tests
	uv run pytest

train:        ## Train the model (available starting from Phase 2)
	uv run python -m src.models.train

serve:        ## Launch the API (available starting from Phase 4)
	uv run uvicorn src.api.main:app --reload

clean:        ## Clean caches
	rm -rf __pycache__ .pytest_cache .ruff_cache
```

This way, a new contributor only needs to run `make setup` to get everything ready, or `make test` to run the tests, without needing to know the underlying details. This is a practice that communicates care and maturity.

> **Watch out for a classic detail:** Makefiles require **tabs**, not spaces, for command indentation. If your editor converts tabs to spaces, the Makefile will fail with a confusing error. Configure your editor to respect tabs in this file. If you are working on Windows and do not have `make`, you can install it or, alternatively, document the commands directly in the README.

---

## Step 7 — Download the Dataset

Now, acquire the data. We will use the Kaggle API to download it via the command line, which is more reproducible than downloading it manually from a browser.

First, configure your Kaggle credentials. Go to your Kaggle profile, visit the account section, and create an API token: this downloads a `kaggle.json` file. Place it in the directory expected by the CLI and restrict its permissions for security:

```bash
# macOS / Linux
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

(On Windows, the file goes in `C:\Users\YOUR_USERNAME\.kaggle\kaggle.json`.)

Now, download the dataset. To start, the classic **Credit Card Fraud Detection** dataset is a solid choice: it contains around 284,000 real transactions, is highly imbalanced (fraud is only about 0.17% of cases, making it well-suited for practicing imbalance handling), and has a manageable file size. Download it to `data/raw`:

```bash
uv run kaggle datasets download -d mlg-ulb/creditcardfraud -p data/raw --unzip
```

This leaves a `creditcard.csv` file in `data/raw/`. One characteristic of this dataset is that, for confidentiality, most features are anonymized as PCA components (`V1` to `V28`); only `Time`, `Amount`, and the label `Class` (0 = legitimate, 1 = fraud) are interpretable. This limits feature engineering slightly, which is acceptable for getting started.

> **If you want more depth:** once you master the workflow, consider migrating to the **IEEE-CIS Fraud Detection** dataset (non-anonymized features, more realistic, better for demonstrating feature engineering) or **PaySim** (synthetic data, excellent for simulating drift later on). For Phase 0 and getting started, the classic dataset is a solid choice due to its simplicity.

---

## Step 8 — Initial Exploratory Data Analysis (EDA)

This is the most substantive step of the phase and one you should not rush, because understanding the data is what will allow you to make sound decisions later. You will use a notebook here, as interactive exploration is exactly what notebooks are designed for. Create `notebooks/01_exploration.ipynb` and open it:

```bash
uv run jupyter lab
```

Remember the principle from the fundamentals: this notebook is **only for exploration**, never production code. Its purpose is for you to understand the data, not to produce reusable assets.

These are the questions your exploration must answer, along with concrete checks for each:

**What is the shape and structure of the data?** Start with the basics: how many rows and columns there are, the data type of each column, and a quick look at the first few rows.

```python
import pandas as pd

df = pd.read_csv("../data/raw/creditcard.csv")
print(df.shape)
df.info()
df.head()
df.describe()
```

**Are there null values or quality issues?** Check for missing data, as this will influence your preprocessing.

```python
df.isnull().sum()
df.duplicated().sum()  # are there duplicate rows?
```

**How imbalanced is the problem?** This is the most important question in fraud detection, as the imbalance dictates the model choice, metrics, and strategy. Look at the exact proportion of each class:

```python
df["Class"].value_counts()
df["Class"].value_counts(normalize=True)  # proportion
```

You will see that the positive class (fraud) is a miniscule fraction. Internalize that number: it is the reason why accuracy will be unhelpful as a metric and why you will need to use imbalance handling techniques.

**How are the features distributed?** Explore the distributions of the interpretable variables (`Amount`, `Time`) and observe if they differ between legitimate and fraudulent transactions. Differences in distributions between classes are a good sign: they indicate information the model can leverage.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Distribution of Amount by Class
sns.histplot(data=df, x="Amount", hue="Class", bins=50, log_scale=(False, True))
plt.show()

# Compare Amount statistics between classes
df.groupby("Class")["Amount"].describe()
```

**Are there relevant correlations?** A correlation heatmap gives you a quick overview of how features relate to each other and to the label.

```python
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.show()
```

As you explore, **write down your observations** in markdown cells within the notebook itself: the proportion of fraud, which features appear to be discriminative, what quality issues you detected, and what preprocessing decisions you anticipate. These notes will form the basis of the next step and the feature engineering in Phase 1. Exploration is not about generating nice-looking plots, but about building an understanding that guides the rest of the project.

---

## Step 9 — Documenting Design Decisions

Before wrapping up the phase, spend some time on a task that is often overlooked but sets a project apart: **documenting your decisions and their reasoning in writing**. It is not enough to make good decisions; being able to articulate them demonstrates solid judgment in an interview, and keeping them written down prevents you from forgetting or contradicting them later.

Create a document, such as `docs/decisions.md` (a decision log, often called an Architecture Decision Record or ADR), where you note each important choice and its justification. In this phase, the key decisions to document are:

**The business metric that matters to you.** This is the most critical decision of the phase. As discussed in the fundamentals, accuracy is unhelpful in highly imbalanced fraud detection, and you must deliberately choose which type of error concerns you more. Document your reasoning: you will likely prioritize a **high recall** (detecting as much fraud as possible, since letting fraud slip through is costly), while controlling precision to avoid generating too many false positives that disrupt legitimate customers. Document that you will use the **precision-recall curve and its area (PR-AUC)** as your primary metric instead of ROC-AUC, specifically because under severe imbalance, ROC-AUC can give a deceptively optimistic impression. This is exactly the kind of reasoning an interviewer looks for.

**Cost asymmetry.** Note your reasoning regarding the relative cost of a false positive versus a false negative, even if using estimates. Framing the problem in terms of business cost, rather than just abstract metrics, elevates the project from a technical exercise to a thoughtful solution.

**Dataset choice and limitations.** Document which dataset you are using and why, including its known limitations (such as features being anonymized as PCA, which limits feature engineering). Acknowledging the limitations of your own approach demonstrates professional honesty and maturity.

**Stack decisions already made.** Although you developed these in the fundamentals, it is helpful to have a summary of the core choices here (uv, MLflow, lightweight deployment without Kubernetes, etc.), as you will refer to this when writing the "design decisions" section of the README.

---

## Step 10 — The Initial README

Finally, write an initial version of the README. It does not need to be the final version (you will polish that in Phase 9), but it should set up the structure and clearly convey what the project is about. The README is the most impactful document in your portfolio, as many people will review the project without cloning it.

At this stage, the initial README should contain at least: a title and an engaging one-line description; an explanation of the problem (fraud detection) and why it matters; the planned architecture diagram (you can reuse the one from the fundamentals); the technology stack you will use; and basic installation instructions (`make setup`). Leave it ready to expand: as you complete phases, you will add results, screenshots, and the design decisions section.

A concrete recommendation: create the architecture diagram using a tool like [Excalidraw](https://excalidraw.com) or [draw.io](https://draw.io), which produce clean, professional diagrams, and embed it as an image in the README. A good visual diagram near the top of the README makes an excellent first impression.

---

## Verification: The "Definition of Done"

Do not consider the phase complete until it passes this check, which is the true measure of success: **a clean clone must work**. This means anyone (or you on another machine) should be able to clone the repository and set it up without hidden steps or relying on knowledge only in your head.

To verify this, pretend to be that person. Clone the repository into a new, clean folder and run the setup command:

```bash
cd /tmp
git clone https://github.com/YOUR_USERNAME/mlops-fraud-pipeline.git clean-test
cd clean-test
make setup
```

If `make setup` installs the environment and configures pre-commit without errors, leaving the project ready for work, you have passed the test. If it fails at any point, it means a dependency or a step is not properly documented or versioned, and you must fix it before moving forward. This "clean clone" test is one of the best habits to form: it ensures your project is truly reproducible and not a house of cards that only stands on your local machine.

Also, review this final checklist:

- [ ] The repository exists on GitHub with a README, .gitignore, and license.
- [ ] The project is initialized with uv, with the Python version pinned.
- [ ] Dependencies are declared and `uv.lock` is version-controlled.
- [ ] The complete folder structure exists, including `__init__.py` files.
- [ ] ruff and pre-commit are configured, and hooks are installed.
- [ ] The Makefile contains common commands.
- [ ] The dataset is downloaded to `data/raw` (and NOT committed to Git).
- [ ] The exploration notebook answers key questions about the data.
- [ ] The decisions document justifies the business metric and approach.
- [ ] The initial README clearly explains what will be built.
- [ ] The clean clone test (`make setup`) runs without errors.

---

## Deliverables and What Comes Next

Upon wrapping up Phase 0, you have the project's foundation: a clean, structured, and reproducible repository; quality tools that run automatically; the data downloaded and, most importantly, understood; and a decision record articulating the reasoning behind your approach. None of this trains a model yet, but this is what ensures the rest of the project is built on solid ground rather than sand.

The next step, **Phase 1**, is where the serious data work begins: you will initialize DVC to version the data, write validation checks with Pandera that act as a quality contract, build the preprocessing and feature engineering, and define the declarative data pipeline in `dvc.yaml`. That entire phase relies directly on the data understanding you gained here: the preprocessing decisions you noted during exploration will become the code for Phase 1. That is why the time spent understanding the data now pays off significantly in the next step.
