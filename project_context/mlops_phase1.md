# Phase 1 — Data Pipeline and Versioning

> This is the phase where the project moves beyond scaffolding and begins to take shape. The goal is to build the path from raw data to training-ready features, doing so in a highly specific way: **versioned, validated, and reproducible**. Recall the principle from the fundamentals: bad data is the number one cause of failures in production, and those failures are silent. This entire phase exists to tame that source of problems: so that you always know what data you are working with, so that data that does not meet expectations fails loudly before contaminating anything, and so that anyone can regenerate your exact features from the source.

**Phase objective:** convert raw data into features in a reproducible and versioned manner.  
**Duration:** ~2 weeks (weeks 2–3 of the project).  
**Upon completion, you will have:** a declaratively defined data pipeline that goes from raw CSV to training and testing sets ready for use, with the data versioned by DVC, a validation layer acting as a quality contract, and a preprocessing pipeline that prevents data leakage and is saved to be reused in production.

---

## The Big Picture: A Reproducible Data Pipeline

Before diving into the specific steps, it is helpful to keep the overall structure of what you are going to build in mind, as each piece fits into the whole. The data flow of this phase has three chained stages:

```
   data/raw/creditcard.csv   (versioned with DVC)
            │
            ▼
   ┌──────────────────┐
   │  validate        │  ← Pandera: does the data meet the contract?
   └──────────────────┘     produces a validation report
            │
            ▼
   ┌──────────────────┐
   │  preprocess      │  ← feature engineering + split + scaling
   └──────────────────┘     (no data leakage, with the scaler saved)
            │
            ▼
   data/processed/train.parquet
   data/processed/test.parquet
   data/processed/preprocessor.joblib
```

Two technologies underpin this flow, and it is important not to confuse their roles. **DVC** handles *versioning* (what data exists, in which version) and *data pipeline orchestration* (the execution order of the stages, what needs to be re-run when something changes). **Pandera** handles *validation* (ensuring the data meets a quality contract). Together, they turn a manual preprocessing script into a reproducible pipeline with guarantees.

A quick note before starting: we did not add DVC to the project in Phase 0, so the first command of this phase is to add it as a dependency. We add it with Google Drive support in case you want to use it as a remote storage (you can omit this extra if you plan to use a local remote):

```bash
uv add --dev "dvc[gdrive]"
# or simply 'uv add --dev dvc' if using a local remote
```

---

## Step 1 — Initialize DVC and Configure the Remote

DVC is initialized within the Git repository because it is designed to work alongside it. From the project root:

```bash
dvc init
```

This creates a `.dvc/` folder with the configuration and a `.dvcignore` file. Just as you did with Git, you should version these configuration files:

```bash
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

Now configure a **remote**, which is where DVC will store the actual versions of the data (remember: Git only stores lightweight pointer files, while DVC stores the heavy data separately). You have several options depending on your needs:

The simplest option to get started is a **local remote**, which is simply a folder on your disk outside the repository:

```bash
dvc remote add -d localremote /path/to/dvc-storage
```

The `-d` flag sets it as the default remote. This works well for development, though it has the limitation that the data is not accessible from another machine.

For a portfolio, where you might want the project to be reproducible by anyone, a cloud remote is better. **Google Drive** is free and accessible:

```bash
dvc remote add -d gdriveremote gdrive://YOUR_DRIVE_FOLDER_ID
```

(You will need the ID of a Google Drive folder, which appears in its URL, and the `dvc[gdrive]` extra you installed earlier.) If you have an AWS account, **S3** is the standard professional choice (`dvc remote add -d s3remote s3://your-bucket/path`). To begin, a local remote is the fastest option; you can migrate to the cloud later without any issues.

Also version the remote configuration:

```bash
git add .dvc/config
git commit -m "Configure DVC remote"
```

---

## Step 2 — Version the Raw Data with DVC

Now you place the dataset under DVC control. The command is analogous to `git add`, but for data:

```bash
dvc add data/raw/creditcard.csv
```

This does several things that are worth understanding. First, DVC calculates a hash of the file and stores it in its internal cache. Second, it creates a small metadata file called `data/raw/creditcard.csv.dvc`: this text file contains the hash and points to the actual data, and this is what **is** versioned in Git. Third, DVC adds the CSV itself to a `.gitignore` so Git ignores it, as DVC is now responsible for it.

There is an important detail to reconcile with Phase 0 here. In Phase 0, we added a broad `data/raw/*` rule to the root `.gitignore`, but that would also hide `.dvc` files, which we **do** want to version. The cleanest solution is to let DVC manage the data ignore pattern on a per-folder basis (which it does automatically when running `dvc add`) and ensure that `.dvc` files are versioned. In practice, remove the broad rule for data from the root `.gitignore` and let DVC do its job. Then, version the pointer file:

```bash
git add data/raw/creditcard.csv.dvc data/raw/.gitignore
git commit -m "Version raw dataset with DVC"
```

And push the actual data to the remote:

```bash
dvc push
```

From now on, you have a very powerful capability: the joint versioning of code and data. If someone in the future (or you on another machine) clones the repository and runs `dvc pull`, they will get the exact code and data that corresponded to that commit. And if you checkout an old commit with `git checkout`, running `dvc checkout` will bring back the version of the data from that moment. This is the **data reproducibility** we discussed in the fundamentals, now realized.

---

## Step 3 — Data Ingestion

We begin writing the pipeline code. The first stage is ingestion, which in our case is straightforward since the data is a local CSV, but it is best to encapsulate it in a reusable function. Fill the file `src/data/ingest.py`:

```python
import pandas as pd

from src.config import RAW_DATA


def load_raw_data() -> pd.DataFrame:
    """Loads the raw transaction dataset."""
    return pd.read_csv(RAW_DATA)
```

This function will be used by both the validation and preprocessing stages, meaning the logic of "where the data comes from" lives in a single place. In a real system, this would be the stage where you connect to a database, an API, or a data warehouse; encapsulating it this way means that if the source changes, you only modify this file and the rest of the pipeline remains unaffected.

To make it work, also complete `src/config.py` (which we created empty in Phase 0) with the centralized paths of the project:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"

TARGET = "Class"
```

Centralizing paths and constants here, rather than scattering them throughout the code, adheres to the "centralized configuration" principle from the fundamentals: nothing is hardcoded in disjointed locations, and if you change a path, you do so in a single place.

---

## Step 4 — Validation with Pandera

This is the core piece of data quality and one of the stages that best demonstrates maturity, as almost no one coming solely from notebooks includes it. The idea is to define a **contract**: a specification of what shape the data must have to be acceptable, so that any data that does not comply is detected immediately and fails loudly, before contaminating training or, worse, reaching the production model [1].

Complete `src/data/validate.py`. We will use Pandera's functional API (`DataFrameSchema`), which fits well here because our dataset contains 28 PCA features (`V1` to `V28`) that we can generate programmatically:

```python
import json

import pandera.pandas as pa

from src.config import REPORTS_DIR
from src.data.ingest import load_raw_data

# The 28 anonymized features of the dataset
V_COLUMNS = [f"V{i}" for i in range(1, 29)]

# The contract: what shape the raw data must have to be valid
raw_schema = pa.DataFrameSchema(
    columns={
        "Time": pa.Column(float, pa.Check.ge(0)),
        **{v: pa.Column(float, nullable=False) for v in V_COLUMNS},
        "Amount": pa.Column(float, pa.Check.ge(0)),
        "Class": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,   # unexpected columns cause validation to fail
    coerce=True,   # attempts to convert to the declared type before validating
)


def validate_raw_data() -> dict:
    """Validates raw data against the contract and issues a report."""
    df = load_raw_data()

    # lazy=True accumulates ALL failures before raising an error,
    # instead of stopping at the first one
    raw_schema.validate(df, lazy=True)

    report = {
        "n_rows": len(df),
        "n_fraud": int(df["Class"].sum()),
        "fraud_rate": round(float(df["Class"].mean()), 6),
        "status": "passed",
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "validation.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    result = validate_raw_data()
    print(f"Validation passed: {result}")
```

It is worth stopping to look at the decisions embedded in this code, as each communicates a best practice:

The **schema as an explicit contract.** Each column declares its type and constraints: `Time` must be a non-negative number, the `V` features must be non-nullable floats, `Amount` cannot be negative (a negative amount would mean corrupt data), and `Class` can only be 0 or 1. If data arrives violating any of these rules, the validation fails. This converts implicit assumptions ("surely the amount is always positive") into verified guarantees.

The **strict mode (`strict=True`).** This causes validation to fail if unexpected columns appear. This is an invaluable protection in production: if an upstream change adds or renames a column, you find out immediately instead of letting data with a different structure silently flow to the model.

The **lazy validation (`lazy=True`).** By default, Pandera stops at the first error. With `lazy=True`, it runs all checks and gathers all failures into a single report before raising an exception, which is much more useful for diagnostics: you see everything that is wrong at once, not just the first issue.

The **validation report.** In addition to validating, the function outputs a small JSON report with key statistics (number of rows, fraud cases, fraud rate). This report serves two purposes: it acts as a record of data "health" for each run, and, as you will see in the pipeline step, it will be the output that connects validation with preprocessing in the DVC DAG.

> **Recommended Practice (2026):** Note the import `import pandera.pandas as pa` instead of the generic `import pandera as pa`. This is the recommended pattern since Pandera 0.29 and prepares your code for the future; there is also `import pandera.polars as pa` if you decide to use Polars instead of pandas. It is a minor detail that shows you are following the library's current conventions.

---

## Step 5 — Preprocessing and Feature Engineering

Now you transform the validated data into training-ready features. This stage contains two of the most important ideas in the entire project regarding Machine Learning correctness, so pay close attention: **avoiding data leakage** and **saving the preprocessing state to reuse in production**.

Complete `src/data/preprocess.py`:

```python
import joblib
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    PREPROCESSOR_PATH,
    PROCESSED_DIR,
    PROJECT_ROOT,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from src.data.ingest import load_raw_data


def load_params() -> dict:
    """Loads the versioned preprocessing parameters."""
    with open(PROJECT_ROOT / "params.yaml") as f:
        return yaml.safe_load(f)["preprocess"]


def build_preprocessor(scale_columns: list[str]) -> ColumnTransformer:
    """Builds the transformer: scales specified columns, lets the rest pass through."""
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), scale_columns)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def preprocess() -> None:
    params = load_params()
    df = load_raw_data()

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratified split: CRUCIAL due to class imbalance, to preserve
    # the proportion of fraud in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    preprocessor = build_preprocessor(params["scale_columns"])

    # fit ONLY on train; transform on test. This is where data leakage is avoided.
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    X_train_t.assign(**{TARGET: y_train.values}).to_parquet(TRAIN_PATH)
    X_test_t.assign(**{TARGET: y_test.values}).to_parquet(TEST_PATH)

    # We save the trained preprocessor to reuse it during inference
    joblib.dump(preprocessor, PREPROCESSOR_PATH)


if __name__ == "__main__":
    preprocess()
    print("Preprocessing completed.")
```

The two critical ideas, explained:

**Avoiding data leakage.** Data leakage occurs when information from the test set leaks into training, leading to misleadingly optimistic metrics that do not hold up in reality. The most common way to commit this is scaling the data *before* splitting: if you calculate the mean and standard deviation of the entire population to scale, that mean already includes information from the test set. The correct approach, shown in the code, is to split first, then **fit the scaling using only the training set** (`fit_transform` on `X_train`) and simply **apply** it to the test set (`transform` on `X_test`), without refitting. The test set must never influence how the data is transformed. Understanding and applying this correctly is a clear hallmark of machine learning rigor.

**Saving the preprocessor for production.** Recall the *training-serving skew* problem from the fundamentals: if the preprocessing in production differs even slightly from training, the model receives data in a form it did not learn, and performance silently drops. The solution is not to reimplement the preprocessing in the API, but to **save the exact fitted object** (using `joblib.dump`) and load it during inference to apply the identical transformation. This is why `preprocessor.joblib` is such an important output of this stage: it ensures production and training speak the same language.

There are also two details reflecting best practices. The **stratified split** (`stratify=y`) is essential here due to the extreme class imbalance: without it, random chance could leave the test set with very few (or zero) fraud cases, making the evaluation useless; stratifying preserves the proportion of fraud in both sets. And using **parquet** instead of CSV for the processed data is more space and speed efficient, and preserves column types.

> **A nuance for fraud detection:** We use a stratified random split for simplicity, but in fraud detection, a **temporal split** (training on older transactions and testing on more recent ones using the `Time` column) is more realistic. This reflects how the system operates in real life: you train on the past and predict the future. Additionally, it fits very well with the project's drift narrative. This is an improvement you can mention in the README or implement as a variant; recognizing this nuance demonstrates that you understand the problem beyond just a standard recipe.

It is also good practice to **validate the preprocessing output** with a second Pandera schema (checking that the processed dataset has the expected columns and types). This applies the principle of "validating at every boundary": not just at the input, but also at the output of each transformation. You can add this as an additional function in `validate.py`.

---

## Step 6 — Versioned Parameters

Notice that the preprocessing does not have magic numbers embedded; it reads its parameters from a `params.yaml` file. This is deliberate and is a core MLOps practice. Fill the `params.yaml` file (which we created empty in Phase 0):

```yaml
preprocess:
  test_size: 0.2
  random_state: 42
  scale_columns:
    - Time
    - Amount
```

The reason for keeping parameters in a versioned file, rather than inside the code, is twofold. First, **reproducibility**: every commit of the project is linked to the exact parameters used, so you can always know what configuration generated a result. Second, as you will see shortly, **DVC reads this file**: it can detect when you change a parameter and automatically re-run only the affected stages. Changing `test_size` from 0.2 to 0.3 will no longer be an invisible code change; it becomes a tracked change that triggers data regeneration.

Also notice a specific decision regarding the dataset: we only include `Time` and `Amount` in `scale_columns`. This is because the features `V1` to `V28` are already PCA components, which are essentially scaled at their origin. The only variables in their original scale are time and amount, and they are the ones that benefit from scaling. Documenting this kind of decision (why you scale some columns and not others) is exactly the reasoning expected of someone who understands what they are doing.

---

## Step 7 — The Declarative Pipeline with dvc.yaml

Now you tie the stages together in a reproducible pipeline. Instead of running the scripts manually and in order, you declare the pipeline in a `dvc.yaml` file and let DVC manage the orchestration. Create the `dvc.yaml` file in the root:

```yaml
stages:
  validate:
    cmd: uv run python -m src.data.validate
    deps:
      - src/data/validate.py
      - src/data/ingest.py
      - data/raw/creditcard.csv
    outs:
      - reports/validation.json:
          cache: false

  preprocess:
    cmd: uv run python -m src.data.preprocess
    deps:
      - src/data/preprocess.py
      - src/data/ingest.py
      - data/raw/creditcard.csv
      - reports/validation.json
    params:
      - preprocess.test_size
      - preprocess.random_state
      - preprocess.scale_columns
    outs:
      - data/processed/train.parquet
      - data/processed/test.parquet
      - data/processed/preprocessor.joblib
```

This file is the heart of pipeline reproducibility and is worth understanding well. Each **stage** declares three things: the command it runs (`cmd`), its dependencies (`deps`, the files it depends on), and its outputs (`outs`, the files it produces). Additionally, a stage can declare which parameters from `params.yaml` it depends on (`params`).

With this information, DVC builds a **dependency graph** (a DAG) between the stages and understands the flow: the `preprocess` stage depends on the report produced by `validate`, so DVC knows that validation must be executed (and pass) before preprocessing. This forms the validate → preprocess chain we outlined at the beginning.

But the most powerful aspect is how DVC uses this graph to be **smart about re-executions**. When you ask it to rebuild the pipeline, DVC checks what has changed (comparing hashes of dependencies, code, and parameters) and re-runs **only what is necessary**. If you change the preprocessing code but not the validation code, DVC re-runs only the preprocessing and reuses the cached result of the validation. If you change a parameter in `params.yaml`, DVC detects exactly which stages depend on it and regenerates only those. This is the same concept as a build system like Make, but applied to data and models.

> **Alternative:** Instead of writing `dvc.yaml` by hand, you can generate the stages with the `dvc stage add` command, which adds them to the file for you. Writing it manually, however, gives you a clearer understanding of the structure, which is what we aim for in this phase.

---

## Step 8 — Run and Verify the Pipeline

With everything defined, you can run the entire pipeline with a single command:

```bash
dvc repro
```

DVC traverses the DAG, runs the stages in the correct order (validate first, preprocess next), and generates all outputs. When finished, it will have created the files `train.parquet`, `test.parquet`, and `preprocessor.joblib`, and, crucially, it will have generated or updated a `dvc.lock` file.

That `dvc.lock` file is the reproducibility record of the pipeline: it captures the exact hashes of each dependency, parameter, and output of each stage in this specific run. It is the pipeline equivalent of what `uv.lock` is for dependencies. You must version it in Git, as it is what allows someone else to reproduce your exact pipeline:

```bash
git add dvc.yaml dvc.lock params.yaml
git commit -m "Define the data pipeline validate -> preprocess"
dvc push   # uploads outputs to the remote
```

You can visualize the pipeline graph to confirm the structure matches your expectations:

```bash
dvc dag
```

And here comes the demonstration that confirms everything works as intended, and serves as the true success criterion of the phase. Change a parameter in `params.yaml` (for example, `test_size` from 0.2 to 0.25) and run again:

```bash
dvc repro
```

You will see that DVC detects the parameter change that the `preprocess` stage depends on, and **automatically re-runs only the preprocessing**, regenerating the training and testing sets with the new ratio, without re-running the validation (whose dependencies did not change). This selectivity (understanding what to redo and what to keep) is the magic of a reproducible pipeline, and seeing it work confirms you have built something truly robust. You can check what is out of date at any time with `dvc status`.

---

## Step 9 — Data Tests

To close the phase with the same level of quality as previous phases, write some tests to verify that your validation is doing its job. This connects with the quality tools you configured in Phase 0 (pytest ran via `make test`). Complete `tests/test_data.py`:

```python
import pandas as pd
import pandera.pandas as pa
import pytest

from src.data.validate import V_COLUMNS, raw_schema


def _valid_row() -> dict:
    """Builds an example valid row."""
    return {
        "Time": [0.0],
        **{v: [0.1] for v in V_COLUMNS},
        "Amount": [10.0],
        "Class": [0],
    }


def test_schema_accepts_valid_data():
    df = pd.DataFrame(_valid_row())
    # Should not raise any exception
    raw_schema.validate(df)


def test_schema_rejects_invalid_class():
    data = _valid_row()
    data["Class"] = [5]  # value outside the allowed set {0, 1}
    with pytest.raises(pa.errors.SchemaError):
        raw_schema.validate(pd.DataFrame(data))


def test_schema_rejects_negative_amount():
    data = _valid_row()
    data["Amount"] = [-5.0]  # a negative amount represents corrupt data
    with pytest.raises(pa.errors.SchemaError):
        raw_schema.validate(pd.DataFrame(data))


def test_schema_rejects_unexpected_column():
    data = _valid_row()
    data["extra_column"] = [1.0]  # strict=True should reject this
    with pytest.raises(pa.errors.SchemaError):
        raw_schema.validate(pd.DataFrame(data))
```

These tests verify that the data contract works: that it accepts valid data and rejects invalid data for the correct reasons (a class out of range, a negative amount, or an unexpected column). Having tests on your validation demonstrates an uncommon level of care: you are not only validating the data, but also checking that your validation actually protects it. Run them with `make test` or `uv run pytest`.

---

## Verification: The "Definition of Done"

The phase is complete when the following conditions are met. The core criterion, as we saw, is that the pipeline is reproducible and selective:

- [ ] DVC is initialized and a remote is configured.
- [ ] The raw dataset is versioned with DVC (there is a `creditcard.csv.dvc`) and pushed to the remote using `dvc push`.
- [ ] The `validate.py` file defines a Pandera schema that acts as a contract and fails loudly when presented with invalid data.
- [ ] The `preprocess.py` file generates training and testing sets without data leakage (fit only on train) and saves the trained preprocessor.
- [ ] Preprocessing parameters live in a versioned `params.yaml` file.
- [ ] `dvc.yaml` defines the validate → preprocess stages with their dependencies, parameters, and outputs.
- [ ] `dvc repro` rebuilds the entire pipeline and generates the `dvc.lock` file, which is versioned.
- [ ] **The key test:** when changing a parameter in `params.yaml` and running `dvc repro`, DVC detects the change and re-runs only the affected stages.
- [ ] Data tests pass using `make test`.

If you pass the key test (changing a parameter and seeing DVC re-run exactly what it should, no more and no less), you have confirmation that you have built a genuinely reproducible data pipeline, rather than a sequence of scripts that you execute with your fingers crossed.

---

## Deliverables and Next Steps

By closing Phase 1, you have a complete and robust data pipeline: versioned raw data, a validation layer that guarantees its quality and fails loudly when something is off, a correct preprocessing pipeline that avoids data leakage and is saved for production, and all of this declaratively orchestrated by DVC in a way that is reproducible and selective. You have moved from "I have some data" to "I have a reliable, versioned process that converts raw data into ready-to-use features."

The next step, **Phase 2**, is where machine learning finally enters: you will use the `train.parquet` and `test.parquet` sets you just produced to train models, rigorously logging each experiment with MLflow and comparing parameters and metrics to find the best model. That entire phase is built directly on top of the features generated by this pipeline: the saved preprocessor and the datasets you created are, quite literally, the raw materials for the next stage. And because the pipeline is reproducible, you can connect the training stage to `dvc.yaml` as an additional step, extending the data graph into the model graph.
