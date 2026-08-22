# Decision 15: Packaged model contract — artifacts injection, raw in, probability + decision out

- **Date:** 2026-08-22
- **Status:** Accepted

## Context

Phase 2's model was trained on `train.parquet`: features already scaled by the Phase 1
`ColumnTransformer`. The Phase 4 API will receive **raw** transactions — the columns of
`data/raw/creditcard.csv` minus `Class`, unscaled. Something must bridge that gap, and the
naive answer (reimplement the scaling inside the API) is *training-serving skew*, the silent
failure [0008](0008-preprocessor-persistence.md) exists to prevent.

`project_context/mlops_phase3.md` proposes an `mlflow.pyfunc.PythonModel` wrapper and shows it
with a three-argument constructor, `FraudModel(preprocessor, model, threshold)`, holding both
fitted objects as instance state.

## Decision

Package preprocessor + classifier + threshold as `FraudModel(mlflow.pyfunc.PythonModel)`, with
a contract that diverges from the reference material on one axis and matches it on another:

- **`__init__(self, threshold: float)` — threshold only.** The preprocessor and classifier
  arrive through `load_context(context)`, read from `context.artifacts` under the keys
  `"preprocessor"` and `"model"`. *(Diverges from the reference material.)*
- **`predict(...) -> pd.DataFrame`** with columns `fraud_probability` and `is_fraud`, indexed
  like the input. *(Matches the reference material, whose own Step 8 test asserts exactly these
  columns.)*
- **`log_model(..., code_paths=[".../src"])`**, shipping `src/` inside every version.

## Alternatives considered

- **The reference material's three-argument constructor** — rejected, but not for the reason
  first assumed. It was tested: a fitted `ColumnTransformer` *does* round-trip through
  cloudpickle by value (2,354 bytes, `get_feature_names_out()` intact), so that design is not
  broken. It was rejected because `context.artifacts` rebuilds each component through its own
  MLflow flavor, with the requirements MLflow recorded for it, instead of collapsing a
  scikit-learn pipeline and an XGBoost booster into one opaque blob whose library-version
  compatibility is implicit.
- **Returning a bare `pd.Series` of 0/1** — rejected. `CLAUDE.md` requires logging *"input +
  output + probability + timestamp"* for every prediction, and a pyfunc artifact exposes no
  channel other than `predict()`; a Series makes the probability structurally unreachable,
  breaking Phase 4's prediction log and Phase 8's drift inputs at once.
- **Omitting `code_paths`** — rejected on evidence. `FraudModel` is cloudpickled **by
  reference**: the pickle is 77 bytes and embeds the literal string `src.models.register`.
  Without `code_paths` the artifact loads only where this repository is importable, and model
  versions are immutable, so the omission could not have been repaired later. Verified by
  loading `models:/fraud-detector@production` from `/tmp`, with the repo off `sys.path`.

## Justification

The artifact is self-contained by construction: it carries the *same fitted preprocessor
object* used in training, so production preprocessing cannot drift from training preprocessing —
skew becomes structurally impossible rather than merely avoided by discipline. The registered
signature makes the contract public and machine-checkable: input `['Time', 'V1'…'V28',
'Amount']` (the raw order; the processed order would be `Time, Amount, V1…`), output
`['fraud_probability': float, 'is_fraud': long]`.

Consumers reach it through `load_production_model()`, which resolves
`models:/fraud-detector@production` and returns the artifact. It is a deliberately thin wrapper:
its value is that the alias URI is written once, so Phase 4 imports a function instead of
re-deriving a URI string in `src/api/`.

## Trade-offs / consequences

- **`code_paths` snapshots `src/` at registration time.** A later edit to `src/` does not change
  an already-registered version. Correct for an immutable artifact, but it means the code inside
  an old version can lag the repository.
- **The artifact is heavier** than a bare estimator: it carries the preprocessor, the model, and
  a copy of `src/`.
- **`load_context()` is not unit-tested**; it resolves real paths off disk, which is an
  integration concern. `predict()` is tested in isolation by injecting doubles directly, which
  is where the packaging logic actually lives.
- **The reference material's Step 4 snippet no longer applies verbatim**: `build_packaged_model()`
  returns `(FraudModel, artifacts)` and `register_model()` passes `artifacts=` to `log_model()`.
