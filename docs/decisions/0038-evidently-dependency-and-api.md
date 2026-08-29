# Decision 38: `evidently` as a production dependency, and reading its result by measurement

- **Date:** 2026-08-29
- **Status:** Accepted

## Context

Two questions arrived together with the first real `detect_drift()`, and both had a plausible
wrong answer already written down somewhere.

**The dependency group.** Phase 7 put `prefect` in `dev` ([0031](0031-prefect-dependency-group.md)),
and the obvious move was to follow that precedent. **The precedent, read properly, points the other
way** — 0031's rule is *"dev iff no module under `src/` imports it"*, with `dvc` as the true
analogue rather than `mlflow`. Evidently is the opposite case.

**The API.** `project_context/mlops_phase8.md:101` opens Step 3 with an unusually pointed warning:
Evidently has *"two very different generations of APIs"*, the pre-0.6.7 one importing from
`evidently.report` and `evidently.metric_preset`, the current one from `evidently` and
`evidently.presets` with the arguments to `run` reversed to `(current, reference)`. The document
then adds, about its own helper: *"the exact structure may vary depending on the version; inspect
`eval_result.dict()` in your installation to confirm the path."* We were separately handed a
skeleton written against the **old** generation — `report.as_dict()["metrics"][0]["result"]
["share_of_drifted_columns"]`.

## Decision

**`uv add evidently` — production, not dev.** Resolved to `evidently==0.7.21`, pulling 21 packages.

**Silenced in mypy by name**, appended to the existing override list alongside `sklearn.*`,
`imblearn.*` and `joblib`, after verifying it ships no `py.typed`.

**Every access to Evidently's result written against a probe of the installed version**, not
against either document. The aggregate is located by matching `config.type ==
"evidently:metric_v2:DriftedColumnsCount"` rather than by list index.

**`uv add --dev requests` deliberately not run**, though the document pairs it with the Evidently
install in the same preamble block: it serves the alerting and simulator steps, neither of which
is in this scope.

## Alternatives considered

- **`--dev`, mirroring `prefect`.** Rejected. `src/monitoring/drift.py` imports Evidently directly,
  and `.dockerignore` excludes `pipelines/` while `COPY . /app` ships **all** of `src/` into an
  image built with `uv sync --frozen --no-dev`. A dev placement would put an import of an
  undeclared package inside the image — inert today because nothing in the image imports
  `src.monitoring`, and a latent break the first time anything does.
- **`ignore_missing_imports = true` globally** instead of naming `evidently.*`. Rejected on the
  grounds 0026 already recorded: a global switch would also stop checking libraries that *do* ship
  types, and would silently absorb any future dependency's missing stubs.
- **Transcribing the supplied skeleton.** Rejected — it does not run. See below.
- **Reading the aggregate as `metrics[0]`**, which is where it happens to sit today. Rejected: the
  preset is free to reorder or add metrics between releases, and an index would then read a
  per-column p-value as if it were the dataset verdict. Both are floats, so nothing would raise —
  the loop would just start answering a different question.
- **Pinning an exact `==0.7.21`** rather than the `>=` `uv add` writes. Rejected as inconsistent
  with every other dependency here; `uv.lock` already pins the resolution for CI and clones.

## Justification

**The probe, run in this repository against 0.7.21:**

```
hasattr(Report([DataDriftPreset()]), "as_dict")   ->  False
Report([DataDriftPreset()], include_tests=True).run(current, reference).dict()
  top-level keys: ['metrics', 'tests']
  metrics[0]  DriftedColumnsCount(drift_share=0.5)
              config.type = 'evidently:metric_v2:DriftedColumnsCount'
              value = {'count': 4.0, 'share': 1.0}
  metrics[1:] ValueDrift(column=f1, method=K-S p_value, threshold=0.05)  -> float p-value
```

So the supplied skeleton was wrong three times over: `as_dict` does not exist, there is no
`result` key, and there is no `share_of_drifted_columns` key. It would have raised `AttributeError`
on the first call. The reference document's shape — `.dict()`, and a `value` dict carrying
`"share"` — is correct.

**The measurement that justified the row floor.** The same probe, with a 3-row current frame
against a 500-row reference:

```
share = 1.0   (count 4.0 of 4)
```

Evidently runs K-S regardless and reports every column drifted. `logs/predictions.jsonl` currently
holds exactly **3 records**, two of them all-zero smoke payloads. Without
`config.DRIFT_MIN_ROWS`, the first scheduled monitoring run after this lands would have fired a
retrain off that noise. The floor is 100 rows, overridable by environment variable.

**`py.typed` checked, not assumed:** `site-packages/evidently/py.typed` does not exist, while
`site-packages/prefect/py.typed` does — which is why `prefect` is genuinely type-checked and
`evidently.*` is not.

## Trade-offs / consequences

- **The runtime image grows by 21 packages** — `statsmodels`, `plotly`, `litestar`, `nltk`,
  `faker` among them — for code the API never calls. That is the honest cost of 0031's rule
  applied consistently: the alternative was a rule that bends whenever the result is inconvenient.
  The image also `.dockerignore`s `data/`, `reports/` and `logs/`, so `src/monitoring/` is
  importable there but has nothing to read.
- **Evidently is unchecked by mypy**, so a wrong attribute on a `Report` surfaces at runtime rather
  than in CI. Partly compensated by matching the metric on its config type and raising
  `DriftDetectionError` when the expected metric is absent — a shape change is reported as a
  failure instead of silently returning `0.0` as the reference material's helper does.
- **The test suite got noisier.** `litestar`, pulled in transitively, emits deprecation warnings on
  import: `pytest` went from 1 warning to 11. No test fails and nothing is configured to error on
  warnings, but the summary line is uglier than it was.
- **This is the first dependency added since the image contract in
  [0022](0022-container-image-contract.md) was written**, and the first whose weight is carried
  purely for a `src/` module that serves the lifecycle path rather than the prediction path. If
  that footprint ever matters, the correct fix is splitting the image, not relabelling the
  dependency.
