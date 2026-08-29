# Decision 37: The drift reference dataset — training rows, raw space, frozen as a stage output

- **Date:** 2026-08-29
- **Status:** Accepted

## Context

Drift is a comparison, and this project had never written down what the comparison is *against*.
Phase 7 shipped `detect_drift()` as a constant, so the question never arose; Phase 8 cannot avoid
it, because the number the function returns is meaningless until one side of the comparison is
pinned.

Three axes had to be settled, and only the first is the one the reference material discusses:

1. **Which rows** — the training split, the held-out test split, or the whole raw file.
2. **Which feature space** — raw units, or the scaled space `preprocess` writes.
3. **Computed or materialised** — recomputed per call, or persisted as an artifact.

`project_context/mlops_phase8.md:70-84` answers only implicitly, with
`pd.read_csv(RAW_DATA).drop(columns=[TARGET]).sample(n=5000, random_state=42)` computed inside
`load_reference()` on every call.

## Decision

**Training rows, in raw feature space, materialised as `data/monitoring/reference.parquet` by a
dedicated `reference` stage in `dvc.yaml`.**

| Axis | Choice |
|---|---|
| Rows | The training split, re-derived from the raw CSV with `preprocess`'s own parameters |
| Space | Raw units — no scaler applied |
| Size / seed | `monitoring.rows: 5000`, `monitoring.random_state: 42` in `params.yaml` |
| Format | Parquet |
| Location | `data/monitoring/reference.parquet`, git-ignored by DVC's own per-directory file |
| Produced by | `src/monitoring/reference.py`, run as the `reference` stage |

`src/config.py` exposes it as `REFERENCE_DATA_PATH`, overridable by environment variable on the
pattern [0024](0024-environment-based-tracking-uri.md) established.

## Alternatives considered

- **The held-out test split (`data/processed/test.parquet`)** — the intuitive answer, and the one
  proposed to us on the grounds that comparing against data the model optimised on "would
  invalidate the comparison by design". **Rejected on two counts.** First, that rule belongs to
  *performance* evaluation, not distributional drift: data drift asks whether production inputs
  have moved away from the distribution the model learned, so the learned distribution is exactly
  the right baseline. Second, and decisively, it is measurably the wrong *space* — see below.
- **A sample of the whole raw file**, as the reference material does. Rejected only narrowly: it
  is deterministic given a DVC-pinned CSV, so it is genuinely reproducible. But it mixes test rows
  into the baseline, and it re-reads 150 MB on every monitoring run to produce 5,000 rows.
- **Recomputing the reference per call** rather than persisting it. Rejected — this is the whole
  point. A baseline that is recomputed makes a changed verdict ambiguous between "reality moved"
  and "the yardstick moved". Freezing it gives the baseline a hash, so a drift number can be
  traced to the exact bytes that produced it.
- **A bare `dvc add`** on a hand-built file. Rejected: it would version the artifact but drop it
  out of the graph, so changing `monitoring.rows` or the raw CSV would leave a stale baseline that
  `dvc repro` has no reason to rebuild. The stage keeps it inside the DAG with everything else.
- **An extra output of the existing `preprocess` stage.** Rejected. `preprocess` writes the
  *scaled* splits the model trains on; this writes *raw* rows to compare against raw traffic —
  different artifacts for different consumers. Bolting it on would also make every
  monitoring-side parameter change invalidate the training data and force a retrain.
- **CSV, as the reference material's `pd.read_csv` implies.** Rejected in favour of parquet:
  dtypes survive the round trip exactly, which matters more than usual for an artifact whose only
  job is distributional fidelity.

## Justification

**The feature space is not a preference — it is forced, and it was measured.** The API logs
untransformed request payloads (`{"Amount": 149.62, "Time": 0.0, …}`), while `test.parquet` comes
out of `preprocess` with `Time` and `Amount` standardised:

```
test.parquet   Time   min/max/mean  -1.9980 / 1.6404 / -0.0075
test.parquet   Amount min/max/mean  -0.3517 / 51.1433 / 0.0035
```

Comparing that against raw traffic would report permanent, meaningless drift on exactly those two
columns and compare nothing on the other 28. The artifact this decision produces is in the same
space as the traffic:

```
reference.parquet  shape (5000, 30)   Class absent
                   Time   min/max/mean  22.00 / 172783.00 / 94894.91
                   Amount min/max/mean   0.00 /   4861.64 /    87.86
                   dtypes {float64}
```

**Re-deriving the training split is exact, not approximate.** `train_test_split` is deterministic
given the same `X`, `y`, `test_size`, `random_state` and `stratify`. `src/monitoring/reference.py`
imports `load_params` from `src.data.preprocess` rather than re-reading `params.yaml` itself, so
the two can never silently diverge: change `preprocess.test_size` and both the model's split and
the monitoring baseline move together. The split is re-derived rather than read from disk because
nothing in the repository persists the training rows in raw units — `preprocess.py` writes only
the transformed frames.

**Why 5,000 rows.** A baseline an order of magnitude larger than any realistic current window buys
no statistical power and costs time on every scheduled run. The figure is inherited from the
reference material; it lives in `params.yaml` and is declared under the stage's `params:`, so
changing it rebuilds the artifact rather than silently shifting the meaning of past verdicts.

## Trade-offs / consequences

- **Drift numbers are not comparable across a change to `monitoring.rows` or `random_state`.**
  Changing either produces a new baseline. DVC makes the change visible in `dvc.lock` rather than
  invisible, which is the point, but the discontinuity is real and anyone reading a history of
  drift scores has to know where it falls.
- **The baseline is static and will eventually be wrong.** A model retrained on newer data is no
  longer described by this reference, so after the loop starts promoting new versions the artifact
  has to be rebuilt or drift will be measured against a distribution nothing in production
  learned. This is deliberately not automated yet: rebuilding the reference on every retrain would
  make the system unable to detect slow, sustained drift, since the baseline would chase the
  traffic. Left as an explicit, documented decision for whoever operates the loop.
- **This deviates from `project_context/mlops_phase8.md` in three places** — rows, space and
  materialisation. The document's version would have run, and on this dataset the *rows* axis is
  nearly immaterial (train and test are one stratified split of one file). The *space* axis is
  not: its `load_reference()` reads the raw CSV, so it happens to be in the right space, which is
  why the document works and why the test-split proposal would not have.
- **`data/monitoring/` is a new top-level data directory.** DVC wrote `data/monitoring/.gitignore`
  containing `/reference.parquet` when the stage first ran; no root `.gitignore` change was needed.
- **The reference is not reproducible without the dataset.** Like everything else downstream of the
  local DVC remote ([0005](0005-dvc-local-remote.md)), a clean clone cannot rebuild it. Consistent
  with the rest of the pipeline, and the reason `tests/` never touches it.
