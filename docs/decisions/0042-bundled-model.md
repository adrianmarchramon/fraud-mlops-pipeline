# Decision 42: The public deployment carries its model inside the image

- **Date:** 2026-08-30
- **Status:** Accepted (supersedes, in scope, one rejected alternative of [0022](0022-container-image-contract.md))

## Context

Phase 9 puts the API on a public URL. Every phase since 3 has resolved the model the same way —
`models:/fraud-detector@production`, an alias against a live MLflow server — and a free-tier
service has no such server. `project_context/mlops_phase9.md` §"Step 1" anticipates this and
recommends bundling the model into the image.

Two facts measured in this repository turn that from a preference into a constraint, and one of
them contradicts the reference material's own suggestion:

**Nothing model-shaped is versioned anywhere.** `dvc.yaml`'s `train` stage declares only
`metrics: reports/metrics.json (cache: false)` — no model output. The DVC-tracked artifacts are
the raw CSV, the three `data/processed/` files and `data/monitoring/reference.parquet`.
`.gitignore` excludes `mlruns/`, `mlflow.db` and `models/`. So the trained model exists in exactly
one place: a git-ignored, DVC-untracked store on the author's machine.

That is what makes the reference material's two requirements — *"the image it deploys is the same
one your CI already builds and publishes in Phase 6"* **and** *"include the registered model
inside the image"* — mutually unsatisfiable as written. A GitHub runner has no dataset (the DVC
remote is local, [0005](0005-dvc-local-remote.md)) and no Registry. Neither does Render, which
builds from the same repository. Anything the image is to contain must already be in Git.

**The MLflow store is not relocatable.** The obvious approach — copy `mlflow.db` and `mlruns/`
into the image and point `MLFLOW_TRACKING_URI` at them — fails, because the store records absolute
host paths in four columns:

```
experiments.artifact_location    = /home/amr/…/mlruns/1
runs.artifact_uri                = /home/amr/…/mlruns/1/<run_id>/artifacts
logged_models.artifact_location  = /home/amr/…/mlruns/1/models/m-58af4cad…/artifacts
model_versions.storage_location  = /home/amr/…/mlruns/1/models/m-58af4cad…/artifacts
```

Copied to `/app/mlruns`, the registry rows resolve and the artifacts are then chased at a path
that does not exist in the container.

## Decision

**Export the `@production` artifact into `deploy/model/` and commit it** — 38 files, 1.4 MB,
byte-identical to the registry's copy but for two stray `.pyc` files. `scripts/export_model.py`
produces it and `make export-model` runs it; the export is never done by hand.

**Add one environment-conditioned branch, not a second code path.** `src/config.py` gains
`MODEL_PATH = os.getenv("MODEL_PATH", "")`. Empty — every local run, every Compose run, every
test — resolves the alias exactly as before. Set, `load_bundled()` reads the directory and no
Registry is contacted. This is the shape [0024](0024-environment-based-tracking-uri.md)
established and explicitly nominated for this moment: *"the pattern is now the precedent …
which is exactly what Phase 6's CI and Phase 9's deployment target will need."*

**Read the version from the bundle, never from a constant.** MLflow writes `registered_model_meta`
(`model_name`, `model_version`) inside every registered artifact, so `/model-info` keeps answering
truthfully with nothing to ask. A bundle naming a different registered model is refused at startup.

**Keep ruff and the rewriting pre-commit hooks off `deploy/model/`.** The bundle carries MLflow's
own snapshot of `src/`, which belongs to the frozen model rather than to the working tree.

## Alternatives considered

- **Copying the MLflow store and rewriting the four absolute-path columns at build time.** The
  original recommendation, and the one this audit killed: it means a scripted `UPDATE` against
  MLflow's private schema, which no upstream contract protects, to gain an alias indirection that
  is meaningless when only one version is present.
- **Recreating the store at the identical absolute path inside the image**
  (`/home/amr/Documentos/…`). Works. Rejected on sight.
- **Downloading the artifact during the Docker build.** Rejected: the builder is a GitHub runner
  with neither dataset nor Registry, which is the whole problem.
- **Letting Render build from `docker/Dockerfile`**, as the reference `render.yaml` shows.
  Rejected twice over — it hits the same missing artifact, and it would produce a second image
  nobody verified. See [0043](0043-render-deployment.md).
- **Running a public MLflow server for the demo to point at.** Rejected: a second always-on
  service with its own storage and its own failure modes, to serve one model that never changes.
- **A hardcoded version constant in the image.** Rejected, and then proven necessary to reject —
  see below.
- **Keeping the artifact out of Git and building the deployed image locally.** Rejected: the
  deployed image would no longer be the one CI verified, forfeiting the strongest property Phase 6
  earned.

## Justification

**The offline load was proven before a line of this was written.** The exported artifact was
loaded from a directory outside the repository, with `cwd` outside it, `PYTHONPATH` empty and
`MLFLOW_TRACKING_URI=http://127.0.0.1:1`. An instrumented `joblib.load` recorded exactly one call,
inside the export directory. MLflow follows the **relative** `path` in `MLmodel`, never the
absolute `uri` recorded beside it — so the absolute-path disease that killed the store-copying
approach does not touch this one. The prediction matched the registry-loaded artifact to the last
digit: `4.2796866182470694e-05`.

**The container was then verified as a controlled experiment**, one image, one variable:

| | `MODEL_PATH=/app/deploy/model` | unset |
|---|---|---|
| `/health` | `{"status":"ok"}` | `{"status":"no_model"}` |
| `/model-info` | `200`, version `1` | `503` |
| `/predict` | `200`, `0.000042796866182470694` | `503` |
| Docker `HEALTHCHECK` | `healthy` | not healthy |

The bundled container answered with **no MLflow reachable at any address**. Resident memory after
serving all three endpoints: **199 MiB**.

**A mutation test caught a real gap in the first version of these tests.** Three mutations were
applied: inverting the branch (caught), removing the model-name check (caught), and replacing the
metadata read with a literal `version = "1"` — **not caught**, because the committed bundle *is*
version 1, so every assertion still passed. That is precisely the failure the metadata file exists
to prevent: an image shipping a stale export while announcing whatever number was last typed into
the code. `test_the_reported_version_comes_from_the_bundle_not_from_the_code` was added, and the
mutation now fails.

**`.gitignore` was silently mutilating the bundle.** `models/`, written in Phase 2 for the local
`models/` output directory, is unanchored and therefore matches at any depth — including
`deploy/model/code/src/models/`. Four real files of the artifact were being excluded,
`register.py` among them, which is where the `FraudModel` class the bundle unpickles is defined.
The rule is now `/models/`. Verified in both directions: `models/confusion_matrix.png` is still
ignored, and every one of the 38 exported files is now tracked.

**`end-of-file-fixer` was rewriting the artifact.** Run over the bundle it reported *"Fixing"* on
seven files — `requirements.txt` and the input-example JSONs — silently editing a model bundle
whose entire value is being identical to the registered version. Both rewriting hooks now exclude
`deploy/model/`; `check-yaml` and `check-added-large-files` only read, so they stay.

**The bundle is now stable and complete**: 38 files on disk, 38 tracked, none ignored, and a full
`pre-commit run` over it leaves the checksum unchanged.

## Trade-offs / consequences

- **The public model is frozen at build time.** Promoting a new version changes nothing on the
  deployed service until someone runs `make export-model`, commits, and redeploys. This is the
  exact cost [0022](0022-container-image-contract.md) named when it rejected baking the model in,
  and that rejection still stands **for the Compose topology**, which is untouched: `make serve`
  and `docker compose up` still resolve the alias, so a promotion still lands there with no
  rebuild. What is superseded is the claim that bundling is wrong *everywhere* — on a platform
  with no Registry to resolve against, the alias indirection has nothing to indirect through.
- **A binary artifact now lives in Git**, a first for this repository. It is not data — the rule
  in `CLAUDE.md` is about datasets — but it is 1.4 MB of pickles that `git log` will carry
  forever, and each re-export writes a new copy. The largest single file, MLflow's bundled
  `uv.lock` at 728 KB, sits under `check-added-large-files --maxkb=1024` with 29% headroom; an
  export that grew past 1 MB in any one file would trip the hook.
- **Two loading paths now exist**, and only one is exercised by any given run. The bundled path is
  covered by tests that load the real committed artifact, which is what keeps it from rotting.
- **The bundle contains a snapshot copy of `src/`** that is deliberately excluded from ruff. A
  reader who opens `deploy/model/code/src/register.py` is reading the model, not the codebase, and
  editing it there does nothing.
- **`/model-info` on the deployed service reports the bundled version, not the Registry's.** If
  `@production` moves and the export does not, the public API will honestly report the older
  version it is actually serving — which is the correct answer, and only confusing if someone
  expects the two to be the same system. They are not.
