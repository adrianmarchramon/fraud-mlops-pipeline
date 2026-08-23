# Decision 25: How Phase 5 is verified, and the honest scope of the "key test"

- **Date:** 2026-08-23
- **Status:** Accepted

## Context

`project_context/mlops_phase5.md` §"Verification: The Definition of Done" ends with the item it
calls the most revealing:

> **The key test:** someone without Python installed can clone your project and run it entirely
> using only Docker (`docker compose up`).

The `phase-5-complete` tag is a public claim that this holds. It does not, as literally worded,
and the reason is structural rather than a defect to be patched.

Compose namespaces named volumes by project name, which it derives from the **compose file's
directory** — not the repository root. Because this project keeps its compose file in `docker/`,
the project name is always `docker`, and the artifacts are `docker_mlflow-data`, `docker-api` and
`docker_default` **regardless of what the checkout is called**. Two clones of this repository on
one machine therefore share Docker state silently; verified in a clone at
`/tmp/mlops-fraud-pipeline-phase5-test`, where `docker compose config` still resolved
`docker_mlflow-data`.

A registry that genuinely starts empty therefore has to be requested explicitly. When it is, the
Model Registry inside it holds nothing, `lifespan` finds no `@production` alias, and by ADR 0020
the API comes up anyway and reports `no_model` instead of crashing. Seeding that registry means running
`train` and `register`, which requires Python and uv on the host. The gap is unavoidable given
ADR 0015: the model is deliberately *not* baked into the image.

## Decision

**The Phase 5 reproducibility check runs in a throwaway clone, not in place.** A clone under
`/tmp`, `make setup`, `dvc pull`, `dvc repro`, `dvc status` for the inherited level; then, within
that same clone and depending on no pre-existing Docker state, the full Phase 5 cycle: build the
image from the committed `Dockerfile`, bring up `mlflow` alone, populate its registry by
retraining and registering against it, bring up the whole system, and serve real predictions
over HTTP.

**The check runs under an explicit `-p` project name** (`docker compose -p mlops-phase5-check
…`). This is not a preference: without it the clone inherits the project name `docker` and would
mount the working environment's populated volume, which would both invalidate the "empty
registry" premise and put the only copy of the phase's registry at risk.

The clone's own namespaced volume (`mlops-phase5-check_mlflow-data`) and image
(`mlops-phase5-check-api`) are deleted afterwards. **The working environment's
`docker_mlflow-data` is never touched** — this verification is deliberately redundant with it,
not a replacement for it.

**The key test is recorded as met in substance and not literally.** What Phase 5 genuinely
delivers is that *the running system* is entirely containerized and behaves identically on any
machine with Docker. What it does not deliver is a zero-Python path from a virgin clone to a
served prediction, because the registry must be seeded once. The README states this plainly
rather than repeating the roadmap's slogan.

## Alternatives considered

- **Running the check in place, reusing the already-populated volume** — rejected as the primary
  strategy. It demonstrates that the teardown is reversible, which is worth knowing and was
  verified separately, but it reuses the very artifacts under test: it would prove nothing about
  whether the committed `Dockerfile` still builds, or whether an empty registry can be populated
  from scratch.
- **Adding a `trainer` service or Compose profile that seeds the registry in a container** —
  rejected *for this phase*, not on merit. It would make the literal claim true, and it is the
  natural way to close the gap. It also changes `docker-compose.yml`, introduces a container that
  needs the dataset, and designs orchestration — which is Phase 7's subject. Deferred rather
  than dismissed.
- **Committing a seed model artifact so a clone starts populated** — rejected. It contradicts
  ADR 0015 (the model is pulled by alias, never baked in) and the project's rule that artifacts
  never enter Git.
- **Declaring Phase 5 not closed until the gap is closed** — rejected. The phase's stated
  objective is "package everything so that it runs identically on any machine", and that is met
  and demonstrated. Holding the phase open over a slogan would misrepresent where the project
  actually stands.
- **Repeating the claim as written and staying quiet about the caveat** — rejected. A README that
  overstates what the system does is the specific failure this project's decision log exists to
  prevent.

## Justification

The inherited level exists because Phase 5 touched `src/config.py`, which is a declared
dependency of the `validate`, `preprocess` and `train` DVC stages; re-running the pipeline in a
clean clone is what proves that change did not disturb reproducibility.

The Phase 5 level exists because two of the Definition of Done's items — *"layer caching is
correctly ordered"* and *"you can build the image, populate the registry, and spin up the system"*
— cannot be evidenced by reusing an image that was already built and a volume that was already
populated. Only a build from scratch, against an empty registry, tests them.

Separating the throwaway clone's Docker resources from the working environment's is not
bookkeeping. The working volume is the only copy of the registry populated during this phase, and
recreating it costs a full retrain and re-registration; the clone's namespacing is what makes
`docker volume rm` safe to run at all.

## Trade-offs / consequences

- **The check is slow and disk-hungry.** A full image build plus a retrain and registration
  against an empty registry, and a second image and volume that exist only to be deleted.
- **The README now carries a caveat** that the roadmap's phrasing does not. It is less impressive
  and more accurate.
- **The gap remains open until a later phase closes it.** Phase 6's CI or Phase 7's orchestration
  are the natural places; until then, "clone and `docker compose up`" gives a healthy MLflow and
  an API honestly reporting `no_model`.
- **Two independent registries now exist** — the host's `mlflow.db` and the container's volume —
  with no synchronisation between them. That is correct and intentional, but it means "which
  registry?" is a question every future phase has to answer explicitly.
- **Any second checkout of this repository on the same machine collides by default.** Phase 6's CI
  and anyone testing a branch alongside `main` must pass `-p` or set `COMPOSE_PROJECT_NAME`, or
  they will silently share — and can destroy — the registry volume of the other checkout.
