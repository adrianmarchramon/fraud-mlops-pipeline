# Decision 43: Render, the free plan, and deploying the image CI already published

- **Date:** 2026-08-30
- **Status:** Accepted

## Context

Phase 9's first half is *making it real*: a public URL anyone can send a transaction to.
`project_context/mlops_phase9.md` surveys Render, Railway, Fly.io, Modal and Cloud Run and
recommends Render for its permanent free tier and direct Docker support. The project's
no-Kubernetes stance ([0004](0004-stack-summary.md)) already rules out the heavyweight options.

The material also contains an internal contradiction worth naming rather than quietly resolving.
Its prose says *"the image it deploys is the same one your CI already builds and publishes in
Phase 6, closing the loop between containerization, automation, and deployment"*, while the
`render.yaml` it prints does the opposite — `dockerfilePath: ./docker/Dockerfile` and
`dockerContext: .`, which makes Render build a **new** image on its own infrastructure and never
mentions GHCR at all.

[0029](0029-cd-workflow-and-ghcr-publication.md) had already left an instruction for this moment:
*"`latest` is mutable. Phase 9 should deploy the `sha-` tag, not `latest`, or it will not know
which build is live."*

## Decision

**Render, `plan: free`.**

**Deploy the prebuilt GHCR image; build nothing.** `runtime: image` with an `image.url` pinned to
an immutable `sha-` tag. No `creds:` — the package is public.

**Health check on `/model-info`, not `/health`.**

**Tell Render the port rather than override the command.** `PORT: 8000` as an environment
variable, because Render routes to `$PORT` (default `10000`) while the image's `CMD` binds 8000.

**Deploy the API alone.** MLflow, Prefect, the monitoring flow and the retraining loop stay local.

**`render.yaml` ships with a deliberately invalid image tag** (`sha-REPLACE_WITH_MERGE_COMMIT_SHA`).

## Alternatives considered

- **Letting Render build from `docker/Dockerfile`**, as the material's own example does. Rejected
  on two independent grounds. It would produce a second image that no CI run ever verified, from
  the same source but a different builder — discarding the property Phase 6 exists to establish.
  And it would not work anyway: Render builds from this repository, so it faces exactly the
  missing-artifact problem [0042](0042-bundled-model.md) documents.
- **Deploying `:latest` or `:main`.** Rejected on 0029's standing instruction. Both move on every
  merge, so "which build is live?" would be unanswerable after the fact, and a redeploy could
  silently ship a different image than the one that was tested.
- **`healthCheckPath: /health`**, the obvious choice and the one the reference material shows.
  Rejected for the reason [0022](0022-container-image-contract.md) rejected a bare `curl -f`:
  `/health` answers `200` with `{"status":"no_model"}` by design, so a platform that reads only
  the status code would call a modelless container perfectly healthy. The Dockerfile solves this
  by matching the body; Render offers no such option, so the check points instead at the endpoint
  whose **status code** already carries the answer — `/model-info` returns `503` until a model is
  loaded. Measured on both containers: the bundled one `200`, the modelless one `503`.
- **Overriding the container command with `dockerCommand`** to bind `$PORT` directly. Rejected:
  the deployed process would no longer be the one `docker/Dockerfile` declares and CI exercised,
  for no gain over naming the port.
- **The paid instance (~$7/month) to avoid cold starts.** Rejected for a portfolio demo. Revisit
  if the cold start proves to cost more than it saves.
- **Railway, Fly.io, Modal, Cloud Run.** Railway has no permanent free tier; Fly.io requires a
  credit card; Modal is serverless-ML-shaped and would mean a second serving path beside the
  FastAPI app; Cloud Run is credible but drags in a GCP project and IAM for no benefit here.
- **Deploying the whole system.** Rejected — it is the mistake the phase document warns about.
  The public URL proves the API is real; the video proves the system is.

## Justification

**The free plan fits, measured rather than hoped.** Render's free instance type is 0.1 CPU and
**512 MB**. The container serving `/health`, `/model-info` and `/predict` from the bundled model
sat at **199 MiB** — under 40% of the cap. The earlier bare-process figures bracket it: 340 MB
peak RSS loading through the Registry, 297 MB loading the bundle.

**The image is genuinely public**, verified anonymously: a GHCR pull token with no credentials
fetches the manifest for `latest` and for `sha-155416ba…` with `HTTP 200`. Nothing has to be
provisioned in Render for it to pull.

**The `sha-` tag is real and per-commit.** The tag list carries one for every merge, including
`sha-155416bac5e2cec9d58e772e64456a295b75dd7f` for the current `HEAD`, so any deployed build is
traceable to a commit.

**The placeholder is deliberate, not unfinished work.** The image that will serve this deployment
is built by CD *from the commit that merges `render.yaml`*, which does not exist while
`render.yaml` is being written. Any earlier `sha-` names an image built before `deploy/model/`
existed — it would start, find no bundle, and serve `503`s while looking deployable. An invalid
tag fails at the registry with a message that says so.

## Trade-offs / consequences

- **The documented cold start has not been observed, and that may cost more than it saves.**
  Render documents a **15-minute** idle spin-down and roughly a minute to wake. Measured once
  against the live service, the first request after a **17-minute** window with no traffic from
  this machine returned in **0.449 s** — no wake-up at all, against 0.300 s warm. The most likely
  explanation is that the `healthCheckPath` Render polls counts as inbound traffic and keeps the
  instance alive.
  One 17-minute window does not prove the service never sleeps, and the README says so rather than
  claiming either way. But if it genuinely never sleeps, the consequence is the row below: an
  always-on free service consumes **~730 of the 750** monthly instance hours, so the allowance is
  effectively spent by this one service and a second free service would exceed it. That is worth
  knowing before assuming the free tier scales to a second demo.
- **750 free instance hours per month, per workspace** — see above; a service kept warm by its own
  health check uses very nearly all of them.
- **Every redeploy needs the pin updated.** Changing the deployed build means editing
  `image.url` to a new `sha-` tag and committing — deliberate friction, and the price of knowing
  what is live.
- **A promotion does not reach the public URL on its own.** It needs `make export-model`, a
  commit, a merge, and a re-pin. See [0042](0042-bundled-model.md).
- **The blueprint is written but not applied.** No Render account, service or deployment exists as
  of this record; nothing here has been confirmed against a running service. That is Step 1b, and
  until it happens this file describes an intent, not an observation — the same distinction
  [0035](0035-phase-7-live-verification.md) drew between wiring and live verification.
- **The deployed service logs predictions to a disk nobody reads.** `log_prediction()` still
  appends every request to `logs/predictions.jsonl` inside the container, and Render's filesystem
  is ephemeral, so that file dies with each spin-down and nothing ever analyses it. The drift loop
  reads the log on the machine where MLflow, Prefect and the dataset live. This is the intended
  split — the public URL is the API half of the system, not the loop — but the README must say so
  plainly, because "detects drift and retrains itself" beside a public link invites exactly the
  wrong inference. The deployed API does not retrain; the system does, where it can.
- **The health check makes Registry-independence load-bearing.** Because `/model-info` gates
  Render's view of health, a deployment whose bundle failed to load will be reported unhealthy and
  restarted rather than left serving `503`s quietly. That is the intent, but it also means a
  broken export shows up as a crash-looping service rather than a degraded one.
