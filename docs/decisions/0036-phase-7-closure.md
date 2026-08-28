# Decision 36: Phase 7 closure — how the DoD was verified, and what is deliberately unfinished

- **Date:** 2026-08-28
- **Status:** Accepted

## Context

Phase 7's Definition of Done ends with a criterion unlike every phase before it: *"You run the
training pipeline with a command and see each stage execute in the dashboard, with automatic
retries if a task fails."* Phases 1–4 and 6 closed on assertions a machine re-runs — `pytest`,
`dvc repro`, an HTTP body, a red CI blocking a merge. This one closes on **observation**, which
cannot be asserted and cannot be re-derived by a third party. This record fixes how it was
verified at closure, and names the two things that are deliberately not finished.

## Decision

**Verify by re-querying the live Prefect history rather than re-running the pipeline.** The
Steps 5–6 runs were re-read from `~/.prefect/prefect.db` during this closure, not cited from the
interaction that produced them.

**Do not re-fire any trigger at closure.** No new flow run, no new deployment trigger, no second
CSV rename, no seventh Registry version.

**Correct [0035](0035-phase-7-live-verification.md) rather than supersede it**, and correct the
same claim everywhere it had propagated — `Makefile`, `pipelines/serve.py`,
`pipelines/training_pipeline.py`.

**State in the README that the loop is wired but not armed**, naming `detect_drift()` and Phase 8
explicitly.

**Add no `.gitignore` entry for Prefect.** `PREFECT_HOME` defaults to `~/.prefect`, outside the
repository, and nothing here overrides it — verified again at closure: the only Prefect artifacts
are `~/.prefect/{prefect.db,memo_store.toml,storage}`.

## Alternatives considered

- **A full live repetition of Step 6 at closure**, as Phase 5's closure re-ran its reproducibility
  check. Rejected **because the precondition Phase 5 faced did not hold here.** Phase 5's evidence
  lived in a Docker volume that could have been lost or mutated between interactions, so it had to
  be re-derived. Phase 7's evidence was audited first and found intact, with the retry cycle,
  task `run_count=4`, both deployments and all seven runs still readable. Re-running would have
  produced a second identical proof plus a seventh unpromoted Registry version.
- **Re-firing the event-driven trigger** to reconfirm the closed loop — rejected on the same
  ground: `favorite-coyote` is in the history with `deployment` provenance and four completed
  tasks, created by `run_deployment()` from inside `horned-phoenix`.
- **Superseding 0035 with a new record** instead of correcting it — rejected. The false statement
  would remain in the record that made it, and this project already has the opposite precedent
  (`9f7cb1f`, correcting the Compose project-name claim in ADR 0025 in place).
- **Reverting the `PREFECT_API_URL` fix** once its stated justification turned out to be wrong —
  rejected. The fix is still correct; only the severity was overstated. See below.
- **Claiming the "visible on the dashboard" DoD item as literally met** — rejected. See deviations.

## Justification

**Evidence re-queried at closure, not carried forward:**

```
dexterous-porpoise    training-pipeline    FAILED     direct      tasks=1  max_run_count=4
audacious-cockatrice  training-pipeline    COMPLETED  direct      tasks=4  max_run_count=1
solid-koel            training-pipeline    COMPLETED  deployment  tasks=4  max_run_count=1
wild-jacamar          monitoring-pipeline  COMPLETED  direct      tasks=1
horned-phoenix        monitoring-pipeline  COMPLETED  direct      tasks=2
favorite-coyote       training-pipeline    COMPLETED  deployment  tasks=4  max_run_count=1
```

`max_run_count=4` on the failed run is one attempt plus exactly `retries=3`, and the log lines
remain readable — `Retry 1/3`, `2/3`, `3/3`, `Retries are exhausted`, ten seconds apart, on
`src.exceptions.DataIngestionError`. `prefect deployment ls` still returns both
`monitoring-pipeline/daily` and `training-pipeline/on-demand`.

**All three trigger mechanisms are represented in that table**: `direct` invocation,
`deployment`-sourced runs from the manual trigger and from the drift event, and the cron schedule
that produced the pre-computed monitoring runs.

**The Registry is unchanged since Steps 5–6** — `@production` → v1, six versions, v4–v6 timestamped
16:40–16:43 — which is what makes the closure a verification rather than a new experiment.

## Trade-offs / consequences

### One DoD item met by substitution, not literally

*"Flows and their runs are visible on the Prefect dashboard."* The dashboard was confirmed serving
(`/api/health` and `/` both 200), and every fact it renders was read from the same database
through the API. But the visual confirmation itself — a person watching states transition in a
browser — is evidence this environment cannot capture. Recorded as **met in substance, not
literally**, on the same footing as Phase 6's two documented deviations
([0030](0030-phase-6-verification-and-dod-deviations.md)) rather than scored as a clean pass.

### One deliverable deliberately unfinished

`detect_drift()` returns a constant. The monitoring flow can therefore only ever take its "no
drift" branch in production, and the event-driven trigger was demonstrated by patching the
predicate in memory. **This is the phase boundary, not a gap**: Phase 7's deliverable is the
wiring, Phase 8's is the signal. It is stated in the module docstring, in
[0033](0033-orchestration-design.md), and now in the README, so no reader can mistake "closed
loop" for "drift detection is live".

### The correction to 0035, and what it cost

0035 claimed a run without `PREFECT_API_URL` was lost with its database. Re-querying at closure
disproved it: `excellent-chipmunk`, the run that produced that claim, is still present with its
task run and six log lines. The error was reading `Stopping temporary server` as data loss;
Prefect's ephemeral server is a temporary API **process** over the same `~/.prefect/prefect.db`.

The `Makefile` fix survives the correction because its real justification survives: without the
variable a run is not observable **live**, each invocation pays server startup, two API servers
write one SQLite file, and `serve.py` registers deployments nothing is serving. Corrected in
0035 in place and in all three files that had repeated it.

The lesson is narrower than "verify claims": the failing run *was* verified. What was not verified
was the **inference** drawn from it — that a stopped server implies discarded data. A live run
produces a lot of output, and the tempting reading is not always the true one.

### Other consequences

- **This verification is not reproducible by a third party.** It needs the dataset (local DVC
  remote, [0005](0005-dvc-local-remote.md)), a live MLflow, and a local Prefect server. Unlike
  Phase 6's, whose evidence lives on GitHub and is queryable by anyone with access.
- **The evidence is one unbacked SQLite file.** Everything material has been quoted into 0035 and
  this record, so losing `~/.prefect/prefect.db` costs the ability to re-query, not the record.
- **Three unpromoted Registry versions (v4–v6) are permanent.** They are the honest cost of
  demonstrating three trigger mechanisms, since each necessarily runs the real pipeline.
- **Phase 8 inherits a hard contract**: replace the body of `detect_drift()`, keep the signature,
  and change nothing in `pipelines/`.
