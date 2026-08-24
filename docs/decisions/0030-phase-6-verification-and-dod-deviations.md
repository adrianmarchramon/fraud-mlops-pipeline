# Decision 30: Phase 6 verification strategy, interaction granularity, and DoD deviations

- **Date:** 2026-08-24
- **Status:** Accepted

## Context

Phase 6's Definition of Done ends with what the roadmap calls "the key test": *"You push, see the
checks running on GitHub, and if everything passes, the image builds and publishes on its own."*
Unlike every prior phase, that criterion cannot be demonstrated locally at all — it lives
entirely on GitHub's platform. This record fixes how it was verified, how the phase was split
across interactions, and the two places where the phase's literal wording was not met.

## Decision

**Verify the key test with a real change, not a disposable one.** The end-to-end cycle was
demonstrated by PR #2, which carried the actual Step 5–6 deliverables (`cd.yml` and the README
badge) rather than a throwaway commit created only to watch the pipeline run.

**Verify the gates by making them fail first.** Both blocking behaviours were demonstrated red
before being accepted green:

| Gate | Red demonstration | Green demonstration |
|---|---|---|
| Model quality gate | PR #1, `pr_auc` forced to `0.61` → CI `failure` | PR #2 → CI `success` |
| Branch protection | PR #1 `mergeStateStatus: BLOCKED`, `merged=null` | PR #2 merged 14:29:10Z |
| CD trigger | — | run `32738991528` `success`, `headSha` = PR #2's merge commit |

**Split the phase across three interactions**: Steps 1-2-4 (CI + the gate), then Step 3 + Steps
5-6 (protection, CD, badge), then this closure. The original pre-phase recommendation was two.

**Re-query all evidence at closure rather than cite it.** Every URL, conclusion and tag in this
record was fetched live during the closure interaction, not carried forward from the interaction
that produced it.

**Merge by rebase, never squash.**

## Alternatives considered

- **A disposable test PR to demonstrate the pipeline** — rejected. It would prove the same
  mechanism while adding a merge to `main` that ships nothing, and the real deliverable would
  have needed its own second cycle anyway.
- **Squash merging** — rejected. Each closure PR carries two genuinely distinct logical changes,
  and squashing collapses them into one commit that does two things — the opposite of the atomic
  standard the squash was supposed to serve. Rebase preserves both *and* keeps `main` linear,
  which matters because this repository has never contained a merge commit.
- **A full re-run of the E2E cycle at closure**, as Phase 5's closure re-ran its reproducibility
  check — rejected as unnecessary here. Phase 5's evidence lived in a local Docker volume that
  could have been lost or mutated between interactions; Phase 6's lives on GitHub, is immutable,
  and is queryable again at any time. A live re-query is equivalent and cheaper. (The closure PR
  itself produces a fresh cycle regardless, as a side effect of shipping this record.)
- **Two interactions instead of three** — the pre-phase default. Overtaken by events: Step 3 is a
  browser action that cannot be scripted from this machine, which made a natural boundary the
  original estimate had not anticipated.

## Justification

A gate observed only in its passing state is not verified — it is assumed. PR #1 exists solely to
prove the two blocking behaviours, and it was closed unmerged (`merged=null`) so the degraded
metric never touched `main`. That negative evidence is what upgrades "the pipeline is green" from
an observation to a property.

## Trade-offs / consequences

### Two places where the DoD's literal wording is not met

Recording both rather than quietly scoring them as passes:

- **"`ci.yml` runs … on every push and pull request."** Ours runs on `pull_request` and on
  `push` to `main` only, so a push to a feature branch with no open pull request runs nothing.
  Judged **met in substance**: the reference's own stated purpose — verify quality on every
  proposed change — is fully served, and the literal form double-runs on any push to a branch
  with an open PR. Rationale in [0026](0026-ci-workflow-shape.md).
- **"Nothing in `src/` altered by this phase."** False as written. Phase 6 modified
  `src/data/preprocess.py` (+5/−1) and `src/models/train.py` (+23/−5) — the four fixes required
  to make `mypy --strict` pass, one of which removed a latent `AttributeError`. Judged **met in
  substance**: no regression occurred (49 tests green, `ruff` clean, `mypy` clean, Registry
  intact at `@production` → v1), the change was explicitly authorised, and it was a net
  improvement. `docker/`, `dvc.yaml`, `src/api/`, `params.yaml` and the Model Registry are
  genuinely untouched.

### Other consequences

- **This record dates quickly.** The run IDs and the `sha-` tag it cites are pinned to
  2026-08-24; the next merge to `main` republishes `main` and `latest`. Read the tags as "what
  was published when Phase 6 closed", never as "what is current".
- **The verification is not reproducible by a third party.** Re-running the key test requires
  write access to this repository. A clone can run `ci.yml`'s checks locally, but cannot
  demonstrate protection or publication. This is inherent to the phase, not a gap in the method.
- **Phase 7 inherits a hard constraint**: every future change reaches `main` through a pull
  request, and orchestration work will land the same way.
