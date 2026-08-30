# Decision 44: Phase 9 closure — what the final tag claims, and what it does not

- **Date:** 2026-08-30
- **Status:** Accepted

## Context

Phase 9's Definition of Done is unlike the eight before it. Every prior phase ended in something
a machine could assert: a passing test, a green badge, a flow run with `deployment` provenance, a
version in the Registry. This one ends with *"a recruiter can understand the entire project in two
minutes using only the README and the video, without cloning anything"* — a judgement made by a
stranger, which nothing in this repository can verify.

`phase-9-complete` is also the tag that closes the whole nine-phase project, so it deserves the
same scepticism every phase closure has received here, applied to claims that are softer than
usual.

## Decision

**Record the video-length deviation rather than restate the criterion as met.** The DoD asks for a
**2–3 minute video starring the drift and retraining loop**. What exists is `docs/videos/intro.mp4`,
**20.01 seconds** (verified by parsing the MP4 `mvhd` and `mdhd` atoms), an introduction to what
the project is. The author chose this deliberately after the length was raised. It is recorded as
a deviation, not a pass.

**Do not self-certify the key test.** No claim is made here that a recruiter understands the
project in two minutes. What can be asserted is that the artifacts the test depends on exist,
resolve, and are accurate — nothing more.

**Ship the video in Git rather than on a third-party host,** and scope the large-file guard by path
rather than raising its ceiling.

**Close the roadmap in the README with an explicit statement of what the project is not.**

## Alternatives considered

- **Recording a full 2–3 minute walkthrough to satisfy the DoD literally.** The author's call, made
  after the gap was raised, and it is a defensible one: the loop already has a timestamped
  transcript in the README and a full account in [0040](0040-closed-loop-demonstration.md) and
  [0041](0041-phase-8-closure.md). Recorded as a deviation so the roadmap and the repository do not
  quietly disagree.
- **Marking the key test as passed.** Rejected. It is the one criterion in nine phases that this
  repository cannot evaluate about itself, and asserting it would be the exact failure mode every
  closure record here has avoided.
- **Raising `check-added-large-files --maxkb` to admit the video.** Rejected. That ceiling is what
  keeps datasets out of Git; widening it globally to admit one reviewed 3.4 MB file would silently
  permit the next one nobody looked at. `docs/videos/` is excluded by path instead.
- **Hosting the video on YouTube and linking it**, as the reference material suggests. Not chosen:
  in-repo keeps the artifact self-contained and free of a third-party account, and GitHub renders
  `.mp4` blobs with a player. The cost is ~3.5 MB in every clone, permanently.
- **Leaving the roadmap's Phase 9 row as "in progress" until a recruiter confirms the key test.**
  Rejected as unfalsifiable — it would mean the project could never be closed by anyone.

## Justification

### Definition of Done, item by item

| # | Criterion | Verdict |
|---|---|---|
| 1 | API deployed, public URL, `/docs` working | **Met.** `/health`, `/docs`, `/model-info` and `/predict` all 200 against the live URL, re-verified at closure. |
| 2 | Clean architecture diagram of the system and the loop | **Met.** Redrawn in `8a6a1d3`, checked label by label against the repository. |
| 3 | Definitive README, design decisions weighted | **Met.** Rewritten in `a0ba13c`; ten decisions each linking to its record. |
| 4 | 2–3 minute video starring the drift loop | **Deviation.** 20.01 s, an introduction. Deliberate. |
| 5 | A post sharing the story and reasoning | **Met on the author's word.** Published to LinkedIn; not an artifact this repository can verify. |
| 6 | Polished: clean clone, working links, careful writing | **Met, and measured.** See below. |
| 7 | The key test: understood in two minutes without cloning | **Not self-certifiable.** No claim made. |

### The clean-clone test, run as a stranger would

`git clone` over HTTPS with no credentials → **7.8 MB** → `make setup` → `make test` → **83
passed in 13.17 s**. No dataset, no `mlflow.db`, no DVC pull, no hidden steps. The clone also
surfaced something now documented in the README: `MODEL_PATH=deploy/model` serves real predictions
from a bare checkout, so a reviewer can run the API in one command without Kaggle credentials.

**All 21 relative links and 11 external links resolve.** No doubled words, no trailing whitespace,
consistent typography; the spell-checker's only flags were technical vocabulary.

### Two corrections this phase produced, both mine

**The cold start.** I measured a 17-minute idle window, saw 0.449 s, and wrote into both the README
and [0043](0043-render-deployment.md) that the service appeared never to sleep — and reasoned
onward that an always-warm instance would consume ~730 of the 750 free monthly hours. Later
measurements demolished it:

| Idle | First response |
|---|---|
| 17 minutes | 0.449 s — still warm |
| ~29 minutes | **191 s** |
| ~3.4 hours | **111 s** |

The service sleeps, the wake takes **two to three minutes**, and it is not proportional to how long
it slept. The README now leads its demo section with that wait, because a reviewer who hits an
unexplained three-minute hang concludes the link is dead — the worst available outcome for the
project's most visible artifact.

**A broken link, caught at closure.** The video was committed as `intro.mp4` after a rename, while
the README still pointed at `project_resume.mp4`. The Step 6 link check found it. Had the closure
skipped that check, the finished README would have shipped with its headline video link broken.

### State at closure

```
140 commits on main, 0 merge commits ever          9 tags, phase-0 … phase-8
ruff · ruff format · mypy --strict (25 files) · 83 tests · dvc: up to date
working tree clean · only main exists, locally and at origin · no stale refs
CI 33319683665 and CD 33319683640 both success on HEAD
Registry: 8 versions, @production -> v1, 7 unpromoted
Deployed image: ghcr.io/adrianmarchramon/fraud-mlops-pipeline:sha-6df9d154…, pinned in render.yaml
```

## Trade-offs / consequences

- **The video does not satisfy the criterion it was written for.** A 20-second introduction cannot
  star a loop; the README's timestamped transcript carries that weight instead. Anyone reading the
  roadmap as "all criteria met" should read this record.
- **`main`'s branch protection was never reconfirmed in Phase 9.** The `gh` token authenticates as
  a read-only account (`admin: false`), so the protection endpoint returns 404 — a permissions
  artifact, not evidence of absence. The indirect evidence is strong: 16 pull requests, zero merge
  commits, and [0027](0027-branch-protection-boundary.md) recording PR #1 blocked with a red CI.
  It remains unverified for this phase, and a browser check would close it.
- **~3.5 MB of video now lives in Git permanently.** Every re-record adds its full weight again.
- **The public demo will be slow for most first-time visitors**, because most will arrive after the
  service has spun down. Documented rather than paid for; the fix is $7/month.
- **The deployed model is frozen** ([0042](0042-bundled-model.md)), and a promotion reaches the
  public URL only through a re-export, a commit and a redeploy.
- **Everything Phase 8 left open remains open**: the webhook has never reached a real endpoint, the
  drift share never reaches the Prefect dashboard, and no retrained model has ever been promoted
  because retraining on unchanged data ties the incumbent.

## What the tag claims

`phase-9-complete` asserts that this repository contains a working, documented, publicly reachable
machine-learning system, built in nine gated phases, whose every design decision is written down —
44 records — and whose drift-to-retrain loop has been observed running end to end without human
intervention.

It does **not** assert that the system has real users, receives real labels, has ever promoted a
retrained model, or is reproducible by a third party without a Kaggle account. It does not assert
that the model is good; the model was never the point. And it does not assert the key test — that
a stranger understands all this in two minutes — because that is the one claim only a stranger can
make.

**With this tag the project is finished.**
