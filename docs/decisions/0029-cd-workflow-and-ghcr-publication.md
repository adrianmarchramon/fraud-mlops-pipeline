# Decision 29: CD workflow, image tags, and public GHCR publication

- **Date:** 2026-08-24
- **Status:** Accepted

## Context

Phase 5 produced an image contract ([0022](0022-container-image-contract.md)) that was only ever
built by hand. Phase 6 automates it. Three decisions had real alternatives: what triggers the
build, what the published image is called, and who may pull it.

## Decision

**`.github/workflows/cd.yml`, triggered on `push: branches: [main]` and nothing else.**

**It re-runs none of CI's checks** — no `ruff`, no `mypy`, no `pytest`.

**Permissions are declared explicitly and narrowly:**

```yaml
permissions:
  contents: read
  packages: write
```

**Authentication is `GITHUB_TOKEN`** via `docker/login-action`, with no stored secret.

**It builds `docker/Dockerfile` with `context: .`**, reproducing exactly the pairing
`docker/docker-compose.yml` already uses. Not one line of the Phase 5 Dockerfile is rewritten.

**Tags are declared explicitly** rather than left to `metadata-action`'s defaults:

```yaml
type=ref,event=branch                              # main
type=sha,format=long                               # sha-<40 chars>
type=raw,value=latest,enable={{is_default_branch}} # latest
```

**The GHCR package is public.**

**`concurrency` without `cancel-in-progress`** — the opposite of `ci.yml`.

## Alternatives considered

- **Also triggering on `pull_request`** — rejected. It would build and push a 2.3 GB image for
  every intermediate commit of every proposal, including ones never merged, turning "published
  image" from a record of decisions into a record of activity.
- **Repeating the CI checks inside `cd.yml`** — rejected as theatre. The only commits reaching
  `main` are those a protected pull request let through, so the checks would re-derive a verdict
  GitHub has already recorded against that exact commit.
- **Relying on `metadata-action`'s default tags** — rejected once measured. For a branch push the
  defaults emit **only `main`**: a single mutable tag with no way to pin a build. (The reference
  material's claim that the defaults include a commit-hash tag is simply incorrect.) Phase 9 will
  need to deploy a known-good image, and `sha-<commit>` is that handle.
- **A private package** — rejected. See below.
- **`cancel-in-progress: true`**, mirroring `ci.yml` — rejected: cancelling here would abort a
  half-finished registry push. In CI a superseded verdict is worthless; in CD an interrupted
  publish is harmful.
- **Pinning actions by commit SHA** rather than exact tag — the stricter supply-chain posture,
  but it renders the workflow unreadable for a portfolio reviewer, and this repository publishes
  nothing a compromised action could exfiltrate. Exact version tags were chosen instead, all
  verified current at the time of writing: `actions/checkout@v7.0.1`,
  `docker/setup-buildx-action@v4.3.0`, `docker/login-action@v4.6.0`,
  `docker/metadata-action@v6.2.0`, `docker/build-push-action@v7.3.0`. Every reference pin was
  one to four majors stale; the breaking changes in each bump are runtime plumbing (Node 24,
  ESM) and touch none of the inputs used here.

## Justification

**On trust:** the design's elegance is that `cd.yml` inherits its guarantee from
[0027](0027-branch-protection-boundary.md) rather than re-establishing it. The chain is: model
gate decides whether CI is green → protection decides whether a red CI can merge → merging is the
only way to reach `main` → reaching `main` is the only thing that triggers this file. The weak
link is the protection rule, not this workflow — which is why 0027 verifies it behaviourally.

**On `GITHUB_TOKEN`:** GitHub mints it per run and destroys it at job end, scoped by the
`permissions` block in the same file as the steps it authorises. Declaring that block at all
switches the token away from the repository default to *only* what is listed, so it revokes as
much as it grants. Against Docker Hub or ECR this step would instead mean creating a durable
credential and owning its rotation — the posture
[0024](0024-environment-based-tracking-uri.md) already committed this project to avoiding.

**On public visibility:** the repository is already `PUBLIC`, so the package exposes nothing that
`pyproject.toml` and `uv.lock` do not already publish. The point of automated CD in a portfolio
is that a reviewer can pull the artifact and watch it run; a private image makes the phase's most
tangible deliverable unreachable by its intended audience.

Verified without credentials — an anonymous pull token was issued (a private package refuses
this) and the tag list returned:

```json
{"name":"adrianmarchramon/fraud-mlops-pipeline",
 "tags":["main","latest","sha-a5cb07f4f24fbeeb44edde689df35bc6e5f0bcc8"]}
```

The `sha-` tag equals the merge commit of PR #2, so the published image is traceable to the
commit that produced it. CD run `32738991528`: `conclusion: success`, 4m53s.

## Trade-offs / consequences

- **GHCR publishes new packages private by default, regardless of repository visibility, and no
  YAML can change it.** Making it public was a **one-time manual action** in the package settings
  after the first successful publish. Any future package created by this repository will start
  private again and need the same manual step.
- **Every merge to `main` republishes**, moving `main` and `latest` and adding a `sha-` tag.
  Tags accumulate; the registry will need occasional pruning, which nothing here automates.
- **`latest` is mutable.** Phase 9 should deploy the `sha-` tag, not `latest`, or it will not know
  which build is live.
- **`.dockerignore` does not exclude `.github/`**, so the workflow files land inside the image via
  `COPY . /app`. Harmless — a few KB, never executed — but every workflow edit invalidates the
  source layer and forces that step to rebuild.
- **`cache-from`/`cache-to: type=gha`** speeds rebuilds but consumes the repository's Actions
  cache quota alongside `ci.yml`'s uv cache.
