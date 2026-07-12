# Decision 5: DVC remote — local filesystem store (for now)

- **Date:** 2026-07-12
- **Status:** Accepted (interim — cloud migration deferred)

## Context

Phase 1 puts the dataset under DVC control (`data/raw/creditcard.csv.dvc`, committed in
`1c1d2b5`) and its processed outputs (`train.parquet`, `test.parquet`, `preprocessor.joblib`)
under the `preprocess` stage. DVC stores the heavy data in a **remote**, keeping only
lightweight pointers in Git. A remote had to be chosen in Step 1 before any `dvc push`.

Evidence in repo: `.dvc/config` declares a single default remote
`localremote → /home/amr/.dvc-remotes/fraud-mlops-pipeline` (committed in `c07c892`);
`dvc remote list` confirms it as `(default)`; `dvc status -c` reports the cache and remote in sync.

## Decision

Use a **local-filesystem DVC remote** (a directory outside the repository) as the default
remote for Phase 1. `project_context/mlops_phase1.md` (Step 1) explicitly sanctions this as the
"fastest option to get started", with cloud migration available later "without any issues".

## Alternatives considered

- **Google Drive** (`dvc[gdrive]`, `gdrive://FOLDER_ID`) — free and machine-independent, but
  needs OAuth credentials and adds a dependency extra; friction disproportionate to a
  single-developer Phase 1.
- **AWS S3** (`s3://bucket/path`) — the standard professional choice and genuinely portable,
  but requires an AWS account, credentials, and cost management not yet warranted at this stage.

## Justification

A local remote is zero-credential, zero-cost, and instant to set up, which is all Phase 1
needs to exercise the full `dvc add` / `dvc push` / `dvc pull` / `dvc repro` loop. Because the
remote is pure configuration (`.dvc/config`), migrating to S3 or GDrive later is a config
change with **no impact on pipeline code** — the exact portability the roadmap promises.

## Trade-offs / consequences

- **Not reproducible across machines.** A clone on another host cannot `dvc pull` the data,
  because the remote path is local to the author's machine. This is documented honestly in the
  README ("Getting the data"): third parties reproduce instead by fetching the raw CSV from
  Kaggle and running `dvc repro`, which rebuilds identical processed artifacts deterministically.
- The Phase 1 reproducibility check (clean clone + `dvc pull` + `dvc repro`) is therefore a
  **same-machine** guarantee for now; a shared/cloud remote is the planned upgrade before any
  deployment phase that needs cross-host data access.
- No secrets are involved (a local path), so nothing sensitive lives in the versioned
  `.dvc/config`.
