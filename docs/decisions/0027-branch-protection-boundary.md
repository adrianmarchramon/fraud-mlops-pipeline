# Decision 27: Branch protection as the enforcement boundary

- **Date:** 2026-08-24
- **Status:** Accepted

## Context

`ci.yml` can only ever *report* a status — a coloured mark attached to a commit. Nothing about
publishing a red mark prevents anyone from clicking Merge. The authority to refuse a merge lives
in the repository's own settings, on GitHub's side of the boundary.

This matters beyond tidiness: `cd.yml` (see [0029](0029-cd-workflow-and-ghcr-publication.md))
deliberately re-runs none of CI's checks, and its entire right to do so derives from the
guarantee that nothing reaches `main` unvalidated. Without a real protection rule, that
guarantee is an assumption, and the phase's "self-verifying pipeline" claim collapses into two
workflows running beside each other.

## Decision

**A classic branch protection rule on `main`**, applied manually through the GitHub web UI, with:

| Setting | Value |
|---|---|
| Require a pull request before merging | on |
| **Required approvals** | **`0`** |
| Require status checks to pass | on |
| **Required check** | **`Lint, format, types and tests`** |
| Require branches to be up to date | off |
| Require linear history | off |
| **Do not allow bypassing the above settings** | **on** |

The required check string is the **job `name:`** at `.github/workflows/ci.yml:29`, not the job
*key* (`quality`) and not the workflow name (`CI`). Those three differ, and only the job name is
what GitHub records and matches against.

**Configured by hand, not by `gh api`.** The `gh` CLI on this machine is authenticated as a
*different* account (`amr299-ua`) from the repository owner (`adrianmarchramon`), holding
`viewerPermission: READ`. No `PUT` to the protection endpoint would have succeeded at any token
scope, so the manual route was not a preference but the only available path.

## Alternatives considered

- **A repository ruleset** instead of a classic rule — rejected for now. More expressive and
  exportable as JSON, but a reviewer opening Settings sees an unfamiliar screen where the
  conventional rule was expected, and the reference material describes the classic path.
- **`gh api -X PUT …/branches/main/protection`** — impossible here, as above. It would also carry
  a larger blast radius: a check-name typo silently blocks every future merge, including the
  author's own.
- **Required approvals `1`** — rejected, and this is the trap worth recording. On a solo
  repository the owner cannot approve their own pull request, and with bypass disabled there is
  no override. The combination produces a `main` that **nobody can merge to**. `0` keeps CI as
  the gate without adding an approval nobody can grant.
- **Leaving bypass enabled ("include administrators" off)** — rejected. With it off, protection
  asserts nothing about the only person who commits here, and `cd.yml`'s trust becomes
  decorative.
- **Requiring linear history** — rejected: it forces squash merges, which would flatten the
  atomic commits this project deliberately keeps. Linearity is preserved by *choosing* rebase
  merges instead (see [0030](0030-phase-6-verification-and-dod-deviations.md)).

## Justification

The rule was verified **behaviourally rather than by reading its configuration**, which is the
stronger evidence and also the only evidence available:

```
PR #1  CLOSED  merged=null           check "Lint, format, types and tests" = FAILURE  → refused
       mergeStateStatus: BLOCKED,  mergeable: MERGEABLE
PR #2  MERGED  2026-08-24T14:29:10Z  check "Lint, format, types and tests" = SUCCESS  → allowed
```

PR #1 was a deliberately degraded model (`pr_auc` forced to `0.61`, under the `0.75` floor). Its
`mergeable: MERGEABLE` means Git found no conflict — so the `BLOCKED` state came from a *rule*,
not from the merge itself. A red check refused a merge; a green one permitted it; the check name
matches `ci.yml:29` exactly. That is the property the whole phase rests on, demonstrated rather
than asserted.

## Trade-offs / consequences

- **Direct pushes to `main` are now refused, for everyone including the owner.** Every change
  from here on — Phase 7's orchestration included — arrives through a branch and a pull request.
  This is a real workflow change: the first 60 commits of this project went straight to `main`.
- **Pull requests must be created and merged in the browser**, since the `gh` CLI here is
  read-only. `git push` still works, because Git goes over SSH as the owner while `gh` goes over
  HTTPS as the other account.
- **The rule's configuration cannot be read back from this machine.**
  `gh api …/branches/main/protection` returns `404` under a READ token — which means *"you may
  not ask"*, not *"no protection exists"*. Only `.protected: true` on the branch object and the
  behavioural evidence above are verifiable from here; the individual settings are trusted from
  the browser session that set them.
- **The check name is now a coupling.** Renaming the `quality` job's `name:` in `ci.yml` breaks
  the rule silently: the required check would never report, and every pull request would hang
  waiting for it. Rename only alongside the settings page.
