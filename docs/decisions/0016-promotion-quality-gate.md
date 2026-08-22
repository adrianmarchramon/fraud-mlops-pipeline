# Decision 16: Promotion quality gate — PR-AUC alone, `@production` alone, strictly better

- **Date:** 2026-08-22
- **Status:** Accepted (inherits [0002](0002-cost-asymmetry.md)'s illustrative-cost caveat)

## Context

Registering a version does not put it into production. Something must decide whether a
candidate replaces what is serving. MLflow offers two vocabularies for this: the deprecated
**stages** (`None`/`Staging`/`Production`/`Archived`, moved with
`transition_model_version_stage()`) and **aliases + tags**. `project_context/mlops_phase3.md`
(line 52) records that stages are deprecated and slated for removal, while
`mlops_generalRoadmap.md` (lines 217-223) and `mlops_fundamentals.md` (line 198) still teach the
stage vocabulary — a divergence resolved here in favour of the detailed phase document.

## Decision

- **Aliases, never stages.** A single alias, `@production`, moved with
  `set_registered_model_alias()`. No stage API appears anywhere in `src/`.
- **`pr_auc` is the only promotion criterion**, read from the candidate's own version **tag**,
  compared against the tag of the version currently holding `@production`.
- **Strictly better wins.** `candidate > incumbent` promotes; equal does not.
- **No production model yet is a valid state, not an error**: `get_production_metric()` returns
  `None`, and the candidate is promoted unconditionally.

## Alternatives considered

- **Adding a business-cost non-regression guard** (using `cost_optimal_threshold()` from
  Phase 2) — rejected as a conscious simplification, not an oversight. A candidate could raise
  `pr_auc` while worsening expected cost at its own threshold, so the gap is real. But
  [0002](0002-cost-asymmetry.md) marks its costs *"illustrative, pending real business data"*,
  and gating deployments on a number the project itself calls provisional is worse than not
  gating on it. Revisit when real per-error costs exist — the same trigger
  [0014](0014-cost-optimal-threshold.md) records for re-deriving the threshold.
- **Seeding `@champion`/`@challenger` now** — rejected. A/B testing is filed as a stretch goal
  (`mlops_endRoadmap.md` line 35), and an alias with no consumer is the dangling dependency
  [0012](0012-imbalance-strategy.md) avoided with optional SMOTE.
- **Comparing with `>=`** — rejected. Re-registering an unchanged model produces a
  byte-identical `pr_auc`; `>=` would churn the alias for no gain. Verified live: a second
  registration of the losing run was refused.

## Justification

Reading the metric back from a **tag** rather than recomputing it makes the decision cheap,
deterministic, and auditable: the comparison is against the number that version was actually
promoted on, not a re-derivation that might use different data. Aliases separate a version's
*identity* (permanent — v3 is forever v3) from its *role* (transient — today's production is
next month's archive), which the single mutable stage field conflated.

The gate was demonstrated live in all three branches against the real Registry, with authentic
Phase 2 metrics and no fabricated numbers: a bootstrap promotion with nothing in production, a
genuine comparison promotion (0.8760 beats 0.7249), and a genuine refusal (0.7249 does not beat
0.8760, alias untouched).

`find_best_run()` breaks ties on `attributes.start_time DESC`. This is not cosmetic: re-running
an unchanged pipeline produces runs with byte-identical metrics — six such runs exist — so
without an explicit second key the winner would depend on MLflow's undeclared secondary
ordering.

## Trade-offs / consequences

- **A candidate can improve `pr_auc` and still cost more.** Documented above; accepted until
  real cost data exists.
- **Rejected versions stay registered forever.** Deliberate: the Registry is a history, not a
  tidy shortlist, and the refused versions are the evidence that the gate refuses.
- **The gate protects the alias, not the artifact.** Nothing prevents a human from moving
  `@production` by hand; the guarantee is that the automated path will not.
- **`get_production_metric()` cannot rely on error codes alone.** MLflow reports an unknown
  registered model as `RESOURCE_DOES_NOT_EXIST` but a known model without the alias as
  `INVALID_PARAMETER_VALUE` — a code that also covers real errors. It therefore tests the alias
  map explicitly. This was found the hard way: an early run registered a version and then
  crashed at promotion, leaving it unaliased.
