# Decision 10: Pandera contract — `strict=True` and `lazy=True`

- **Date:** 2026-07-12
- **Status:** Accepted

## Context

Validation is the quality gate that keeps bad data from silently reaching training or the
production model. Two Pandera options materially change how loudly and how completely the
contract fails: `strict` (whether unexpected columns are tolerated) and `lazy` (whether all
failures are reported at once or only the first).

Evidence in repo: `src/data/validate.py` defines `raw_schema` and `processed_schema` with
`strict=True, coerce=True`; both `validate_raw_data()` and `validate_processed_data()` call
`.validate(df, lazy=True)`. The behaviour is pinned by tests in `tests/test_data.py` (a strict
*column* violation raises `SchemaErrors`; value violations raise `SchemaError`).

## Decision

- **`strict=True`** — reject any column not declared in the schema.
- **`lazy=True`** — run every check and accumulate all failures before raising.
- **`coerce=True`** — attempt declared-type coercion before validating.

Applied at **both** boundaries: raw input (`raw_schema`) and the preprocessing output
(`processed_schema`), embodying "validate at every boundary".

## Alternatives considered

- **Non-strict schema** — rejected: an upstream rename or an added column would pass silently,
  letting differently-shaped data flow to the model — exactly the silent failure this phase
  exists to prevent.
- **Eager validation** (default, first-error-only) — rejected: it hides the full extent of a
  data problem, forcing slow one-error-at-a-time debugging instead of a complete diagnostic in
  one pass.

## Justification

`strict` turns "surely the columns are what I expect" into an enforced guarantee; `lazy` turns a
failure into a complete report rather than a single symptom. Together they make the contract both
loud and informative — and the tests prove the contract actually rejects each category of bad
data, not merely that "some exception was raised".

## Trade-offs / consequences

- `strict=True` will reject even *benign* extra columns; this is intentional — a schema change
  should be a conscious edit to `validate.py`, not an accident.
- `lazy=True` collects all failure cases in memory before raising; negligible here and well worth
  the diagnostic payoff.
- Callers must distinguish `SchemaError` (value checks) from `SchemaErrors` (strict-column
  violations); both are chained into `DataValidationError` so no diagnostic detail is lost.
