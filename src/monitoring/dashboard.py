"""Deliberately empty: the monitoring dashboard is drift.py plus its report.

Planned implementation phase: Phase 8 - Monitoring and Drift.
Current status: no logic, and none intended.

Phase 0 created this file speculatively, and the README architecture diagram
names it inside the Monitoring box. Phase 8 resolved that the thing the
diagram calls a dashboard is not a module here: it is the self-contained
interactive HTML page src/monitoring/drift.py writes to
config.DRIFT_REPORT_PATH on every check -- one section per feature, reference
and current distributions overlaid, per-column test results, and the drifted
summary.

Rendering it from this module would mean re-running the Evidently comparison
that already produced the verdict, to draw what that run had just computed.
The report and the number are two views of one evaluation, so they are
produced together. See docs/decisions/0039-alerting-and-report-placement.md.

Kept rather than deleted because the diagram references it and an empty file
that explains itself is more useful to a reader than a dangling name. If a
persistent UI over historical reports is ever wanted, this is where it goes --
it has no owner today.
"""
