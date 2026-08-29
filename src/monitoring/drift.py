"""Data-drift detection with Evidently.

Implementation phase: Phase 8 - Monitoring and Drift (Steps 1-3).
Current status: real column-level data drift. detect_drift() compares the
fixed, DVC-versioned reference distribution (src/monitoring/reference.py)
against a current window resolved from configuration, and reports whether the
share of drifted columns has reached config.DRIFT_THRESHOLD.

This replaces the Phase 7 placeholder that returned a constant False. The
signature is unchanged and takes no arguments, so pipelines/ needs no edit:
the monitoring flow built in Phase 7 keeps calling detect_drift() exactly as
it did, and the loop it wired energises without touching the wiring. That is
why the current dataset is resolved through config.CURRENT_DATA_PATH rather
than passed in.

**Data drift, not concept drift.** Concept drift is a change in P(y|X) and
needs y. Nothing in this project captures it: logs/predictions.jsonl records
the model's own decision, never the outcome, and in fraud the truth arrives
days or weeks later with a chargeback (label delay). So what is measured here
is P(X) — the input distribution — which is the honest signal available, and
also the early one: it moves before any label could have arrived. Prediction
drift on the score distribution belongs to a later step; the module docstring
of a future revision will say so when it lands.

**No prefect import, ever.** Business logic in src/ knows nothing about how it
is invoked; pipelines/monitoring_pipeline.py owns get_run_logger() and the
retry budget. That boundary is why the same function is callable from a test,
from a container and from a shell with no orchestrator installed.

Evidently's public API changed shape between generations. Everything below is
written against what evidently 0.7.21 actually returns, measured in this
repository rather than copied from documentation: Report(...).run() takes
(current, reference) in that order, exposes .dict() and not .as_dict(), and
reports the aggregate under a DriftedColumnsCount metric whose value is
{"count": float, "share": float}. See
docs/decisions/0038-evidently-dependency-and-api.md.
"""

import json
import logging
from typing import Any

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.config import (
    CURRENT_DATA_PATH,
    DRIFT_MIN_ROWS,
    DRIFT_THRESHOLD,
    REFERENCE_DATA_PATH,
)
from src.exceptions import DriftDetectionError

logger = logging.getLogger(__name__)

# The aggregate metric DataDriftPreset emits alongside the per-column ones.
# Matched on this stable config type rather than by list position: the preset
# is free to reorder or add metrics between releases, and an index would then
# read a per-column p-value as if it were the dataset verdict — silently, since
# both are numbers.
DRIFTED_COLUMNS_METRIC = "evidently:metric_v2:DriftedColumnsCount"


def _load_reference() -> pd.DataFrame:
    """Load the fixed reference distribution built by the `reference` stage.

    Returns:
        The reference features, in raw feature space, target column absent.

    Raises:
        DriftDetectionError: if the artifact has not been built.
    """
    try:
        return pd.read_parquet(REFERENCE_DATA_PATH)
    except (FileNotFoundError, OSError) as exc:
        raise DriftDetectionError(
            f"Reference dataset not found at {REFERENCE_DATA_PATH}. "
            "Build it with `uv run dvc repro reference`."
        ) from exc


def _load_current() -> pd.DataFrame:
    """Load the current window from the API's prediction log.

    One JSON object per line; only the "input" key is read — the raw request
    payload as the API received it. That key, the file name and the record
    shape are the contract frozen in Phase 4 by
    docs/decisions/0021-prediction-log-and-api-tests.md, written down back then
    precisely so this function could exist.

    Returns:
        One row per logged prediction, one column per input feature. Empty if
        the log has no records yet.

    Raises:
        DriftDetectionError: if the file is missing, or a line is not JSON or
            carries no "input" key. A malformed log is reported, never skipped:
            silently dropping records would bias the very distribution being
            measured.
    """
    records: list[dict[str, float]] = []
    try:
        with open(CURRENT_DATA_PATH) as f:
            for number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DriftDetectionError(
                        f"{CURRENT_DATA_PATH}:{number} is not valid JSON"
                    ) from exc
                try:
                    records.append(record["input"])
                except (KeyError, TypeError) as exc:
                    raise DriftDetectionError(
                        f'{CURRENT_DATA_PATH}:{number} has no "input" key'
                    ) from exc
    except FileNotFoundError as exc:
        raise DriftDetectionError(
            f"Current dataset not found at {CURRENT_DATA_PATH}. "
            "The API writes it on its first prediction."
        ) from exc

    return pd.DataFrame(records)


def _align_to_reference(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Reorder the current frame onto the reference's columns.

    JSON objects preserve insertion order, so the log's column order follows
    the API schema while the reference follows the CSV — same 30 names, two
    orders. Evidently matches by name, so this is belt-and-braces for the
    ordering, but it is load-bearing for the missing-column check: a payload
    that lost a feature would otherwise be compared on whatever columns
    happened to survive.

    Args:
        reference: the fixed baseline, whose columns define the schema.
        current: the recent window to align.

    Returns:
        `current` restricted and reordered to the reference's columns.

    Raises:
        DriftDetectionError: if the current window is missing any reference
            column.
    """
    missing = [column for column in reference.columns if column not in current.columns]
    if missing:
        raise DriftDetectionError(
            f"Current dataset at {CURRENT_DATA_PATH} is missing "
            f"{len(missing)} reference column(s): {', '.join(missing)}"
        )
    return current[list(reference.columns)]


def _drifted_share(reference: pd.DataFrame, current: pd.DataFrame) -> float:
    """Run the Evidently comparison and return the share of drifted columns.

    Evidently picks the statistical test per column from its type and sample
    size — K-S on these 30 numerical features — and aggregates the per-column
    verdicts into one share. Only that aggregate is read here; the threshold
    applied to it is this project's policy, not Evidently's.

    Args:
        reference: the fixed baseline distribution.
        current: the recent window, already aligned to the reference schema.

    Returns:
        The proportion of columns Evidently judged drifted, in [0, 1].

    Raises:
        DriftDetectionError: if the result carries no aggregate metric, which
            would mean the preset's output shape changed under us.
    """
    # (current, reference) — this argument order reversed in Evidently 0.7 and
    # is silently wrong rather than an error if swapped: drift is not symmetric
    # in general, and the report would simply describe the wrong direction.
    report = Report([DataDriftPreset()], include_tests=True)
    result: dict[str, Any] = report.run(current, reference).dict()

    for metric in result.get("metrics", []):
        if metric.get("config", {}).get("type") == DRIFTED_COLUMNS_METRIC:
            return float(metric["value"]["share"])

    raise DriftDetectionError(
        f"Evidently returned no {DRIFTED_COLUMNS_METRIC} metric; "
        "the DataDriftPreset result shape has changed"
    )


def detect_drift() -> bool:
    """Report whether recent data has drifted from the training distribution.

    Called by pipelines/monitoring_pipeline.py on its daily schedule; a True
    triggers retraining with no human in the path.

    Returns:
        True if the share of drifted columns is at least
        config.DRIFT_THRESHOLD. False if it is below it, and also False when
        the current window is too small to support a verdict — declining to
        answer is reported as "no drift" because that is the branch that costs
        nothing, and it is logged at WARNING so the silence is visible.

    Raises:
        DriftDetectionError: if either dataset cannot be read or the Evidently
            result cannot be interpreted. Deliberately not swallowed: a broken
            monitor must fail loudly rather than report "all clear".
    """
    reference = _load_reference()
    current = _load_current()

    if len(current) < DRIFT_MIN_ROWS:
        logger.warning(
            "Drift check skipped: %d rows in %s, below the %d-row minimum. "
            "Reporting no drift; a test on this few points measures noise.",
            len(current),
            CURRENT_DATA_PATH,
            DRIFT_MIN_ROWS,
        )
        return False

    share = _drifted_share(reference, _align_to_reference(reference, current))
    drift_detected = share >= DRIFT_THRESHOLD

    logger.info(
        "Drift check: %.4f of columns drifted across %d rows "
        "(threshold %.4f) -> drift=%s",
        share,
        len(current),
        DRIFT_THRESHOLD,
        drift_detected,
    )
    return drift_detected
