"""Drift-detection tests — deterministic, offline, and without touching disk.

Implementation phase: Phase 8 - Monitoring and Drift (Step 8).

These are the complement to the live demonstration, not a substitute for it.
The demo proves the pieces connect once, on one machine, with three servers up;
it can never run in CI, which has no dataset, no Registry and no Prefect. These
prove the part that CAN be checked on every commit: that the Evidently
extraction path still returns a real share, and that the threshold and
minimum-row policies decide correctly on top of it.

The extraction is the fragile half. `_drifted_share` reaches into Evidently's
result by metric type, and an upgrade that renamed or restructured that metric
would break the loop **silently** — detect_drift() would keep returning
something, just always the wrong thing. Only an end-to-end call through real
Evidently catches that, which is why these tests run the library for real on
synthetic frames rather than mocking it.

Two deliberate choices worth stating. First, `_drifted_share` is private and
tested anyway: it is the unit that carries the risk, and reaching it through
detect_drift() would hide the very thing under test behind two loaders. Second,
an autouse fixture redirects the report path, because _drifted_share writes a
~6 MB HTML file on every call — unredirected, this module would write tens of
megabytes per run and overwrite the demonstration artifact. The fixture patches
the name in `drift`, not in `src.config`: drift.py imported it by value, so
patching config would silently do nothing (the trap
docs/decisions/0021-prediction-log-and-api-tests.md records).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.exceptions import DriftDetectionError
from src.monitoring import drift


@pytest.fixture(autouse=True)
def redirect_drift_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Send every report this module triggers to a temporary directory."""
    report = tmp_path / "drift_report.html"
    monkeypatch.setattr(drift, "DRIFT_REPORT_PATH", report)
    return report


def _frame(mean: float, seed: int, rows: int = 500) -> pd.DataFrame:
    """Build a two-column frame drawn from a normal distribution."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"f1": rng.normal(mean, 1, rows), "f2": rng.normal(mean, 1, rows)}
    )


# --------------------------------------------------------------------------
# The measurement itself, through real Evidently
# --------------------------------------------------------------------------


def test_detects_drift_when_distribution_shifts() -> None:
    share = drift._drifted_share(_frame(0.0, seed=0), _frame(5.0, seed=1))
    assert share > 0.5


def test_no_drift_when_distribution_is_stable() -> None:
    # Different draws from the SAME distribution: this is what "no drift" means
    # in practice, not two identical frames, which would be a weaker check.
    share = drift._drifted_share(_frame(0.0, seed=0), _frame(0.0, seed=1))
    assert share < 0.5


def test_drifted_share_is_a_proportion() -> None:
    share = drift._drifted_share(_frame(0.0, seed=0), _frame(5.0, seed=1))
    assert 0.0 <= share <= 1.0


def test_report_is_written_beside_the_verdict(redirect_drift_report: Path) -> None:
    # The report is a byproduct of the same evaluation, so it must appear
    # whenever a share is computed — that is what makes Step 4's artifact
    # available after any monitoring run.
    drift._drifted_share(_frame(0.0, seed=0), _frame(5.0, seed=1))

    assert redirect_drift_report.exists()
    assert redirect_drift_report.stat().st_size > 0
    assert "<html" in redirect_drift_report.read_text(errors="replace")[:2000].lower()


# --------------------------------------------------------------------------
# The policy on top of the measurement
# --------------------------------------------------------------------------


@pytest.fixture
def stub_loaders(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace both dataset loaders so no artifact or log is required."""
    frame = _frame(0.0, seed=0, rows=200)
    monkeypatch.setattr(drift, "_load_reference", lambda: frame)
    monkeypatch.setattr(drift, "_load_current", lambda: frame)


@pytest.mark.parametrize(
    ("share", "threshold", "expected"),
    [
        (0.90, 0.5, True),
        (0.10, 0.5, False),
        # The comparison is >=, so a share sitting exactly on the threshold
        # reports drift. Pinned because flipping it to > would make a
        # threshold of 1.0 unreachable and silently disable the loop.
        (0.50, 0.5, True),
    ],
)
def test_threshold_decides_the_verdict(
    monkeypatch: pytest.MonkeyPatch,
    stub_loaders: None,
    share: float,
    threshold: float,
    expected: bool,
) -> None:
    monkeypatch.setattr(drift, "_drifted_share", lambda reference, current: share)
    monkeypatch.setattr(drift, "DRIFT_THRESHOLD", threshold)

    assert drift.detect_drift() is expected


def test_too_few_rows_reports_no_drift_without_measuring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Measured in ADR 0038: Evidently on a 3-row current frame reports
    # share=1.0, every column drifted. Without this floor the three smoke-test
    # records in the real log would fire a retrain on the first scheduled run.
    called = False

    def _fail(reference: pd.DataFrame, current: pd.DataFrame) -> float:
        nonlocal called
        called = True
        return 1.0

    monkeypatch.setattr(drift, "_load_reference", lambda: _frame(0.0, seed=0))
    monkeypatch.setattr(drift, "_load_current", lambda: _frame(0.0, seed=1, rows=3))
    monkeypatch.setattr(drift, "_drifted_share", _fail)
    monkeypatch.setattr(drift, "DRIFT_MIN_ROWS", 100)

    assert drift.detect_drift() is False
    assert not called, "the guard must short-circuit before Evidently runs"


# --------------------------------------------------------------------------
# Failing loudly rather than reporting a false all-clear
# --------------------------------------------------------------------------


def test_missing_reference_column_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = _frame(0.0, seed=0)
    current = reference.drop(columns=["f2"])

    with pytest.raises(DriftDetectionError, match="missing 1 reference column"):
        drift._align_to_reference(reference, current)


def test_alignment_reorders_onto_the_reference_schema() -> None:
    reference = _frame(0.0, seed=0)
    current = _frame(0.0, seed=1)[["f2", "f1"]]

    assert list(drift._align_to_reference(reference, current).columns) == ["f1", "f2"]


def test_unreadable_current_dataset_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = Path("/nonexistent/predictions.jsonl")
    monkeypatch.setattr(drift, "CURRENT_DATA_PATH", missing)

    with pytest.raises(DriftDetectionError, match="Current dataset not found"):
        drift._load_current()
