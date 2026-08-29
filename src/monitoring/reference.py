"""Builds the fixed reference distribution that drift is measured against.

Implementation phase: Phase 8 - Monitoring and Drift (Step 1).

Drift is a comparison, so exactly one side of it has to be pinned or the
number means nothing: if the baseline is recomputed on every run, a changed
verdict is ambiguous between "reality moved" and "the yardstick moved". This
module produces that pinned side once, as a DVC-versioned artifact with a
hash, on the same reproducibility argument that brought DVC into Phase 1.

Two properties of the output are load-bearing, and both differ from the
reference material's `pd.read_csv(RAW_DATA).sample(n=5000)`:

**Training rows, not the whole file.** Data drift asks whether production
inputs have moved away from what the model learned, so the baseline is the
learned distribution. The split is re-derived here rather than read from disk
because preprocess.py persists only the *scaled* splits; nothing in the
repository holds the training rows in raw units. Re-deriving is exact, not
approximate: train_test_split is deterministic given the same X, y, test_size,
random_state and stratify, and those come from the same params.yaml block
preprocess.py reads, so the rows are the same rows.

**Raw feature space.** This is forced by the other side of the comparison.
src/api/main.py logs untransformed request payloads, while
data/processed/test.parquet has Time and Amount standardized (measured at the
time of writing: Time in [-1.998, 1.640], Amount in [-0.352, 51.143]).
Comparing a scaled reference against raw traffic would manufacture permanent
drift on exactly those two columns and compare nothing on the other 28.

See docs/decisions/0037-reference-dataset.md.
"""

import logging
from typing import TypedDict

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.config import (
    MONITORING_DIR,
    PROJECT_ROOT,
    REFERENCE_DATA_PATH,
    TARGET,
)
from src.data.ingest import load_raw_data
from src.data.preprocess import load_params as load_preprocess_params
from src.exceptions import DriftDetectionError

logger = logging.getLogger(__name__)


class MonitoringParams(TypedDict):
    """Shape of the `monitoring:` block read from params.yaml."""

    rows: int
    random_state: int


def load_monitoring_params() -> MonitoringParams:
    """Load the versioned monitoring parameters from params.yaml.

    Returns:
        The `monitoring` block of params.yaml.

    Raises:
        DriftDetectionError: if params.yaml is missing, is not valid YAML, or
            has no top-level `monitoring` key.
    """
    params_path = PROJECT_ROOT / "params.yaml"
    try:
        with open(params_path) as f:
            all_params = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise DriftDetectionError(f"params.yaml not found at {params_path}") from exc
    except yaml.YAMLError as exc:
        raise DriftDetectionError(
            f"params.yaml is not valid YAML: {params_path}"
        ) from exc

    try:
        # yaml.safe_load() is typed as returning Any, so the annotation here is
        # what states the contract this function promises to its callers.
        params: MonitoringParams = all_params["monitoring"]
    except (KeyError, TypeError) as exc:
        raise DriftDetectionError(
            "params.yaml has no top-level 'monitoring' key"
        ) from exc
    return params


def build_reference() -> pd.DataFrame:
    """Draw the reference sample from the raw-space training rows.

    The preprocessing parameters are imported from src.data.preprocess rather
    than re-read here, so the split can never silently diverge from the one the
    model was actually trained on: change test_size in params.yaml and both
    move together.

    Returns:
        The sampled reference features, with the target column dropped.

    Raises:
        DriftDetectionError: if the requested sample is larger than the
            training split.
    """
    preprocess_params = load_preprocess_params()
    monitoring_params = load_monitoring_params()

    df = load_raw_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Same call, same arguments, same stratification as preprocess.py:133 —
    # only the transform that follows it there is deliberately not applied.
    X_train, _, _, _ = train_test_split(
        X,
        y,
        test_size=preprocess_params["test_size"],
        random_state=preprocess_params["random_state"],
        stratify=y,
    )

    rows = monitoring_params["rows"]
    if rows > len(X_train):
        raise DriftDetectionError(
            f"monitoring.rows={rows} exceeds the {len(X_train)}-row training "
            "split; lower it in params.yaml"
        )

    reference: pd.DataFrame = X_train.sample(
        n=rows, random_state=monitoring_params["random_state"]
    )
    return reference


def main() -> None:
    """Build the reference dataset and persist it for the `reference` stage."""
    reference = build_reference()

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(REFERENCE_DATA_PATH)

    logger.info(
        "Reference dataset written to %s: %d rows, %d columns",
        REFERENCE_DATA_PATH,
        len(reference),
        len(reference.columns),
    )


if __name__ == "__main__":
    main()
