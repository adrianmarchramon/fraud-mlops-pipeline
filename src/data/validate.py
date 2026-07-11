"""Data validation: the Pandera quality contract for the raw dataset.

Any data that does not match the expected shape fails loudly here,
before it can reach preprocessing or, worse, a production model.
"""

import json
import logging
from typing import TypedDict

import pandera.pandas as pa

from src.config import REPORTS_DIR
from src.data.ingest import load_raw_data
from src.exceptions import DataValidationError

logger = logging.getLogger(__name__)

# The 28 anonymized PCA features of the dataset
V_COLUMNS = [f"V{i}" for i in range(1, 29)]

# The contract: what shape the raw data must have to be valid
raw_schema = pa.DataFrameSchema(
    columns={
        "Time": pa.Column(float, pa.Check.ge(0)),
        **{v: pa.Column(float, nullable=False) for v in V_COLUMNS},
        "Amount": pa.Column(float, pa.Check.ge(0)),
        "Class": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,  # unexpected columns cause validation to fail
    coerce=True,  # attempt to convert to the declared type before validating
)


class ValidationReport(TypedDict):
    """Shape of the JSON report written to reports/validation.json."""

    n_rows: int
    n_fraud: int
    fraud_rate: float
    status: str


def validate_raw_data() -> ValidationReport:
    """Validate raw data against the contract and write a health report.

    Returns:
        A small report with row count, fraud count, fraud rate, and the
        validation status. It is also persisted to reports/validation.json.

    Raises:
        DataValidationError: if the raw data violates the schema contract.
            The underlying pandera.errors.SchemaErrors is chained so no
            diagnostic detail is lost.
    """
    df = load_raw_data()

    try:
        # lazy=True accumulates ALL failures before raising, instead of
        # stopping at the first one — better for diagnostics.
        raw_schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        logger.error(
            "Raw data failed validation: %d failure case(s)",
            len(exc.failure_cases),
        )
        raise DataValidationError(
            "Raw data does not satisfy the quality contract"
        ) from exc

    report: ValidationReport = {
        "n_rows": len(df),
        "n_fraud": int(df["Class"].sum()),
        "fraud_rate": round(float(df["Class"].mean()), 6),
        "status": "passed",
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    with open(REPORTS_DIR / "validation.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(
        "Validation passed: %d rows, fraud rate %.4f%%",
        report["n_rows"],
        report["fraud_rate"] * 100,
    )
    return report


if __name__ == "__main__":
    validate_raw_data()
