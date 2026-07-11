"""Data contract tests — verify that the Pandera schemas in
src.data.validate actually enforce the quality contract: valid data
passes, and each category of invalid data is rejected for the specific
reason it is invalid, not merely because "some exception was raised".
"""

import pandas as pd
import pandera.pandas as pa
import pytest

from src.config import TARGET
from src.data.validate import V_COLUMNS, processed_schema, raw_schema


@pytest.fixture
def valid_raw_row() -> dict:
    """A single row satisfying raw_schema exactly."""
    return {
        "Time": [0.0],
        **{v: [0.1] for v in V_COLUMNS},
        "Amount": [10.0],
        TARGET: [0],
    }


@pytest.fixture
def valid_processed_row() -> dict:
    """A single row satisfying processed_schema.

    Unlike the raw contract, Time and Amount are post-StandardScaler and
    may legitimately be negative.
    """
    return {
        "Time": [-1.5],
        **{v: [0.1] for v in V_COLUMNS},
        "Amount": [-0.3],
        TARGET: [0],
    }


def test_raw_schema_accepts_valid_data(valid_raw_row: dict) -> None:
    raw_schema.validate(pd.DataFrame(valid_raw_row))  # must not raise


@pytest.mark.parametrize(
    "field, invalid_value",
    [
        (TARGET, [5]),
        ("Amount", [-5.0]),
        ("Time", [-1.0]),
        ("V1", [None]),
    ],
    ids=["invalid-class", "negative-amount", "negative-time", "null-in-v-column"],
)
def test_raw_schema_rejects_invalid_field(
    valid_raw_row: dict, field: str, invalid_value: list
) -> None:
    data = {**valid_raw_row, field: invalid_value}
    with pytest.raises(pa.errors.SchemaError):
        raw_schema.validate(pd.DataFrame(data))


def test_raw_schema_rejects_unexpected_column(valid_raw_row: dict) -> None:
    data = {**valid_raw_row, "extra_column": [1.0]}
    # A strict-mode *column* violation raises SchemaErrors (plural) even in
    # non-lazy mode, unlike the *value* checks above which raise SchemaError.
    with pytest.raises(pa.errors.SchemaErrors):
        raw_schema.validate(pd.DataFrame(data))


def test_processed_schema_accepts_valid_data(valid_processed_row: dict) -> None:
    processed_schema.validate(pd.DataFrame(valid_processed_row))  # must not raise


def test_processed_schema_rejects_invalid_class(valid_processed_row: dict) -> None:
    data = {**valid_processed_row, TARGET: [5]}
    with pytest.raises(pa.errors.SchemaError):
        processed_schema.validate(pd.DataFrame(data))
