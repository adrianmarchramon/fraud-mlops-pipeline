"""Raw data ingestion: the single source of truth for how data enters.

Any future change of data source (database, API, data warehouse) is
confined to this module; validation and preprocessing depend on it and
never read the raw source directly.
"""

import logging

import pandas as pd

from src.config import RAW_DATA
from src.exceptions import DataIngestionError

logger = logging.getLogger(__name__)


def load_raw_data() -> pd.DataFrame:
    """Load the raw transaction dataset from the path declared in config.

    Returns:
        The raw transactions exactly as read from disk — no cleaning or
        transformation applied at this stage.

    Raises:
        DataIngestionError: if the raw data file is missing or cannot be
            parsed as CSV.
    """
    logger.info("Loading raw data from %s", RAW_DATA)
    try:
        df = pd.read_csv(RAW_DATA)
    except FileNotFoundError as exc:
        logger.error("Raw data file not found at %s", RAW_DATA)
        raise DataIngestionError(f"Raw data file not found at {RAW_DATA}") from exc
    except pd.errors.ParserError as exc:
        logger.error("Raw data file could not be parsed as CSV: %s", RAW_DATA)
        raise DataIngestionError(f"Could not parse {RAW_DATA} as CSV") from exc

    logger.info("Raw data loaded: %d rows, %d columns", len(df), len(df.columns))
    return df
