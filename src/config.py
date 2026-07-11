"""Centralized configuration: filesystem paths and shared constants.

Nothing outside this module should hardcode a path; every other module
imports its paths and constants from here so a change lands in one place.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA = DATA_DIR / "raw" / "creditcard.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
PREPROCESSOR_PATH = PROCESSED_DIR / "preprocessor.joblib"

TARGET = "Class"
