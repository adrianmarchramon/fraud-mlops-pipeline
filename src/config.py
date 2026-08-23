"""Centralized configuration: filesystem paths and shared constants.

Nothing outside this module should hardcode a path; every other module
imports its paths and constants from here so a change lands in one place.
"""

import os
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

# MLflow experiment tracking. A SQLite backend is used from the start (instead of
# flat mlruns/ storage) because it is required by the Model Registry in Phase 3;
# choosing it now avoids migrating the backend later.
#
# Read from the environment so one build behaves correctly everywhere: the
# default keeps local runs on the SQLite file, while docker-compose injects
# http://mlflow:5000 to reach the MLflow service. This is the override
# 0011-mlflow-sqlite-backend.md deferred to Phase 5, and the only reason the
# same train.py and register.py can populate a containerized registry unchanged.
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
EXPERIMENT_NAME = "fraud-detection"
# The Model Registry entry, deliberately distinct from EXPERIMENT_NAME: an
# experiment groups training runs, a registered model groups packaged versions.
# They are two different MLflow entities and must not share a name.
MODEL_NAME = "fraud-detector"
MODELS_DIR = PROJECT_ROOT / "models"

# Every prediction the API serves is appended here, one JSON object per line.
# Phase 8 imports this exact constant and reads each record's "input" key, so
# the name, the path and the record schema are a contract with a phase that
# does not exist yet — not a preference. The directory is created lazily on
# first write; it is a runtime artifact and stays out of Git.
PREDICTIONS_LOG = PROJECT_ROOT / "logs" / "predictions.jsonl"
