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

# ---------------------------------------------------------------------------
# Drift monitoring (Phase 8)
# ---------------------------------------------------------------------------
# The fixed baseline drift is measured against, built by the `reference` stage
# in dvc.yaml and versioned like any other pipeline output. It is an artifact
# rather than a computation on purpose: a baseline recomputed per run makes a
# changed verdict ambiguous between "reality moved" and "the yardstick moved".
MONITORING_DIR = DATA_DIR / "monitoring"
REFERENCE_DATA_PATH = Path(
    os.getenv("REFERENCE_DATA_PATH", str(MONITORING_DIR / "reference.parquet"))
)

# The other side of the comparison. It defaults to the API's own prediction log
# — the contract ADR 0021 froze in Phase 4 — and is overridable so a shifted
# batch can be pointed at without editing code. It is deliberately NOT a
# detect_drift() argument: Phase 7 wired that call as detect_drift(), and the
# monitoring flow must keep working unchanged.
CURRENT_DATA_PATH = Path(os.getenv("CURRENT_DATA_PATH", str(PREDICTIONS_LOG)))

# Share of drifted columns at or above which detect_drift() reports drift.
# This is the whole decision policy of the monitoring loop compressed into one
# number, which is exactly why it lives here and not inside the function:
# too low and the system burns a full retrain on noise until its operator
# learns to ignore it; too high and the loop is decorative while recall decays.
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.5"))

# Below this many current rows, detect_drift() declines to answer instead of
# guessing. Measured, not assumed: Evidently 0.7.21 run on a 3-row current
# frame against a 500-row reference reports share=1.0 — every column drifted —
# because a K-S test on three points is noise, not evidence. Without this floor
# the first scheduled monitoring run would fire a retrain off the three
# smoke-test records already in the log.
DRIFT_MIN_ROWS = int(os.getenv("DRIFT_MIN_ROWS", "100"))
