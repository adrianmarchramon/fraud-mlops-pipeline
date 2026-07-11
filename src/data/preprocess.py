"""Preprocessing and feature engineering: validated raw data to features.

Owns the single most consequential correctness boundary in the pipeline
so far: the scaler is fit exclusively on the training split and only ever
applied (never re-fit) to the test split, so no test-set information
leaks into the transformation.
"""

import logging
from typing import TypedDict

import joblib
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import (
    PREPROCESSOR_PATH,
    PROCESSED_DIR,
    PROJECT_ROOT,
    TARGET,
    TEST_PATH,
    TRAIN_PATH,
)
from src.data.ingest import load_raw_data
from src.data.validate import validate_processed_data
from src.exceptions import DataPreprocessingError

logger = logging.getLogger(__name__)


class PreprocessParams(TypedDict):
    """Shape of the `preprocess:` block read from params.yaml."""

    test_size: float
    random_state: int
    scale_columns: list[str]


def load_params() -> PreprocessParams:
    """Load the versioned preprocessing parameters from params.yaml.

    Returns:
        The `preprocess` block of params.yaml.

    Raises:
        DataPreprocessingError: if params.yaml is missing, is not valid
            YAML, or has no top-level `preprocess` key.
    """
    params_path = PROJECT_ROOT / "params.yaml"
    try:
        with open(params_path) as f:
            all_params = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise DataPreprocessingError(f"params.yaml not found at {params_path}") from exc
    except yaml.YAMLError as exc:
        raise DataPreprocessingError(
            f"params.yaml is not valid YAML: {params_path}"
        ) from exc

    try:
        return all_params["preprocess"]
    except (KeyError, TypeError) as exc:
        raise DataPreprocessingError(
            "params.yaml has no top-level 'preprocess' key"
        ) from exc


def build_preprocessor(scale_columns: list[str]) -> ColumnTransformer:
    """Build the unfitted transformer: scale given columns, pass the rest.

    Args:
        scale_columns: names of the columns to standardize. In this
            dataset that means Time and Amount only — V1-V28 are already
            PCA components and are deliberately excluded from scaling.

    Returns:
        A ColumnTransformer configured to output pandas DataFrames.
    """
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), scale_columns)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def preprocess() -> None:
    """Run the full preprocessing stage: split, scale, validate, persist.

    This is the single most leakage-sensitive function in the pipeline:
    the ColumnTransformer is fit ONLY on X_train and only ever
    .transform()-ed — never re-fit — on X_test.

    Raises:
        DataPreprocessingError: if the preprocessing parameters cannot
            be loaded.
        DataValidationError: if either resulting split fails the
            post-preprocessing quality contract. Propagated as-is from
            validate_processed_data rather than translated, since the
            failure is genuinely a data-contract violation, not a
            preprocessing failure.
    """
    params = load_params()
    logger.info(
        "Starting preprocessing: test_size=%s, random_state=%s, scale_columns=%s",
        params["test_size"],
        params["random_state"],
        params["scale_columns"],
    )

    df = load_raw_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Stratified split: preserves the fraud ratio in both splits despite
    # the dataset's extreme class imbalance.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=y,
    )

    preprocessor = build_preprocessor(params["scale_columns"])

    # fit ONLY on train; transform (never re-fit) on test — this is the
    # exact boundary where data leakage would otherwise be introduced.
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    train_df = X_train_t.assign(**{TARGET: y_train.values})
    test_df = X_test_t.assign(**{TARGET: y_test.values})

    # Validate every boundary, not just the input: fail fast if the
    # transformation produced something with the wrong shape or type.
    validate_processed_data(train_df, split_name="train")
    validate_processed_data(test_df, split_name="test")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(TRAIN_PATH)
    test_df.to_parquet(TEST_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    logger.info(
        "Preprocessing completed: train=%d rows (fraud rate %.4f%%), "
        "test=%d rows (fraud rate %.4f%%)",
        len(train_df),
        train_df[TARGET].mean() * 100,
        len(test_df),
        test_df[TARGET].mean() * 100,
    )


if __name__ == "__main__":
    preprocess()
