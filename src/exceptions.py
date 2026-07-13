"""Application-specific exception hierarchy for the fraud detection pipeline.

Every exception raised deliberately by this project's business logic
inherits from FraudPipelineError, never from a bare Exception, so that
calling code (API layer, orchestration, tests) can catch pipeline
failures distinctly from unexpected bugs.
"""


class FraudPipelineError(Exception):
    """Base exception for all deliberate, application-level failures."""


class DataIngestionError(FraudPipelineError):
    """Raised when the raw dataset cannot be loaded or parsed."""


class DataValidationError(FraudPipelineError):
    """Raised when raw data fails the Pandera quality contract."""


class DataPreprocessingError(FraudPipelineError):
    """Raised when preprocessing config cannot be loaded or the transform fails.

    Covers a missing or malformed params.yaml (including a missing
    'preprocess' key); data-contract failures keep their own
    DataValidationError.
    """


class ModelTrainingError(FraudPipelineError):
    """Raised when loading training data or training the model fails.

    Covers a missing or unreadable processed split and the absence of the
    target column; downstream training/evaluation failures reuse it too.
    """
