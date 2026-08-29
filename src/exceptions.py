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
    target column.
    """


class ModelEvaluationError(FraudPipelineError):
    """Raised when model evaluation (cross-validation, threshold search) fails.

    Kept distinct from ModelTrainingError because evaluating a model and
    training one are different responsibilities, in different modules — the
    same criterion that separates DataValidationError from
    DataPreprocessingError.
    """


class ModelRegistrationError(FraudPipelineError):
    """Raised when packaging, registering, or promoting a model fails.

    Covers the whole src/models/register.py domain — building the packaged
    inference artifact, reading the versioned decision threshold, and (from
    Steps 4-5) registering versions and moving the production alias — on the
    same one-exception-per-module criterion that makes ModelTrainingError
    cover all of train.py.
    """


class PredictionError(FraudPipelineError):
    """Raised when the API cannot load the production model or score a request.

    Covers the whole src/api/predict.py domain — resolving the @production
    alias, loading the artifact, and scoring a transaction — on the same
    one-exception-per-module criterion that makes ModelRegistrationError cover
    all of register.py.

    It stops at the inference boundary: src/api/main.py catches it and decides
    which HTTP status the caller sees. Keeping that translation in the route
    layer is what lets predict.py stay free of any FastAPI import.
    """


class DriftDetectionError(FraudPipelineError):
    """Raised when drift cannot be measured.

    Covers the whole src/monitoring/ domain — building the reference
    distribution, reading the reference or current datasets, and extracting a
    verdict from the Evidently result.

    It means "the question could not be answered", never "the answer is no
    drift". That distinction is the reason this exists rather than returning
    False on failure: a monitoring loop that silently reports "all clear" when
    its own inputs are broken is worse than one that is switched off, because
    it manufactures confidence nobody checked.
    """
