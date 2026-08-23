"""Pydantic schemas that type the API boundary.

This module declares data shape and nothing else: no model loading, no
inference, no logging. It is the contract the outside world is held to on the
way in, and the contract this service holds itself to on the way out.

Validating here is what makes the boundary trustworthy — a malformed
transaction is rejected by FastAPI with a 422 before a single line of
inference code runs, which is the same fail-loudly principle Pandera enforces
on the training side (see docs/decisions/0010-pandera-strict-lazy.md).
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, create_model

# V1..V28 are anonymised PCA components: same type, same constraints, no
# individual meaning to document. Generating them beats 28 identical lines,
# and keeps the count in one place if the feature set ever changes.
_PCA_FEATURES: dict[str, Any] = {
    f"V{i}": (float, Field(description=f"Anonymised PCA component V{i}"))
    for i in range(1, 29)
}

# Time and Amount are the only interpretable columns, so they are declared
# explicitly with their own constraints. ge=0 turns "an amount cannot be
# negative" from an assumption into a guarantee the framework enforces.
Transaction = create_model(
    "Transaction",
    Time=(
        float,
        Field(ge=0, description="Seconds elapsed since the first transaction"),
    ),
    Amount=(float, Field(ge=0, description="Transaction amount")),
    **_PCA_FEATURES,
)
"""One raw transaction, exactly as the packaged model expects it.

The field set matches the registered input signature: the columns of
data/raw/creditcard.csv minus the target. Raw, never pre-scaled — the fitted
preprocessor lives inside the model artifact.
"""


class PredictionResponse(BaseModel):
    """The scored result of one transaction.

    Carries the probability alongside the decision because the packaged model
    returns both (docs/decisions/0015-packaged-model-contract.md): the label
    drives the action, the probability is what makes triage, auditing and the
    Phase 8 drift inputs possible. Dropping it would discard information the
    model has already computed.
    """

    # model_version starts with "model_", a namespace Pydantic v2 reserves;
    # without this the field is legal but emits a warning on every import.
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float = Field(
        ge=0, le=1, description="Estimated probability that the transaction is fraud"
    )
    is_fraud: int = Field(
        description="1 if the probability reaches the model's threshold, else 0"
    )
    model_version: str = Field(
        description="Registry version that produced this prediction"
    )
