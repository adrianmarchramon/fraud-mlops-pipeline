"""Injects a deliberately shifted batch of transactions into the running API.

Implementation phase: Phase 8 - Monitoring and Drift (Step 7).

This is the trigger for the phase's key test. It sends transactions whose
feature distribution is visibly unlike the one the model was trained on, over
real HTTP to the real endpoint, so the API validates them, scores them and
appends them to config.PREDICTIONS_LOG exactly as it would production traffic.
The monitoring flow then reads that log, measures the shift against the frozen
reference, and fires the retraining deployment on its own.

Writing the log file directly would have been simpler and would have proved
nothing: the point is to exercise the path built in Phase 4 -- Pydantic
validation, model scoring, JSONL append -- not to fake its output.

This is a demo client and nothing else. It imports no project code, holds no
business logic, and never imports prefect; print() is appropriate here in a way
it never is under src/ or pipelines/.

Run it with the API up (`make serve`):

    uv run python -m scripts.simulate_drift
"""

import os

import numpy as np
import requests

# Overridable so the same script can drive a containerised API on another port
# without editing it, but hardcoded by default: this is a demo entry point, not
# library code, so the URL living here rather than in src/config.py is
# deliberate -- src/ must not grow configuration that only a script consumes.
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

N_TRANSACTIONS = 300
SEED = 0

# --- Shift calibration -----------------------------------------------------
# Every figure below is justified against the measured statistics of the
# reference artifact (data/monitoring/reference.parquet, 5000 rows), not copied
# from a tutorial:
#
#   V1..V28  pooled mean -0.000, pooled std 1.069
#   Amount   mean 87.858, std 238.939, observed range [0, 4861.64]
#   Time     range [22, 172783]
#
# V_SHIFT_MEAN of 3.0 is roughly +2.8 sigma on the pooled V spread: far outside
# the training distribution, but the same order of magnitude, so the drift is
# unambiguous without being a nonsense input the model could never receive.
V_SHIFT_MEAN = 3.0
V_SHIFT_STD = 2.0

# Amounts an order of magnitude above the mean (+12 sigma) yet still under the
# largest amount the reference actually contains -- anomalously high, not
# impossible. Both bounds are non-negative because the Transaction schema
# constrains Amount with ge=0 and would answer 422 otherwise.
AMOUNT_MIN = 1_000.0
AMOUNT_MAX = 5_000.0

# Time spans roughly the reference's own range ([22, 172783]), so this bound
# was originally chosen to leave one column undrifted. Measured, it does not:
# both live runs reported a share of 1.0000, all 30 columns, Time included.
# The reasoning behind the original choice was wrong in a way worth keeping
# written down -- the drift test compares whole DISTRIBUTIONS, not range or
# mean, and the real Time column is bimodal (two days of transaction cycles)
# while a uniform draw is flat. Matching a mean does not make a distribution
# the same distribution. (The test Evidently actually applied was the normed
# Wasserstein distance, not K-S as earlier comments claimed -- see
# docs/decisions/0038-evidently-dependency-and-api.md.) See also
# docs/decisions/0041-phase-8-closure.md.
TIME_MIN = 0.0
TIME_MAX = 200_000.0


def make_drifted_transactions(
    n: int = N_TRANSACTIONS, seed: int = SEED
) -> list[dict[str, float]]:
    """Build a batch whose distribution is deliberately unlike the training set.

    Args:
        n: how many transactions to generate.
        seed: fixes the draw, so a rerun sends the identical batch and the
            demonstration is reproducible rather than merely repeatable.

    Returns:
        Transactions shaped exactly like the API's Transaction schema: Time,
        V1..V28 and Amount, all floats, Time and Amount non-negative.
    """
    rng = np.random.default_rng(seed)
    transactions: list[dict[str, float]] = []

    for _ in range(n):
        transaction: dict[str, float] = {"Time": float(rng.uniform(TIME_MIN, TIME_MAX))}
        for index in range(1, 29):
            transaction[f"V{index}"] = float(rng.normal(V_SHIFT_MEAN, V_SHIFT_STD))
        transaction["Amount"] = float(rng.uniform(AMOUNT_MIN, AMOUNT_MAX))
        transactions.append(transaction)

    return transactions


def main() -> None:
    """Send the batch to the API and report exactly what happened.

    Failures are counted and printed rather than swallowed: a demonstration
    that silently dropped half its requests would still print a cheerful
    summary while the drift signal it was supposed to create never arrived.

    Raises:
        SystemExit: with status 1 if any request failed, so a broken run cannot
            be mistaken for a successful one.
    """
    transactions = make_drifted_transactions()
    print(f"Sending {len(transactions)} drifted transactions to {API_URL}")

    accepted = 0
    failures: list[str] = []

    for position, transaction in enumerate(transactions, start=1):
        try:
            response = requests.post(API_URL, json=transaction, timeout=10)
        except requests.RequestException as exc:
            failures.append(f"#{position}: {type(exc).__name__}: {exc}")
            continue

        if response.status_code == 200:
            accepted += 1
        else:
            failures.append(
                f"#{position}: HTTP {response.status_code} {response.text[:120]}"
            )

    print(f"Accepted: {accepted}/{len(transactions)}")

    if failures:
        print(f"Failed:   {len(failures)}")
        for failure in failures[:5]:
            print(f"  {failure}")
        if len(failures) > 5:
            print(f"  ... and {len(failures) - 5} more")
        raise SystemExit(1)

    print("All transactions accepted and logged. Run the monitoring pipeline next.")


if __name__ == "__main__":
    main()
