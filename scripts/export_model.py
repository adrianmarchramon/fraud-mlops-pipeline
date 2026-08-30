"""Exports the @production model into deploy/ so it can ride inside the image.

Implementation phase: Phase 9 - Deployment (Step 1).

The public deployment has no MLflow server to ask, so the model has to travel
inside the container. This is the one command that puts it there, and it exists
as a script rather than a remembered incantation because the export has to be
repeated: the bundled artifact is frozen at build time, so every promotion that
should reach the public demo needs a re-export, a commit and a redeploy.

Why the artifact lands in Git rather than being fetched during the build: the
image that gets deployed is the one .github/workflows/cd.yml publishes from a
GitHub runner, and that runner has no dataset (the DVC remote is local,
docs/decisions/0005-dvc-local-remote.md) and no Model Registry to resolve an
alias against. Anything the image is to contain must already be in the
repository when CI checks it out. See docs/decisions/0042-bundled-model.md.

This is a maintenance entry point, not library code. It imports from src/ for
paths and names only, holds no business logic, and prints rather than logs --
the same licence scripts/simulate_drift.py takes.

Run it with a Registry reachable (the local SQLite store is enough):

    make export-model
"""

import shutil
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.artifacts
import mlflow.pyfunc
import pandas as pd

from src.api.predict import BUNDLED_METADATA_FILE
from src.config import DEPLOY_MODEL_DIR, MLFLOW_TRACKING_URI, MODEL_NAME
from src.models.register import PRODUCTION_ALIAS

# The file MLflow writes at the root of any saved model. Used as the proof that
# a directory about to be deleted is an export of ours and not something else
# that happens to sit at the same path.
MODEL_MARKER = "MLmodel"


def _clear_destination(destination: Path) -> None:
    """Empty the destination, refusing anything that is not a previous export.

    Args:
        destination: the directory the export is about to be written into.

    Raises:
        SystemExit: if the path exists but carries no MLmodel marker. A stale
            export must be replaced wholesale -- leaving old files behind would
            produce a directory that is half one version and half another --
            but that is never a licence to delete an arbitrary path.
    """
    if not destination.exists():
        return

    if not (destination / MODEL_MARKER).exists():
        print(
            f"Refusing to clear {destination}: it exists but holds no "
            f"{MODEL_MARKER}, so it is not a previous export.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    shutil.rmtree(destination)


def _verify(destination: Path) -> None:
    """Load the export back and score a row, before anyone commits it.

    A truncated or partial export is not visibly different from a good one --
    the directory listing looks the same -- and the next place it would surface
    is a container that starts, reports no_model, and answers 503. Loading it
    here turns that into a failure at export time.

    Args:
        destination: the directory just written.

    Raises:
        SystemExit: if the export cannot be loaded or scored.
    """
    row = {
        "Time": 0.0,
        **{f"V{index}": 0.0 for index in range(1, 29)},
        "Amount": 149.62,
    }

    try:
        model = mlflow.pyfunc.load_model(str(destination))
        scored = model.predict(pd.DataFrame([row]))
    except Exception as exc:
        print(
            f"The export at {destination} could not be loaded: {exc}", file=sys.stderr
        )
        raise SystemExit(1) from exc

    prediction = scored.to_dict("records")[0]
    print(f"Verified: loaded back and scored one transaction -> {prediction}")

    # Loading the bundle imports the snapshot of src/ it carries, which leaves
    # __pycache__ directories inside the artifact. They are git-ignored, so
    # they would never be committed -- but they make the exported tree differ
    # from what MLflow wrote, and the next person to checksum it would have no
    # idea why. Removed here so the directory on disk IS the artifact.
    for cache in destination.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def main() -> None:
    """Export the @production artifact into DEPLOY_MODEL_DIR and verify it.

    Raises:
        SystemExit: with status 1 if the alias cannot be resolved, the
            destination is not ours to clear, or the export does not load.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    uri = f"models:/{MODEL_NAME}@{PRODUCTION_ALIAS}"
    print(f"Exporting {uri} from {MLFLOW_TRACKING_URI}")

    _clear_destination(DEPLOY_MODEL_DIR)
    DEPLOY_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    try:
        # download_artifacts() is untyped, so the result is annotated rather
        # than allowed to widen this function's locals to Any.
        exported: Any = mlflow.artifacts.download_artifacts(
            artifact_uri=uri, dst_path=str(DEPLOY_MODEL_DIR)
        )
    except Exception as exc:
        print(f"Could not export {uri}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    files = sorted(path for path in DEPLOY_MODEL_DIR.rglob("*") if path.is_file())
    total = sum(path.stat().st_size for path in files)
    largest = max(files, key=lambda path: path.stat().st_size)

    print(f"Exported to {exported}")
    print(f"  {len(files)} files, {total / 1024:.0f} KB total")
    print(
        f"  largest: {largest.relative_to(DEPLOY_MODEL_DIR)} "
        f"({largest.stat().st_size / 1024:.0f} KB)"
    )
    metadata = (DEPLOY_MODEL_DIR / BUNDLED_METADATA_FILE).read_text().strip()
    print(f"  {BUNDLED_METADATA_FILE}: {metadata}")

    _verify(DEPLOY_MODEL_DIR)
    print("Commit deploy/model/ for the change to reach the deployed image.")


if __name__ == "__main__":
    main()
