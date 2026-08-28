"""Deploys both flows as Prefect 3 deployments and serves them.

Writing a flow makes it runnable; registering it as a *deployment* makes it
addressable -- schedulable by the server and triggerable by name from outside
the process that defined it. That addressability is what
pipelines/monitoring_pipeline.py depends on: run_deployment() resolves
"training-pipeline/on-demand", which exists only because this file registers it.

This uses the Prefect 3 model, to_deployment() + serve(). Prefect 2's
Deployment.build_from_flow() is deprecated and is deliberately not used.

Running this file gives the phase its three trigger mechanisms: manual (dashboard
or `prefect deployment run "training-pipeline/on-demand"`), scheduled (the daily
cron below), and event-driven (the monitoring flow firing the training
deployment when drift is detected).
"""

from typing import cast

from prefect import serve
from prefect.deployments.runner import RunnerDeployment

from pipelines.monitoring_pipeline import monitoring_pipeline
from pipelines.training_pipeline import training_pipeline

# Module-level rather than inline so tests/test_pipelines.py can import them and
# check them against the string monitoring_pipeline.py fires. That coupling --
# flow name plus deployment name -- is what the closed loop resolves at runtime,
# and a typo in it fails silently: nothing errors, retraining just never runs.
TRAINING_DEPLOYMENT_NAME = "on-demand"
MONITORING_DEPLOYMENT_NAME = "daily"
MONITORING_CRON = "0 6 * * *"  # every day at 06:00


if __name__ == "__main__":
    # to_deployment() carries Prefect's sync/async dispatch union, so it is
    # typed RunnerDeployment | Coroutine[..., RunnerDeployment]; mypy --strict
    # cannot narrow it in a synchronous context and rejects the serve() call
    # without help. The cast states what the sync path actually returns, the
    # same technique src/models/train.py already uses for its TypedDict
    # mismatch. See docs/decisions/0032-strict-typing-in-pipelines.md.
    #
    # Training is deployed WITHOUT a schedule: it exists to be triggered, either
    # by hand or by the drift event.
    training_deploy = cast(
        RunnerDeployment,
        training_pipeline.to_deployment(name=TRAINING_DEPLOYMENT_NAME),
    )

    # Monitoring runs itself, daily at 06:00. This is the clock that makes the
    # loop autonomous rather than something someone remembers to run.
    monitoring_deploy = cast(
        RunnerDeployment,
        monitoring_pipeline.to_deployment(
            name=MONITORING_DEPLOYMENT_NAME, cron=MONITORING_CRON
        ),
    )

    # One process serving both: it runs monitoring on schedule and training on
    # demand. While it is up (alongside the server), the system is self-watching.
    serve(training_deploy, monitoring_deploy)
