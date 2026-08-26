import os
from pathlib import Path

import mlflow


DIRECTION_EXPERIMENT = (
    "financial-ai-direction"
)

VOLATILITY_EXPERIMENT = (
    "financial-ai-volatility"
)


def configure_mlflow():

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://127.0.0.1:5000",
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )


def set_direction_experiment():

    configure_mlflow()

    mlflow.set_experiment(
        DIRECTION_EXPERIMENT
    )


def set_volatility_experiment():

    configure_mlflow()

    mlflow.set_experiment(
        VOLATILITY_EXPERIMENT
    )