from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from financial_ai.monitoring.evidently_monitor import (
    generate_drift_report,
    missing_feature_rate,
)


def monitor_predictions(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    tracking_uri: str = "http://127.0.0.1:5000",
):
    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        "financial-ai-monitoring"
    )

    reports = generate_drift_report(
        reference_data=reference_data,
        current_data=current_data,
    )

    metrics = {
        "rows_received": float(
            len(current_data)
        ),
        "missing_feature_rate":
            missing_feature_rate(
                current_data
            ),
    }

    # Direction model monitoring
    if {
        "actual_direction",
        "direction_prediction",
    }.issubset(current_data.columns):

        accuracy = (
            current_data[
                "actual_direction"
            ]
            == current_data[
                "direction_prediction"
            ]
        ).mean()

        metrics[
            "direction_accuracy"
        ] = float(accuracy)

    # Volatility model monitoring
    if {
        "actual_volatility",
        "volatility_prediction",
    }.issubset(current_data.columns):

        error = np.abs(
            current_data[
                "actual_volatility"
            ]
            - current_data[
                "volatility_prediction"
            ]
        )

        metrics[
            "volatility_mae"
        ] = float(
            error.mean()
        )

    if (
        "direction_confidence"
        in current_data.columns
    ):
        metrics[
            "mean_direction_confidence"
        ] = float(
            current_data[
                "direction_confidence"
            ].mean()
        )

    with mlflow.start_run(
        run_name="inference-monitoring"
    ):

        mlflow.log_metrics(
            metrics
        )

        mlflow.log_params(
            {
                "direction_model_version": 7,
                "volatility_model_version": 7,
            }
        )

        mlflow.log_artifact(
            reports["html"],
            artifact_path="evidently",
        )

        mlflow.log_artifact(
            reports["json"],
            artifact_path="evidently",
        )

    return metrics