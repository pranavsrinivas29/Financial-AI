import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)


def classification_metrics(
    model,
    X,
    y,
) -> dict:

    predictions = model.predict(
        X
    )

    metrics = {
        "accuracy":
            accuracy_score(
                y,
                predictions,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                predictions,
            ),

        "precision":
            precision_score(
                y,
                predictions,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y,
                predictions,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y,
                predictions,
                zero_division=0,
            ),
    }

    if hasattr(
        model,
        "predict_proba",
    ):

        probability = (
            model
            .predict_proba(X)[
                :, 1
            ]
        )

        if len(set(y)) > 1:

            metrics[
                "roc_auc"
            ] = roc_auc_score(
                y,
                probability,
            )

        metrics[
            "brier_score"
        ] = brier_score_loss(
            y,
            probability,
        )

        metrics[
            "log_loss"
        ] = log_loss(
            y,
            probability,
        )

    return metrics


def regression_metrics(
    model,
    X,
    y,
) -> dict:

    predictions = model.predict(
        X
    )

    return {
        "mae":
            mean_absolute_error(
                y,
                predictions,
            ),

        "rmse":
            np.sqrt(
                mean_squared_error(
                    y,
                    predictions,
                )
            ),

        "r2":
            r2_score(
                y,
                predictions,
            ),
    }