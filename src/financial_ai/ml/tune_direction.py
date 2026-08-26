import itertools
import statistics

import mlflow
import random

from sklearn.impute import (
    SimpleImputer,
)
from sklearn.pipeline import (
    Pipeline,
)

from xgboost import (
    XGBClassifier,
)

from financial_ai.ml import (
    MODEL_FEATURES,
)

from financial_ai.ml.evaluate import (
    classification_metrics,
)

from financial_ai.ml.mlflow_utils import (
    set_direction_experiment,
)

from financial_ai.ml.split import (
    walk_forward_splits,
)


DIRECTION_PARAM_GRID = {

    "n_estimators": [
        200,
        400,
    ],

    "max_depth": [
        2,
        3,
        4,
    ],

    "learning_rate": [
        0.01,
        0.03,
        0.05,
    ],

    "subsample": [
        0.8,
        1.0,
    ],

    "colsample_bytree": [
        0.8,
        1.0,
    ],

    "min_child_weight": [
        1,
        5,
    ],

    "reg_alpha": [
        0.0,
        0.1,
    ],

    "reg_lambda": [
        1.0,
        5.0,
    ],
}

def _parameter_combinations():

    keys = list(
        DIRECTION_PARAM_GRID.keys()
    )

    values = [
        DIRECTION_PARAM_GRID[key]
        for key in keys
    ]

    combinations = []

    for values_tuple in itertools.product(
        *values
    ):

        combinations.append(
            dict(
                zip(
                    keys,
                    values_tuple,
                )
            )
        )

    return combinations

def tune_direction_model(
    development_df,
    max_trials: int = 30,
):

    set_direction_experiment()

    folds = walk_forward_splits(
        development_df,
        n_splits=4,
    )

    parameter_sets = (
        _parameter_combinations()
    )

    # Keep project runtime manageable.
    #parameter_sets = (
    #    parameter_sets[
    #        :max_trials
    #    ]
    #)

    random.seed(42)

    if len(parameter_sets) > max_trials:

        parameter_sets = random.sample(
            parameter_sets,
            max_trials,
        )
    
    best_params = None

    best_score = float(
        "-inf"
    )

    all_results = []

    for trial_number, params in enumerate(
        parameter_sets,
        start=1,
    ):

        fold_scores = []

        with mlflow.start_run(
            run_name=(
                f"xgb_tuning_"
                f"{trial_number}"
            )
        ):

            mlflow.log_params(
                params
            )

            mlflow.log_param(
                "model_type",
                "XGBClassifier",
            )

            mlflow.log_param(
                "forecast_horizon",
                20,
            )

            for (
                fold_number,
                (
                    train_df,
                    validation_df,
                ),
            ) in enumerate(
                folds,
                start=1,
            ):

                train_df = (
                    train_df.dropna(
                        subset=[
                            "target_direction_20d"
                        ]
                    )
                )

                validation_df = (
                    validation_df.dropna(
                        subset=[
                            "target_direction_20d"
                        ]
                    )
                )

                X_train = train_df[
                    MODEL_FEATURES
                ]

                y_train = (
                    train_df[
                        "target_direction_20d"
                    ]
                    .astype(int)
                )

                X_validation = (
                    validation_df[
                        MODEL_FEATURES
                    ]
                )

                y_validation = (
                    validation_df[
                        "target_direction_20d"
                    ]
                    .astype(int)
                )

                model = Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),

                        (
                            "model",
                            XGBClassifier(
                                **params,
                                objective=(
                                    "binary:logistic"
                                ),
                                eval_metric=(
                                    "logloss"
                                ),
                                random_state=42,
                            ),
                        ),
                    ]
                )

                model.fit(
                    X_train,
                    y_train,
                )

                metrics = (
                    classification_metrics(
                        model,
                        X_validation,
                        y_validation,
                    )
                )

                score = metrics.get(
                    "roc_auc",
                    0.0,
                )

                fold_scores.append(
                    score
                )

                for (
                    metric_name,
                    metric_value,
                ) in metrics.items():

                    mlflow.log_metric(
                        (
                            f"fold_"
                            f"{fold_number}_"
                            f"{metric_name}"
                        ),
                        float(
                            metric_value
                        ),
                    )

            mean_auc = statistics.mean(
                fold_scores
            )

            std_auc = (
                statistics.pstdev(
                    fold_scores
                )
                if len(
                    fold_scores
                ) > 1
                else 0.0
            )

            mlflow.log_metric(
                "mean_roc_auc",
                mean_auc,
            )

            mlflow.log_metric(
                "std_roc_auc",
                std_auc,
            )

            all_results.append(
                {
                    "params": params,
                    "mean_roc_auc":
                        mean_auc,
                    "std_roc_auc":
                        std_auc,
                }
            )

            if mean_auc > best_score:

                best_score = (
                    mean_auc
                )

                best_params = (
                    params.copy()
                )

    return {
        "best_params":
            best_params,

        "best_mean_roc_auc":
            best_score,

        "all_results":
            all_results,
    }