import itertools
import random
import statistics

import mlflow

from sklearn.impute import (
    SimpleImputer,
)

from sklearn.pipeline import (
    Pipeline,
)

from xgboost import (
    XGBRegressor,
)

from financial_ai.ml import (
    MODEL_FEATURES,
)

from financial_ai.ml.evaluate import (
    regression_metrics,
)

from financial_ai.ml.mlflow_utils import (
    set_volatility_experiment,
)

from financial_ai.ml.split import (
    walk_forward_splits,
)


VOLATILITY_PARAM_GRID = {

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
        VOLATILITY_PARAM_GRID.keys()
    )

    values = [
        VOLATILITY_PARAM_GRID[key]
        for key in keys
    ]

    return [
        dict(
            zip(
                keys,
                combination,
            )
        )
        for combination
        in itertools.product(
            *values
        )
    ]


def tune_volatility_model(
    development_df,
    max_trials: int = 30,
):

    set_volatility_experiment()

    folds = walk_forward_splits(
        development_df,
        n_splits=4,
    )

    parameter_sets = (
        _parameter_combinations()
    )

    random.seed(42)

    if len(parameter_sets) > max_trials:

        parameter_sets = random.sample(
            parameter_sets,
            max_trials,
        )

    best_params = None

    best_rmse = float(
        "inf"
    )

    all_results = []

    for trial_number, params in enumerate(
        parameter_sets,
        start=1,
    ):

        fold_rmse = []

        with mlflow.start_run(
            run_name=(
                f"xgb_volatility_"
                f"{trial_number}"
            )
        ):

            mlflow.log_params(
                params
            )

            mlflow.log_param(
                "model_type",
                "XGBRegressor",
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
                            "target_volatility_20d"
                        ]
                    )
                )

                validation_df = (
                    validation_df.dropna(
                        subset=[
                            "target_volatility_20d"
                        ]
                    )
                )

                X_train = train_df[
                    MODEL_FEATURES
                ]

                y_train = train_df[
                    "target_volatility_20d"
                ]

                X_validation = (
                    validation_df[
                        MODEL_FEATURES
                    ]
                )

                y_validation = (
                    validation_df[
                        "target_volatility_20d"
                    ]
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
                            XGBRegressor(
                                **params,
                                objective=(
                                    "reg:squarederror"
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
                    regression_metrics(
                        model,
                        X_validation,
                        y_validation,
                    )
                )

                fold_rmse.append(
                    metrics[
                        "rmse"
                    ]
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

            mean_rmse = (
                statistics.mean(
                    fold_rmse
                )
            )

            std_rmse = (
                statistics.pstdev(
                    fold_rmse
                )
                if len(
                    fold_rmse
                ) > 1
                else 0.0
            )

            mlflow.log_metric(
                "mean_rmse",
                mean_rmse,
            )

            mlflow.log_metric(
                "std_rmse",
                std_rmse,
            )

            all_results.append(
                {
                    "params": params,
                    "mean_rmse":
                        mean_rmse,
                    "std_rmse":
                        std_rmse,
                }
            )

            if (
                mean_rmse
                < best_rmse
            ):

                best_rmse = (
                    mean_rmse
                )

                best_params = (
                    params.copy()
                )

    return {
        "best_params":
            best_params,

        "best_mean_rmse":
            best_rmse,

        "all_results":
            all_results,
    }