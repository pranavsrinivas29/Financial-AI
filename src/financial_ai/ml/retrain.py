import mlflow
import mlflow.sklearn

from sklearn.impute import (
    SimpleImputer,
)
from sklearn.pipeline import (
    Pipeline,
)

from xgboost import (
    XGBClassifier,
    XGBRegressor,
)

from financial_ai.ml import (
    MODEL_FEATURES,
)

from financial_ai.ml.evaluate import (
    classification_metrics,
    regression_metrics,
)

from financial_ai.ml.mlflow_utils import (
    set_direction_experiment,
    set_volatility_experiment,
)

def train_final_direction_model(
    development_df,
    holdout_df,
    best_params: dict,
):

    set_direction_experiment()

    development_df = (
        development_df.dropna(
            subset=[
                "target_direction_20d"
            ]
        )
    )

    holdout_df = (
        holdout_df.dropna(
            subset=[
                "target_direction_20d"
            ]
        )
    )

    X_train = (
        development_df[
            MODEL_FEATURES
        ]
    )

    y_train = (
        development_df[
            "target_direction_20d"
        ]
        .astype(int)
    )

    X_holdout = (
        holdout_df[
            MODEL_FEATURES
        ]
    )

    y_holdout = (
        holdout_df[
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
                    **best_params,
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
            X_holdout,
            y_holdout,
        )
    )

    with mlflow.start_run(
        run_name=(
            "direction_final_holdout"
        )
    ) as run:

        mlflow.log_params(
            best_params
        )

        mlflow.log_param(
            "model_type",
            "XGBClassifier",
        )

        mlflow.log_param(
            "training_start",
            str(
                development_df[
                    "feature_date"
                ].min()
            ),
        )

        mlflow.log_param(
            "training_end",
            str(
                development_df[
                    "feature_date"
                ].max()
            ),
        )

        mlflow.log_param(
            "holdout_start",
            str(
                holdout_df[
                    "feature_date"
                ].min()
            ),
        )

        mlflow.log_param(
            "holdout_end",
            str(
                holdout_df[
                    "feature_date"
                ].max()
            ),
        )

        for (
            metric_name,
            metric_value,
        ) in metrics.items():

            mlflow.log_metric(
                metric_name,
                float(
                    metric_value
                ),
            )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="direction_model",
            registered_model_name=(
                "financial-direction-model"
            ),
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

    return {
        "model": model,
        "metrics": metrics,
        "run_id": run.info.run_id,
    }
    
def train_final_volatility_model(
    development_df,
    holdout_df,
    best_params: dict,
):

    set_volatility_experiment()

    development_df = (
        development_df.dropna(
            subset=[
                "target_volatility_20d"
            ]
        )
    )

    holdout_df = (
        holdout_df.dropna(
            subset=[
                "target_volatility_20d"
            ]
        )
    )

    X_train = (
        development_df[
            MODEL_FEATURES
        ]
    )

    y_train = (
        development_df[
            "target_volatility_20d"
        ]
    )

    X_holdout = (
        holdout_df[
            MODEL_FEATURES
        ]
    )

    y_holdout = (
        holdout_df[
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
                    **best_params,
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
            X_holdout,
            y_holdout,
        )
    )

    with mlflow.start_run(
        run_name=(
            "volatility_final_holdout"
        )
    ) as run:

        mlflow.log_params(
            best_params
        )

        mlflow.log_param(
            "model_type",
            "XGBRegressor",
        )

        for (
            metric_name,
            metric_value,
        ) in metrics.items():

            mlflow.log_metric(
                metric_name,
                float(
                    metric_value
                ),
            )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="volatility_model",
            registered_model_name=(
                "financial-volatility-model"
            ),
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

    return {
        "model": model,
        "metrics": metrics,
        "run_id": run.info.run_id,
    }
    
    
import pandas as pd


def get_matured_training_data(
    dataset,
    inference_date: str,
):

    cutoff = pd.Timestamp(
        inference_date
    )

    matured = dataset[
        dataset[
            "target_end_date"
        ]
        <= cutoff
    ].copy()

    matured = matured.dropna(
        subset=[
            "target_direction_20d",
            "target_volatility_20d",
        ]
    )

    if matured.empty:

        raise ValueError(
            "No matured labelled data "
            "available before inference."
        )

    assert (
        matured[
            "target_end_date"
        ].max()
        <= cutoff
    )

    return (
        matured
        .sort_values(
            "feature_date"
        )
        .reset_index(drop=True)
    )
    
def retrain_direction_for_inference(
    dataset,
    inference_date: str,
    approved_params: dict,
):

    matured = (
        get_matured_training_data(
            dataset,
            inference_date,
        )
    )

    X = matured[
        MODEL_FEATURES
    ]

    y = (
        matured[
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
                    **approved_params,
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
        X,
        y,
    )

    set_direction_experiment()

    with mlflow.start_run(
        run_name=(
            "direction_retrain_"
            f"{inference_date}"
        )
    ):

        mlflow.log_params(
            approved_params
        )

        mlflow.log_param(
            "inference_date",
            inference_date,
        )

        mlflow.log_param(
            "training_rows",
            len(matured),
        )

        mlflow.log_param(
            "training_cutoff",
            str(
                matured[
                    "target_end_date"
                ].max()
            ),
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="direction_model",
            registered_model_name=(
                "financial-direction-model"
            ),
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

    return model

def retrain_volatility_for_inference(
    dataset,
    inference_date: str,
    approved_params: dict,
):

    matured = (
        get_matured_training_data(
            dataset,
            inference_date,
        )
    )

    X = matured[
        MODEL_FEATURES
    ]

    y = matured[
        "target_volatility_20d"
    ]

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
                    **approved_params,
                    objective=(
                        "reg:squarederror"
                    ),
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        X,
        y,
    )

    set_volatility_experiment()

    with mlflow.start_run(
        run_name=(
            "volatility_retrain_"
            f"{inference_date}"
        )
    ):

        mlflow.log_params(
            approved_params
        )

        mlflow.log_param(
            "inference_date",
            inference_date,
        )

        mlflow.log_param(
            "training_rows",
            len(matured),
        )

        mlflow.log_param(
            "training_cutoff",
            str(
                matured[
                    "target_end_date"
                ].max()
            ),
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="volatility_model",
            registered_model_name=(
                "financial-volatility-model"
            ),
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

    return model