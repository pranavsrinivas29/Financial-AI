import numpy as np

from sklearn.dummy import (
    DummyRegressor,
)
from sklearn.impute import (
    SimpleImputer,
)
from sklearn.linear_model import (
    LinearRegression,
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from xgboost import XGBRegressor

from financial_ai.ml import (
    MODEL_FEATURES,
)


def _regression_metrics(
    model,
    X,
    y,
) -> dict:

    prediction = model.predict(
        X
    )

    return {

        "mae":
            mean_absolute_error(
                y,
                prediction,
            ),

        "rmse":
            np.sqrt(
                mean_squared_error(
                    y,
                    prediction,
                )
            ),

        "r2":
            r2_score(
                y,
                prediction,
            ),
    }


def train_volatility_models(
    train_df,
    validation_df,
):

    train_df = train_df.dropna(
        subset=[
            "target_volatility_20d"
        ]
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

    # Baseline
    dummy = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "model",
                DummyRegressor(
                    strategy="mean"
                ),
            ),
        ]
    )

    dummy.fit(
        X_train,
        y_train,
    )

    # Linear model
    linear = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    linear.fit(
        X_train,
        y_train,
    )

    # XGBoost
    xgb = Pipeline(
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
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective=(
                        "reg:squarederror"
                    ),
                    random_state=42,
                ),
            ),
        ]
    )

    xgb.fit(
        X_train,
        y_train,
    )

    results = {

        "dummy":
            _regression_metrics(
                dummy,
                X_validation,
                y_validation,
            ),

        "linear_regression":
            _regression_metrics(
                linear,
                X_validation,
                y_validation,
            ),

        "xgboost":
            _regression_metrics(
                xgb,
                X_validation,
                y_validation,
            ),
    }

    return {
        "dummy": dummy,
        "linear": linear,
        "xgboost": xgb,
        "metrics": results,
    }