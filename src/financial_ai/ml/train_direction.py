import joblib

from pathlib import Path

from sklearn.compose import (
    ColumnTransformer,
)
from sklearn.dummy import (
    DummyClassifier,
)
from sklearn.impute import (
    SimpleImputer,
)
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
)

from xgboost import XGBClassifier

from financial_ai.ml import (
    MODEL_FEATURES,
)


def _classification_metrics(
    model,
    X,
    y,
) -> dict:

    prediction = model.predict(
        X
    )

    metrics = {

        "accuracy":
            accuracy_score(
                y,
                prediction,
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y,
                prediction,
            ),

        "precision":
            precision_score(
                y,
                prediction,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y,
                prediction,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y,
                prediction,
                zero_division=0,
            ),
    }

    if hasattr(
        model,
        "predict_proba",
    ):

        probability = (
            model.predict_proba(X)[
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


def train_direction_models(
    train_df,
    validation_df,
):

    train_df = train_df.dropna(
        subset=[
            "target_direction_20d"
        ]
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

    # ---------------------------
    # Baseline
    # ---------------------------

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
                DummyClassifier(
                    strategy="prior"
                ),
            ),
        ]
    )

    dummy.fit(
        X_train,
        y_train,
    )

    # ---------------------------
    # Logistic regression
    # ---------------------------

    logistic = Pipeline(
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
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    logistic.fit(
        X_train,
        y_train,
    )

    # ---------------------------
    # XGBoost
    # ---------------------------

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
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective=(
                        "binary:logistic"
                    ),
                    eval_metric="logloss",
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
            _classification_metrics(
                dummy,
                X_validation,
                y_validation,
            ),

        "logistic_regression":
            _classification_metrics(
                logistic,
                X_validation,
                y_validation,
            ),

        "xgboost":
            _classification_metrics(
                xgb,
                X_validation,
                y_validation,
            ),
    }

    return {
        "dummy": dummy,
        "logistic": logistic,
        "xgboost": xgb,
        "metrics": results,
    }