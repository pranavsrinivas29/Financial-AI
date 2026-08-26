import pandas as pd

from financial_ai.ml import MODEL_FEATURES
from financial_ai.ml.retrain import (
    retrain_direction_for_inference,
    retrain_volatility_for_inference,
)


def get_latest_feature_row(
    feature_dataset: pd.DataFrame,
    inference_date: str,
) -> pd.DataFrame:
    """
    Return the latest available feature row on or before
    the requested inference date.

    This is the row used for prediction.
    """

    cutoff = pd.Timestamp(
        inference_date
    )

    df = feature_dataset.copy()

    df["feature_date"] = pd.to_datetime(
        df["feature_date"]
    )

    eligible = df[
        df["feature_date"] <= cutoff
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No feature row is available "
            f"on or before {inference_date}."
        )

    latest = (
        eligible
        .sort_values(
            "feature_date"
        )
        .iloc[[-1]]
        .copy()
    )

    assert (
        latest["feature_date"].iloc[0]
        <= cutoff
    )

    return latest


def predict_financial_outlook(
    dataset: pd.DataFrame,
    inference_date: str,
    direction_params: dict,
    volatility_params: dict,
) -> dict:
    """
    Retrain the predictive models using all matured
    labels available at inference_date and generate
    the current ML outlook.

    NLP sentiment is intentionally NOT used here.
    """

    # -------------------------------------------
    # 1. Retrain using matured historical labels
    # -------------------------------------------

    direction_model = (
        retrain_direction_for_inference(
            dataset=dataset,
            inference_date=inference_date,
            approved_params=direction_params,
        )
    )

    volatility_model = (
        retrain_volatility_for_inference(
            dataset=dataset,
            inference_date=inference_date,
            approved_params=volatility_params,
        )
    )

    # -------------------------------------------
    # 2. Latest point-in-time feature row
    # -------------------------------------------

    feature_row = get_latest_feature_row(
        feature_dataset=dataset,
        inference_date=inference_date,
    )

    X = feature_row[
        MODEL_FEATURES
    ]

    # -------------------------------------------
    # 3. Direction prediction
    # -------------------------------------------

    direction_probability = float(
        direction_model
        .predict_proba(X)[0, 1]
    )

    direction_class = (
        "positive"
        if direction_probability >= 0.50
        else "negative"
    )

    # -------------------------------------------
    # 4. Volatility prediction
    # -------------------------------------------

    expected_volatility = float(
        volatility_model.predict(X)[0]
    )

    # -------------------------------------------
    # 5. Training maturity metadata
    # -------------------------------------------

    cutoff = pd.Timestamp(
        inference_date
    )

    matured = dataset[
        pd.to_datetime(
            dataset["target_end_date"]
        )
        <= cutoff
    ].copy()

    training_cutoff = (
        matured[
            "target_end_date"
        ].max()
        if not matured.empty
        else None
    )

    return {
        "ticker":
            feature_row[
                "ticker"
            ].iloc[0],

        "as_of_date":
            inference_date,

        "feature_date":
            str(
                feature_row[
                    "feature_date"
                ].iloc[0]
            ),

        "forecast_horizon":
            20,

        "direction": {
            "positive_probability":
                direction_probability,

            "predicted_class":
                direction_class,
        },

        "volatility": {
            "expected_annualized_volatility":
                expected_volatility,
        },

        "training_cutoff": (
            str(training_cutoff)
            if training_cutoff is not None
            else None
        ),
    }