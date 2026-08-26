import pandas as pd

from financial_ai.config.settings import (
    settings,
)

from financial_ai.ml.inference import (
    predict_financial_outlook,
)

from financial_ai.ml.model_config import (
    DIRECTION_APPROVED_PARAMS,
    VOLATILITY_APPROVED_PARAMS,
)


def main():

    ticker = "AAPL"

    inference_date = "2025-12-31"

    dataset_path = (
        settings.FEATURE_DATA_DIR
        / "training"
        / f"{ticker}_ml_features.parquet"
    )

    dataset = pd.read_parquet(
        dataset_path
    )

    result = predict_financial_outlook(
        dataset=dataset,
        inference_date=inference_date,
        direction_params=(
            DIRECTION_APPROVED_PARAMS
        ),
        volatility_params=(
            VOLATILITY_APPROVED_PARAMS
        ),
    )

    print(
        "\n======================================"
    )

    print(
        "FINANCIAL ML OUTLOOK"
    )

    print(
        "======================================"
    )

    print(
        f"\nTicker: "
        f"{result['ticker']}"
    )

    print(
        f"As-of date: "
        f"{result['as_of_date']}"
    )

    print(
        f"Feature date: "
        f"{result['feature_date']}"
    )

    print(
        "\nDirection:"
    )

    print(
        f"Positive probability: "
        f"{result['direction']['positive_probability']:.4f}"
    )

    print(
        f"Predicted class: "
        f"{result['direction']['predicted_class']}"
    )

    print(
        "\nVolatility:"
    )

    print(
        f"Expected annualized volatility: "
        f"{result['volatility']['expected_annualized_volatility']:.4f}"
    )

    print(
        f"\nLatest matured training target: "
        f"{result['training_cutoff']}"
    )


if __name__ == "__main__":
    main()