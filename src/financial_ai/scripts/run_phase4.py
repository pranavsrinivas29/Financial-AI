import pandas as pd

from financial_ai.config.settings import settings

from financial_ai.data.market_data import (
    get_market_data,
)

from financial_ai.features.ml_feature_builder import (
    build_ml_feature_dataset,
)

from financial_ai.ml.targets import (
    add_future_targets,
)

from financial_ai.ml.split import (
    final_holdout_split,
)

from financial_ai.ml.tune_direction import (
    tune_direction_model,
)

from financial_ai.ml.tune_volatility import (
    tune_volatility_model,
)

from financial_ai.ml.retrain import (
    train_final_direction_model,
    train_final_volatility_model,
)


def purge_development_overlap(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prevent target leakage across the
    development/holdout boundary.

    A development row is valid only if its complete
    future target was already known before the
    holdout period begins.
    """

    if development.empty or holdout.empty:
        return development

    holdout_start = (
        holdout["feature_date"].min()
    )

    development = development[
        development["target_end_date"]
        < holdout_start
    ].copy()

    if not development.empty:
        assert (
            development["target_end_date"].max()
            < holdout_start
        )

    return development


def main():

    ticker = "AAPL"

    start_date = "2018-01-01"
    end_date = "2025-12-31"

    print(
        "\nBuilding Phase 4 ML dataset..."
    )

    # ==================================================
    # 1. Build point-in-time ML features
    # ==================================================

    features = build_ml_feature_dataset(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
    )

    # ==================================================
    # 2. Daily market prices
    #
    # Used for LABEL generation only.
    # ==================================================

    daily_market = get_market_data(
        ticker=ticker,
        as_of_date=end_date,
        lookback_years=12,
    )

    # ==================================================
    # 3. Generate future targets
    # ==================================================

    dataset = add_future_targets(
        feature_df=features,
        daily_market_df=daily_market,
        horizon=20,
    )

    # ==================================================
    # 4. Remove samples without matured labels
    # ==================================================

    dataset = dataset.dropna(
        subset=[
            "target_end_date",
            "target_direction_20d",
            "target_volatility_20d",
        ]
    ).copy()

    dataset = (
        dataset
        .sort_values("feature_date")
        .reset_index(drop=True)
    )

    print(
        f"\nUsable labelled rows: "
        f"{len(dataset)}"
    )

    print(
        f"Feature start: "
        f"{dataset['feature_date'].min()}"
    )

    print(
        f"Feature end: "
        f"{dataset['feature_date'].max()}"
    )

    print(
        f"Latest target end: "
        f"{dataset['target_end_date'].max()}"
    )

    # ==================================================
    # 5. Save ML dataset
    # ==================================================

    output_dir = (
        settings.FEATURE_DATA_DIR
        / "training"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{ticker}_ml_features.parquet"
    )

    dataset.to_parquet(
        output_path,
        index=False,
    )

    print(
        f"\nDataset saved to:\n"
        f"{output_path}"
    )

    # ==================================================
    # 6. Development / untouched holdout
    # ==================================================

    development, holdout = (
        final_holdout_split(
            dataset,
            holdout_ratio=0.15,
        )
    )

    print(
        f"\nDevelopment rows "
        f"before purge: "
        f"{len(development)}"
    )

    print(
        f"Holdout rows: "
        f"{len(holdout)}"
    )

    # ==================================================
    # 7. PURGE overlapping labels
    # ==================================================

    development = (
        purge_development_overlap(
            development,
            holdout,
        )
    )

    print(
        f"Development rows "
        f"after purge: "
        f"{len(development)}"
    )

    print(
        f"Development end: "
        f"{development['feature_date'].max()}"
    )

    print(
        f"Development target end: "
        f"{development['target_end_date'].max()}"
    )

    print(
        f"Holdout start: "
        f"{holdout['feature_date'].min()}"
    )

    # ==================================================
    # 8. DIRECTION MODEL TUNING
    # ==================================================

    print(
        "\n======================================"
    )

    print(
        "TUNING DIRECTION MODEL"
    )

    print(
        "======================================"
    )

    direction_tuning = (
        tune_direction_model(
            development_df=development,
            max_trials=20,
        )
    )

    direction_best_params = (
        direction_tuning[
            "best_params"
        ]
    )

    print(
        "\nBest direction parameters:"
    )

    print(
        direction_best_params
    )

    print(
        "\nMean walk-forward ROC-AUC:"
    )

    print(
        direction_tuning[
            "best_mean_roc_auc"
        ]
    )

    # ==================================================
    # 9. FINAL DIRECTION HOLDOUT TEST
    # ==================================================

    print(
        "\nEvaluating direction model "
        "on untouched holdout..."
    )

    direction_final = (
        train_final_direction_model(
            development_df=development,
            holdout_df=holdout,
            best_params=(
                direction_best_params
            ),
        )
    )

    print(
        "\nFinal direction holdout metrics:"
    )

    for (
        metric,
        value,
    ) in direction_final[
        "metrics"
    ].items():

        print(
            f"{metric}: "
            f"{value:.4f}"
        )

    # ==================================================
    # 10. VOLATILITY MODEL TUNING
    # ==================================================

    print(
        "\n======================================"
    )

    print(
        "TUNING VOLATILITY MODEL"
    )

    print(
        "======================================"
    )

    volatility_tuning = (
        tune_volatility_model(
            development_df=development,
            max_trials=20,
        )
    )

    volatility_best_params = (
        volatility_tuning[
            "best_params"
        ]
    )

    print(
        "\nBest volatility parameters:"
    )

    print(
        volatility_best_params
    )

    print(
        "\nMean walk-forward RMSE:"
    )

    print(
        volatility_tuning[
            "best_mean_rmse"
        ]
    )

    # ==================================================
    # 11. FINAL VOLATILITY HOLDOUT TEST
    # ==================================================

    print(
        "\nEvaluating volatility model "
        "on untouched holdout..."
    )

    volatility_final = (
        train_final_volatility_model(
            development_df=development,
            holdout_df=holdout,
            best_params=(
                volatility_best_params
            ),
        )
    )

    print(
        "\nFinal volatility holdout metrics:"
    )

    for (
        metric,
        value,
    ) in volatility_final[
        "metrics"
    ].items():

        print(
            f"{metric}: "
            f"{value:.4f}"
        )

    # ==================================================
    # 12. SUMMARY
    # ==================================================

    print(
        "\n======================================"
    )

    print(
        "PHASE 4 MODEL DEVELOPMENT COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        "\nDirection best params:"
    )

    print(
        direction_best_params
    )

    print(
        "\nDirection holdout:"
    )

    print(
        direction_final[
            "metrics"
        ]
    )

    print(
        "\nVolatility best params:"
    )

    print(
        volatility_best_params
    )

    print(
        "\nVolatility holdout:"
    )

    print(
        volatility_final[
            "metrics"
        ]
    )


if __name__ == "__main__":
    main()