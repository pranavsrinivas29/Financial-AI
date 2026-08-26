import numpy as np
import pandas as pd


def add_future_targets(
    feature_df: pd.DataFrame,
    daily_market_df: pd.DataFrame,
    horizon: int = 20,
) -> pd.DataFrame:
    """
    Add future prediction targets.

    Future data is intentionally used here because
    these columns are LABELS, not input features.

    Also adds target_end_date, which represents the
    exact trading date on which the full target outcome
    becomes known.

    This allows safe retraining later using:

        target_end_date <= inference_date
    """

    features = feature_df.copy()

    prices = (
        daily_market_df
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    prices["Date"] = pd.to_datetime(
        prices["Date"]
    )

    close = (
        prices["Close"]
        .astype(float)
        .to_numpy()
    )

    daily_returns = (
        prices["Close"]
        .astype(float)
        .pct_change()
        .to_numpy()
    )

    n_rows = len(prices)

    # -----------------------------------
    # Empty target containers
    # -----------------------------------

    future_returns = np.full(
        n_rows,
        np.nan,
        dtype=float,
    )

    future_volatility = np.full(
        n_rows,
        np.nan,
        dtype=float,
    )

    target_end_dates = np.full(
        n_rows,
        np.datetime64("NaT"),
        dtype="datetime64[ns]",
    )

    # -----------------------------------
    # Build targets
    # -----------------------------------

    for i in range(n_rows):

        future_index = (
            i + horizon
        )

        if future_index >= n_rows:
            continue

        # --------------------------------
        # Exact date on which this target
        # becomes fully observable
        # --------------------------------

        target_end_dates[i] = (
            prices.iloc[
                future_index
            ]["Date"]
        )

        # --------------------------------
        # Future return
        # --------------------------------

        future_returns[i] = (
            close[future_index]
            / close[i]
            - 1
        )

        # --------------------------------
        # Future realized volatility
        #
        # Uses returns from:
        #
        # t+1 ... t+horizon
        # --------------------------------

        future_return_window = (
            daily_returns[
                i + 1:
                i + horizon + 1
            ]
        )

        if (
            len(future_return_window)
            == horizon
            and not np.isnan(
                future_return_window
            ).all()
        ):

            future_volatility[i] = (
                np.nanstd(
                    future_return_window,
                    ddof=1,
                )
                * np.sqrt(252)
            )

    # -----------------------------------
    # Assign targets
    # -----------------------------------

    prices[
        "target_end_date"
    ] = pd.to_datetime(
        target_end_dates
    )

    prices[
        "target_return_20d"
    ] = future_returns

    prices[
        "target_volatility_20d"
    ] = future_volatility

    # Keep direction missing whenever
    # future return itself is unavailable.

    prices[
        "target_direction_20d"
    ] = pd.Series(
        np.where(
            np.isnan(
                future_returns
            ),
            pd.NA,
            (
                future_returns > 0
            ).astype(int),
        ),
        dtype="Int64",
    )

    # -----------------------------------
    # Target table
    # -----------------------------------

    target_table = prices[
        [
            "Date",
            "target_end_date",
            "target_return_20d",
            "target_direction_20d",
            "target_volatility_20d",
        ]
    ].copy()

    # -----------------------------------
    # Feature dates may be weekly
    # resample dates such as Friday.
    #
    # If Friday was not a trading day,
    # associate it with the most recent
    # available trading date.
    # -----------------------------------

    features[
        "feature_date"
    ] = pd.to_datetime(
        features["feature_date"]
    ).astype("datetime64[ns]")

    target_table[
        "Date"
    ] = pd.to_datetime(
        target_table["Date"]
    ).astype("datetime64[ns]")

    features = (
        features
        .sort_values(
            "feature_date"
        )
        .reset_index(drop=True)
    )

    target_table = (
        target_table
        .sort_values(
            "Date"
        )
        .reset_index(drop=True)
    )

    result = pd.merge_asof(
        features,
        target_table,
        left_on="feature_date",
        right_on="Date",
        direction="backward",
    )

    # Keep the actual trading date used
    # temporarily if useful for debugging.
    result = result.rename(
        columns={
            "Date":
                "target_feature_market_date"
        }
    )

    return result