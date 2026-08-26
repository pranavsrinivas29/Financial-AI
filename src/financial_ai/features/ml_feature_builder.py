import logging

import pandas as pd

from financial_ai.data.fundamentals import (
    get_fundamental_features,
)

from financial_ai.data.market_data import (
    get_market_data,
)

from financial_ai.features.market_features import (
    create_market_features,
)


logger = logging.getLogger(__name__)


MARKET_FEATURES = [
    "return_1d",
    "return_5d",
    "return_20d",
    "momentum_20d",
    "momentum_50d",
    "volatility_20d",
    "volatility_60d",
    "ma_20_50_ratio",
    "volume_change_5d",
    "volume_ratio_20d",
    "drawdown_1y",
]

def build_ml_feature_dataset(
    ticker: str,
    start_date: str,
    end_date: str,
    frequency: str = "W-FRI",
) -> pd.DataFrame:

    ticker = ticker.upper()

    end_date_ts = pd.Timestamp(
        end_date
    )

    # --------------------------------
    # Existing Phase 2 market service
    # --------------------------------

    market = get_market_data(
        ticker=ticker,
        as_of_date=end_date,
        lookback_years=12,
    )

    market = create_market_features(
        market
    )

    market = market[
        market["Date"]
        >= pd.Timestamp(start_date)
    ].copy()

    # --------------------------------
    # S&P 500 context
    # --------------------------------

    spy = get_market_data(
        ticker="SPY",
        as_of_date=end_date,
        lookback_years=12,
    )

    spy = create_market_features(
        spy
    )

    spy = spy.rename(
        columns={
            "return_20d":
                "spy_return_20d",

            "momentum_20d":
                "spy_momentum_20d",

            "volatility_20d":
                "spy_volatility_20d",
        }
    )

    spy_context = spy[
        [
            "Date",
            "spy_return_20d",
            "spy_momentum_20d",
            "spy_volatility_20d",
        ]
    ]

    # Merge market context by date.
    market = pd.merge(
        market,
        spy_context,
        on="Date",
        how="left",
    )

    # --------------------------------
    # Weekly feature snapshots
    # --------------------------------

    market = (
        market
        .set_index("Date")
        .resample(frequency)
        .last()
        .dropna(
            subset=["Close"]
        )
        .reset_index()
    )

    rows = []

    for _, row in market.iterrows():

        feature_date = row["Date"]

        fundamental = (
            get_fundamental_features(
                ticker=ticker,
                as_of_date=str(
                    feature_date.date()
                ),
            )
        )

        record = {

            "ticker":
                ticker,

            "feature_date":
                feature_date,
        }

        for feature in MARKET_FEATURES:

            record[feature] = (
                row.get(feature)
            )

        record[
            "spy_return_20d"
        ] = row.get(
            "spy_return_20d"
        )

        record[
            "spy_momentum_20d"
        ] = row.get(
            "spy_momentum_20d"
        )

        record[
            "spy_volatility_20d"
        ] = row.get(
            "spy_volatility_20d"
        )

        record.update(
            fundamental
        )

        rows.append(record)

    features = pd.DataFrame(
        rows
    )

    return features