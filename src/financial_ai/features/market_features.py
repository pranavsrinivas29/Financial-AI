import numpy as np
import pandas as pd


def create_market_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df = df.sort_values("Date")

    df["return_1d"] = (
        df["Close"]
        .pct_change()
    )

    df["return_5d"] = (
        df["Close"]
        .pct_change(5)
    )

    df["return_20d"] = (
        df["Close"]
        .pct_change(20)
    )

    df["momentum_20d"] = (
        df["Close"]
        / df["Close"].shift(20)
        - 1
    )

    df["ma_20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    df["ma_50"] = (
        df["Close"]
        .rolling(window=50)
        .mean()
    )

    df["volatility_20d"] = (
        df["return_1d"]
        .rolling(window=20)
        .std()
        * np.sqrt(252)
    )

    return df