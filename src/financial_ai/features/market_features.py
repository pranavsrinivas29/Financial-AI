import numpy as np
import pandas as pd


def create_market_features(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    df = df.sort_values("Date").reset_index(
        drop=True
    )

    # -----------------------------
    # Returns
    # -----------------------------

    df["return_1d"] = (
        df["Close"].pct_change()
    )

    df["return_5d"] = (
        df["Close"].pct_change(5)
    )

    df["return_20d"] = (
        df["Close"].pct_change(20)
    )

    # -----------------------------
    # Momentum
    # -----------------------------

    df["momentum_20d"] = (
        df["Close"]
        / df["Close"].shift(20)
        - 1
    )

    df["momentum_50d"] = (
        df["Close"]
        / df["Close"].shift(50)
        - 1
    )

    # -----------------------------
    # Moving averages
    # -----------------------------

    df["ma_20"] = (
        df["Close"]
        .rolling(20)
        .mean()
    )

    df["ma_50"] = (
        df["Close"]
        .rolling(50)
        .mean()
    )

    df["ma_200"] = (
        df["Close"]
        .rolling(200)
        .mean()
    )

    df["ma_20_50_ratio"] = (
        df["ma_20"]
        / df["ma_50"]
    )

    # -----------------------------
    # Volatility
    # -----------------------------

    df["volatility_20d"] = (
        df["return_1d"]
        .rolling(20)
        .std()
        * np.sqrt(252)
    )

    df["volatility_60d"] = (
        df["return_1d"]
        .rolling(60)
        .std()
        * np.sqrt(252)
    )

    # -----------------------------
    # Volume
    # -----------------------------

    df["volume_change_5d"] = (
        df["Volume"]
        .pct_change(5)
    )

    df["volume_ratio_20d"] = (
        df["Volume"]
        / df["Volume"]
        .rolling(20)
        .mean()
    )

    # -----------------------------
    # Drawdown
    # -----------------------------

    rolling_peak = (
        df["Close"]
        .rolling(252, min_periods=1)
        .max()
    )

    df["drawdown_1y"] = (
        df["Close"]
        / rolling_peak
        - 1
    )

    return df