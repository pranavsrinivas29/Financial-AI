from pathlib import Path

import pandas as pd
import yfinance as yf

from financial_ai.config.settings import settings


def get_market_data(
    ticker: str,
    as_of_date: str,
    lookback_years: int = 10,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Return market data available on or before as_of_date.

    Guarantees:
        max(Date) <= as_of_date
    """

    ticker = ticker.upper()
    as_of_date = pd.Timestamp(as_of_date).normalize()

    start_date = as_of_date - pd.DateOffset(years=lookback_years)

    output_dir = settings.RAW_DATA_DIR / "market"
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_path = output_dir / f"{ticker}.parquet"

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)

        df["Date"] = pd.to_datetime(df["Date"])

        cached_max_date = df["Date"].max()

        # Reuse cache only if it reaches our requested inference date.
        if cached_max_date >= as_of_date:
            return _apply_temporal_filter(
                df=df,
                as_of_date=as_of_date,
            )

    # yfinance end is effectively exclusive,
    # therefore request one extra calendar day.
    download_end = as_of_date + pd.Timedelta(days=1)

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=download_end.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"No market data returned for ticker {ticker}"
        )

    df = df.reset_index()

    # yfinance may return MultiIndex columns.
    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

    df["Date"] = pd.to_datetime(df["Date"])
    df["ticker"] = ticker

    df = _apply_temporal_filter(
        df=df,
        as_of_date=as_of_date,
    )

    df.to_parquet(
        cache_path,
        index=False,
    )

    return df


def _apply_temporal_filter(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:

    filtered = (
        df.loc[df["Date"] <= as_of_date]
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if not filtered.empty:
        assert filtered["Date"].max() <= as_of_date

    return filtered