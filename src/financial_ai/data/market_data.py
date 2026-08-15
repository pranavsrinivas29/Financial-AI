from pathlib import Path

import pandas as pd
import yfinance as yf

from financial_ai.config.settings import settings


def download_market_data(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(
            f"No market data returned for ticker: {ticker}"
        )

    df = df.reset_index()

    # yfinance may return MultiIndex columns.
    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

    df["ticker"] = ticker.upper()

    return df


def save_raw_market_data(
    df: pd.DataFrame,
    ticker: str,
) -> Path:

    output_dir = settings.RAW_DATA_DIR / "market"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{ticker.upper()}.parquet"
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    return output_path