from financial_ai.config.settings import settings
from financial_ai.data.market_data import (
    download_market_data,
    save_raw_market_data,
)
from financial_ai.features.market_features import (
    create_market_features,
)


def main():
    ticker = settings.DEFAULT_TICKER

    df = download_market_data(
        ticker=ticker,
        start_date=settings.DEFAULT_START_DATE,
        end_date="2026-08-15",
    )

    raw_path = save_raw_market_data(
        df=df,
        ticker=ticker,
    )

    features = create_market_features(df)

    print(f"Ticker: {ticker}")
    print(f"Rows downloaded: {len(df)}")
    print(f"Raw data saved to: {raw_path}")

    print(
        features[
            [
                "Date",
                "Close",
                "return_1d",
                "return_20d",
                "volatility_20d",
            ]
        ].tail()
    )


if __name__ == "__main__":
    main()