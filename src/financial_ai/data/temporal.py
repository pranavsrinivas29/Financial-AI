import logging

import pandas as pd

from financial_ai.data.market_data import (
    get_market_data,
)
from financial_ai.data.news_data import (
    get_company_news,
)
from financial_ai.data.sec_data import (
    build_filing_metadata,
    get_company_name,
)


logger = logging.getLogger(__name__)

def get_latest_sec_filing(
    ticker: str,
    form: str,
    as_of_date: str,
) -> pd.Series:
    """
    Return the latest filing of the requested form
    that was publicly available on or before
    as_of_date.
    """

    as_of_date = pd.Timestamp(
        as_of_date
    ).normalize()

    filings = build_filing_metadata(
        ticker=ticker
    )

    eligible = filings[
        (filings["form"] == form)
        & (
            filings["available_at"]
            <= as_of_date
        )
    ].copy()

    if eligible.empty:
        raise ValueError(
            f"No {form} filing found for "
            f"{ticker} on or before "
            f"{as_of_date.date()}."
        )

    latest = (
        eligible
        .sort_values(
            "available_at",
            ascending=False,
        )
        .iloc[0]
    )

    assert (
        latest["available_at"]
        <= as_of_date
    )

    return latest


def get_latest_10q(
    ticker: str,
    as_of_date: str,
) -> pd.Series:

    return get_latest_sec_filing(
        ticker=ticker,
        form="10-Q",
        as_of_date=as_of_date,
    )


def get_latest_8k(
    ticker: str,
    as_of_date: str,
) -> pd.Series:

    return get_latest_sec_filing(
        ticker=ticker,
        form="8-K",
        as_of_date=as_of_date,
    )

def get_point_in_time_context(
    ticker: str,
    as_of_date: str,
    lookback_years: int = 10,
    news_lookback_days: int = 7,
    include_news: bool = True,
) -> dict:

    ticker = ticker.upper()

    company_name = get_company_name(
        ticker
    )

    market_data = get_market_data(
        ticker=ticker,
        as_of_date=as_of_date,
        lookback_years=lookback_years,
    )

    latest_10q = get_latest_10q(
        ticker=ticker,
        as_of_date=as_of_date,
    )

    latest_8k = get_latest_8k(
        ticker=ticker,
        as_of_date=as_of_date,
    )

    news_error = None

    if include_news:

        try:

            news = get_company_news(
                ticker=ticker,
                company_name=company_name,
                as_of_date=as_of_date,
                lookback_days=(
                    news_lookback_days
                ),
            )

        except Exception as exc:

            logger.exception(
                "News retrieval failed"
            )

            news = pd.DataFrame()

            news_error = str(exc)

    else:

        news = pd.DataFrame()

    return {
        "ticker":
            ticker,

        "company_name":
            company_name,

        "as_of_date":
            as_of_date,

        "market_data":
            market_data,

        "latest_10q":
            latest_10q,

        "latest_8k":
            latest_8k,

        "news":
            news,

        "news_error":
            news_error,
    }