import pandas as pd

from financial_ai.data.market_data import (
    get_market_data,
)
from financial_ai.data.temporal import (
    get_latest_10q,
    get_latest_8k,
)
from financial_ai.data.sec_data import (
    build_filing_metadata,
)

AS_OF_DATE = pd.Timestamp(
    "2026-07-15"
)


def test_market_data_has_no_future_rows():

    df = get_market_data(
        ticker="AAPL",
        as_of_date=str(
            AS_OF_DATE.date()
        ),
        lookback_years=2,
    )

    assert not df.empty

    assert (
        df["Date"].max()
        <= AS_OF_DATE
    )


def test_10q_has_no_future_filing():

    filing = get_latest_10q(
        ticker="AAPL",
        as_of_date=str(
            AS_OF_DATE.date()
        ),
    )

    assert (
        filing["filed_date"]
        <= AS_OF_DATE
    )


def test_8k_has_no_future_filing():

    filing = get_latest_8k(
        ticker="AAPL",
        as_of_date=str(
            AS_OF_DATE.date()
        ),
    )

    assert (
        filing["filed_date"]
        <= AS_OF_DATE
    )
    
def test_latest_10q_is_actually_latest_eligible():

    filing = get_latest_10q(
        ticker="AAPL",
        as_of_date="2026-07-15",
    )

    metadata = build_filing_metadata(
        ticker="AAPL"
    )

    eligible = metadata[
        (metadata["form"] == "10-Q")
        & (
            metadata["filed_date"]
            <= pd.Timestamp(
                "2026-07-15"
            )
        )
    ]

    expected = (
        eligible
        .sort_values(
            "filed_date",
            ascending=False,
        )
        .iloc[0]
    )

    assert (
        filing["accession_number"]
        == expected["accession_number"]
    )