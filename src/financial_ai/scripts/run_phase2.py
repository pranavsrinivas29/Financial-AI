from financial_ai.data.sec_data import (
    download_filing_document,
)
from financial_ai.data.temporal import (
    get_point_in_time_context,
)


def main():

    ticker = "AAPL"
    as_of_date = "2026-08-23"

    context = get_point_in_time_context(
        ticker=ticker,
        as_of_date=as_of_date,
        lookback_years=10,
    )

    market = context["market_data"]

    latest_10q = context["latest_10q"]
    latest_8k = context["latest_8k"]

    print("\nMARKET DATA")
    print("-" * 50)

    print(
        f"Rows: {len(market)}"
    )

    print(
        f"Latest market date: "
        f"{market['Date'].max()}"
    )

    print("\nLATEST 10-Q")
    print("-" * 50)

    print(
        latest_10q[
            [
                "form",
                "filed_date",
                "reporting_period",
                "document_url",
            ]
        ]
    )

    print("\nLATEST 8-K")
    print("-" * 50)

    print(
        latest_8k[
            [
                "form",
                "filed_date",
                "reporting_period",
                "document_url",
            ]
        ]
    )

    path_10q = download_filing_document(
        latest_10q
    )

    path_8k = download_filing_document(
        latest_8k
    )

    print("\nDOWNLOADED")
    print("-" * 50)

    print(
        f"10-Q: {path_10q}"
    )

    print(
        f"8-K : {path_8k}"
    )


if __name__ == "__main__":
    main()