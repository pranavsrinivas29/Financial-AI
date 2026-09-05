from financial_ai.rag.pipeline import (
    ask_financial_rag,
)


def main():

    ticker = "AAPL"

    as_of_date = "2026-07-15"

    question = (
        "What are the major risks "
        "identified by the company?"
    )

    result = ask_financial_rag(
        ticker=ticker,
        as_of_date=as_of_date,
        question=question,
    )

    print(
        "\n=================================="
    )

    print(
        "FINANCIAL RAG"
    )

    print(
        "=================================="
    )

    print(
        f"\nTicker: "
        f"{result['ticker']}"
    )

    print(
        f"As-of: "
        f"{result['as_of_date']}"
    )

    print(
        f"\nQuestion:\n"
        f"{result['question']}"
    )

    print(
        f"\nAnswer:\n"
        f"{result['answer']}"
    )

    print(
        "\nSources:"
    )

    for source in result[
        "sources"
    ]:

        print(
            f"\n"
            f"[Source "
            f"{source['source_id']}]"
        )

        print(
            f"Filing: "
            f"{source['filing_type']}"
        )

        print(
            f"Filed: "
            f"{source['filed_date']}"
        )

        print(
            f"Section: "
            f"{source['section']}"
        )

        print(
            f"URL: "
            f"{source['source_url']}"
        )


if __name__ == "__main__":
    main()