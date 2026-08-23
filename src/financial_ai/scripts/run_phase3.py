from financial_ai.data.temporal import (
    get_point_in_time_context,
)

from financial_ai.features.news_features import (
    create_news_features,
)

from financial_ai.nlp.sentiment import (
    get_or_create_news_sentiment,
)

from financial_ai.utils.logging_config import (
    configure_logging,
)


def main():

    configure_logging()

    ticker = "AAPL"
    as_of_date = "2026-08-20"

    context = (
        get_point_in_time_context(
            ticker=ticker,
            as_of_date=as_of_date,
            lookback_years=10,
            news_lookback_days=7,
        )
    )

    print(
        f"\nCompany: "
        f"{context['company_name']}"
    )

    print(
        f"As-of date: "
        f"{context['as_of_date']}"
    )

    if context["news_error"]:

        print(
            "\nNEWS WARNING:"
        )

        print(
            context["news_error"]
        )

    news = context["news"]

    print(
        f"\nNews articles: "
        f"{len(news)}"
    )

    if news.empty:

        print(
            "No usable news available."
        )

        return

    # FinBERT with cache
    news_with_sentiment = (
        get_or_create_news_sentiment(
            news_df=news,
            ticker=ticker,
            as_of_date=as_of_date,
        )
    )

    print(
        "\nNEWS SENTIMENT"
    )

    print(
        news_with_sentiment[
            [
                "published_at",
                "title",
                "sentiment",
                "sentiment_confidence",
            ]
        ].tail(10)
    )

    features = (
        create_news_features(
            news_df=
                news_with_sentiment,
            as_of_date=
                as_of_date,
        )
    )

    print(
        "\nNLP FEATURES"
    )

    print("-" * 50)

    for key, value in features.items():

        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()