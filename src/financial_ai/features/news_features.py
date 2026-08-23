import pandas as pd


def create_news_features(
    news_df: pd.DataFrame,
    as_of_date: str,
) -> dict:
    """
    Create aggregated NLP features suitable
    for later ML models.
    """

    if news_df.empty:

        return {
            "news_count_3d": 0,
            "news_count_7d": 0,

            "positive_news_count_7d": 0,
            "neutral_news_count_7d": 0,
            "negative_news_count_7d": 0,

            "news_sentiment_score_3d": 0.0,
            "news_sentiment_score_7d": 0.0,

            "mean_sentiment_confidence_7d": 0.0,
        }

    df = news_df.copy()

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        utc=True,
    )

    cutoff = pd.Timestamp(
        as_of_date
    )

    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(
            "UTC"
        )

    if (
        cutoff.hour == 0
        and cutoff.minute == 0
        and cutoff.second == 0
    ):
        cutoff = (
            cutoff
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )

    df = df[
        df["published_at"] <= cutoff
    ].copy()

    last_3d = df[
        df["published_at"]
        >= cutoff - pd.Timedelta(days=3)
    ]

    last_7d = df[
        df["published_at"]
        >= cutoff - pd.Timedelta(days=7)
    ]

    def sentiment_score(
        temp: pd.DataFrame,
    ) -> float:

        if temp.empty:
            return 0.0

        # continuous score:
        # positive probability - negative probability
        return float(
            (
                temp["positive_probability"]
                - temp["negative_probability"]
            ).mean()
        )

    return {
        "news_count_3d":
            int(len(last_3d)),

        "news_count_7d":
            int(len(last_7d)),

        "positive_news_count_7d":
            int(
                (
                    last_7d["sentiment"]
                    == "positive"
                ).sum()
            ),

        "neutral_news_count_7d":
            int(
                (
                    last_7d["sentiment"]
                    == "neutral"
                ).sum()
            ),

        "negative_news_count_7d":
            int(
                (
                    last_7d["sentiment"]
                    == "negative"
                ).sum()
            ),

        "news_sentiment_score_3d":
            sentiment_score(
                last_3d
            ),

        "news_sentiment_score_7d":
            sentiment_score(
                last_7d
            ),

        "mean_sentiment_confidence_7d":
            (
                float(
                    last_7d[
                        "sentiment_confidence"
                    ].mean()
                )
                if not last_7d.empty
                else 0.0
            ),
    }