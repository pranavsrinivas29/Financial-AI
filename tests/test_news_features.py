import pandas as pd

from financial_ai.features.news_features import (
    create_news_features,
)


def test_news_features():

    df = pd.DataFrame(
        {
            "published_at": [
                "2026-08-18T10:00:00Z",
                "2026-08-19T10:00:00Z",
                "2026-08-20T10:00:00Z",
            ],

            "sentiment": [
                "positive",
                "negative",
                "positive",
            ],

            "positive_probability": [
                0.80,
                0.10,
                0.70,
            ],

            "negative_probability": [
                0.05,
                0.80,
                0.10,
            ],

            "sentiment_confidence": [
                0.80,
                0.80,
                0.70,
            ],
        }
    )

    features = create_news_features(
        news_df=df,
        as_of_date="2026-08-20",
    )

    assert (
        features[
            "news_count_7d"
        ]
        == 3
    )

    assert (
        features[
            "positive_news_count_7d"
        ]
        == 2
    )

    assert (
        features[
            "negative_news_count_7d"
        ]
        == 1
    )