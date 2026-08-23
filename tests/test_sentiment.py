from financial_ai.nlp.sentiment import (
    predict_sentiment,
)


def test_finbert_output():

    result = predict_sentiment(
        "The company reported "
        "strong revenue growth "
        "and higher profits."
    )

    assert (
        result["sentiment"]
        in {
            "positive",
            "neutral",
            "negative",
        }
    )

    probability_sum = (
        result[
            "positive_probability"
        ]
        +
        result[
            "neutral_probability"
        ]
        +
        result[
            "negative_probability"
        ]
    )

    assert abs(
        probability_sum - 1
    ) < 0.01