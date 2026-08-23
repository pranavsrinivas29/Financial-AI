import logging
from functools import lru_cache

import pandas as pd
import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from financial_ai.config.settings import (
    settings,
)


logger = logging.getLogger(__name__)


FINBERT_MODEL = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def load_finbert():

    logger.info(
        "Loading FinBERT model"
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            FINBERT_MODEL
        )
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            FINBERT_MODEL
        )
    )

    model.eval()

    return tokenizer, model


def predict_sentiment(
    text: str,
) -> dict:

    if not text or not text.strip():

        return {
            "sentiment": "neutral",
            "sentiment_confidence": 0.0,
            "positive_probability": 0.0,
            "neutral_probability": 1.0,
            "negative_probability": 0.0,
        }

    tokenizer, model = load_finbert()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():

        outputs = model(
            **inputs
        )

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    result = {}

    for index, probability in enumerate(
        probabilities
    ):

        label = (
            model.config
            .id2label[index]
            .lower()
        )

        result[label] = float(
            probability
        )

    sentiment = max(
        result,
        key=result.get,
    )

    return {
        "sentiment":
            sentiment,

        "sentiment_confidence":
            result[sentiment],

        "positive_probability":
            result.get(
                "positive",
                0.0,
            ),

        "neutral_probability":
            result.get(
                "neutral",
                0.0,
            ),

        "negative_probability":
            result.get(
                "negative",
                0.0,
            ),
    }


def add_news_sentiment(
    news_df: pd.DataFrame,
) -> pd.DataFrame:

    if news_df.empty:
        return news_df.copy()

    df = news_df.copy()

    results = (
        df["title"]
        .fillna("")
        .apply(
            predict_sentiment
        )
    )

    sentiment_df = pd.DataFrame(
        results.tolist(),
        index=df.index,
    )

    return pd.concat(
        [
            df,
            sentiment_df,
        ],
        axis=1,
    )


def get_or_create_news_sentiment(
    news_df: pd.DataFrame,
    ticker: str,
    as_of_date: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Do not rerun FinBERT if processed
    sentiment already exists.
    """

    ticker = ticker.upper()

    output_dir = (
        settings.PROCESSED_DATA_DIR
        / "news"
        / ticker
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        output_dir
        / f"{as_of_date}.parquet"
    )

    if (
        use_cache
        and cache_path.exists()
    ):

        logger.info(
            "Using cached FinBERT results "
            "for %s",
            ticker,
        )

        return pd.read_parquet(
            cache_path
        )

    logger.info(
        "Running FinBERT for %s",
        ticker,
    )

    result = add_news_sentiment(
        news_df
    )

    result.to_parquet(
        cache_path,
        index=False,
    )

    logger.info(
        "Saved FinBERT results to %s",
        cache_path,
    )

    return result