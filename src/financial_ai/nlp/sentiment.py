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
    batch_size: int = 16,
) -> pd.DataFrame:

    if news_df.empty:
        return news_df.copy()

    df = news_df.copy()

    tokenizer, model = load_finbert()

    texts = (
        df["title"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    all_results = []

    for start in range(
        0,
        len(texts),
        batch_size,
    ):

        batch = texts[
            start:start + batch_size
        ]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
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
        )

        for probs in probabilities:

            result = {}

            for index, probability in enumerate(
                probs
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

            all_results.append(
                {
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
            )

    sentiment_df = pd.DataFrame(
        all_results,
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

def get_sentiment_analysis(
    ticker: str,
    as_of_date: str,
) -> dict:

    from financial_ai.data.temporal import (
        get_point_in_time_context,
    )

    from financial_ai.features.news_features import (
        create_news_features,
    )

    print(
        "NLP: getting point-in-time news",
        flush=True,
    )

    context = get_point_in_time_context(
        ticker=ticker,
        as_of_date=as_of_date,
        lookback_years=10,
        news_lookback_days=7,
    )

    news = context["news"]

    print(
        f"NLP: news received: {len(news)}",
        flush=True,
    )

    if news.empty:

        return {
            "ticker": ticker.upper(),
            "as_of_date": as_of_date,
            "article_count": 0,
            "features": create_news_features(
                news_df=pd.DataFrame(),
                as_of_date=as_of_date,
            ),
            "articles": [],
            "news_error": context.get(
                "news_error"
            ),
        }

    print(
        "NLP: running/loading FinBERT",
        flush=True,
    )

    scored = get_or_create_news_sentiment(
        news_df=news,
        ticker=ticker,
        as_of_date=as_of_date,
    )

    print(
        "NLP: FinBERT done",
        flush=True,
    )

    features = create_news_features(
        news_df=scored,
        as_of_date=as_of_date,
    )

    print(
        "NLP: features created",
        flush=True,
    )

    article_df = scored[
        [
            "published_at",
            "title",
            "sentiment",
            "sentiment_confidence",
            "positive_probability",
            "neutral_probability",
            "negative_probability",
        ]
    ].copy()

    article_df["published_at"] = (
        pd.to_datetime(
            article_df["published_at"],
            utc=True,
        )
        .astype(str)
    )

    articles = article_df.to_dict(
        orient="records"
    )

    print(
        "NLP: result prepared",
        flush=True,
    )

    return {
        "ticker": ticker.upper(),
        "as_of_date": as_of_date,
        "article_count": len(scored),
        "features": features,
        "articles": articles,
        "news_error": context.get(
            "news_error"
        ),
    }