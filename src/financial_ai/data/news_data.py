import logging
from difflib import SequenceMatcher

import pandas as pd

from financial_ai.config.settings import (
    settings,
)
from financial_ai.data.news_providers import (
    GDELTNewsProvider,
    GoogleNewsRSSProvider,
)


logger = logging.getLogger(__name__)


def _get_news_cutoff(
    as_of_date: str,
) -> pd.Timestamp:

    cutoff = pd.Timestamp(
        as_of_date
    )

    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize(
            "UTC"
        )
    else:
        cutoff = cutoff.tz_convert(
            "UTC"
        )

    # Date-only input means end of day.
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

    return cutoff


def _apply_temporal_filter(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.DataFrame:

    df = df.copy()

    df["published_at"] = pd.to_datetime(
        df["published_at"],
        utc=True,
        errors="coerce",
    )

    df = df.dropna(
        subset=["published_at"]
    )

    df = df[
        (
            df["published_at"]
            >= start_time
        )
        &
        (
            df["published_at"]
            <= cutoff
        )
    ].copy()

    if not df.empty:

        assert (
            df["published_at"].max()
            <= cutoff
        )

    return df


def _headline_similarity(
    first: str,
    second: str,
) -> float:

    return SequenceMatcher(
        None,
        first.lower(),
        second.lower(),
    ).ratio()


def _deduplicate_news(
    df: pd.DataFrame,
    similarity_threshold: float = 0.90,
) -> pd.DataFrame:

    if df.empty:
        return df

    df = (
        df
        .drop_duplicates(
            subset=["url"]
        )
        .reset_index(drop=True)
    )

    keep_indexes = []

    accepted_titles = []

    for idx, row in df.iterrows():

        title = str(
            row.get("title", "")
        ).strip()

        if not title:
            continue

        is_duplicate = False

        for previous_title in accepted_titles:

            similarity = (
                _headline_similarity(
                    title,
                    previous_title,
                )
            )

            if (
                similarity
                >= similarity_threshold
            ):
                is_duplicate = True
                break

        if not is_duplicate:

            keep_indexes.append(idx)
            accepted_titles.append(title)

    return (
        df.loc[keep_indexes]
        .sort_values(
            "published_at"
        )
        .reset_index(drop=True)
    )


def get_company_news(
    ticker: str,
    company_name: str,
    as_of_date: str,
    lookback_days: int = 7,
    max_records: int = 50,
    use_cache: bool = True,
) -> pd.DataFrame:

    ticker = ticker.upper()

    cutoff = _get_news_cutoff(
        as_of_date
    )

    start_time = (
        cutoff.normalize()
        - pd.Timedelta(
            days=lookback_days - 1
        )
    )

    output_dir = (
        settings.RAW_DATA_DIR
        / "news"
        / ticker
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        output_dir
        / (
            f"{cutoff.strftime('%Y-%m-%d')}"
            f"_{lookback_days}d.parquet"
        )
    )

    # ----------------------------------
    # 1. CACHE FIRST
    # ----------------------------------

    if (
        use_cache
        and cache_path.exists()
    ):

        logger.info(
            "Using cached news for %s",
            ticker,
        )

        df = pd.read_parquet(
            cache_path
        )

        return _apply_temporal_filter(
            df=df,
            start_time=start_time,
            cutoff=cutoff,
        )

    providers = [
        GDELTNewsProvider(),
        GoogleNewsRSSProvider(),
    ]

    provider_errors = []

    rows = []

    # ----------------------------------
    # 2. TRY PROVIDERS IN ORDER
    # ----------------------------------

    for provider in providers:

        try:

            logger.info(
                "Fetching %s news using %s",
                ticker,
                provider.__class__.__name__,
            )

            rows = provider.fetch(
                ticker=ticker,
                company_name=company_name,
                start_time=start_time,
                end_time=cutoff,
                max_records=max_records,
            )

            if rows:
                break

        except Exception as exc:

            logger.warning(
                "News provider %s failed: %s",
                provider.__class__.__name__,
                exc,
            )

            provider_errors.append(
                str(exc)
            )

    if not rows:

        logger.warning(
            "All news providers failed "
            "or returned no results."
        )

        return pd.DataFrame(
            columns=[
                "ticker",
                "company_name",
                "title",
                "url",
                "domain",
                "language",
                "source_country",
                "published_at",
                "provider",
            ]
        )

    # ----------------------------------
    # 3. TEMPORAL FILTER
    # ----------------------------------

    df = pd.DataFrame(rows)

    df = _apply_temporal_filter(
        df=df,
        start_time=start_time,
        cutoff=cutoff,
    )

    # ----------------------------------
    # 4. DEDUPLICATE
    # ----------------------------------

    df = _deduplicate_news(
        df
    )

    # ----------------------------------
    # 5. CACHE RAW DATA
    # ----------------------------------

    df.to_parquet(
        cache_path,
        index=False,
    )

    logger.info(
        "Saved %s news articles to %s",
        len(df),
        cache_path,
    )

    return df