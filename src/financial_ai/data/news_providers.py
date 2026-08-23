from abc import ABC, abstractmethod
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


GDELT_URL = (
    "https://api.gdeltproject.org/api/v2/doc/doc"
)


class NewsProvider(ABC):

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        company_name: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        max_records: int,
    ) -> list[dict]:
        pass


def _build_http_session() -> requests.Session:

    retry_strategy = Retry(
        total=4,
        backoff_factor=2,
        status_forcelist=[
            429,
            500,
            502,
            503,
            504,
        ],
        allowed_methods=["GET"],
        respect_retry_after_header=True,
    )

    adapter = HTTPAdapter(
        max_retries=retry_strategy
    )

    session = requests.Session()

    session.mount(
        "https://",
        adapter,
    )

    session.headers.update(
        {
            "User-Agent":
                "FinancialAIResearch/1.0"
        }
    )

    return session


class GDELTNewsProvider(NewsProvider):

    def __init__(self):
        self.session = _build_http_session()

    @staticmethod
    def _format_date(
        timestamp: pd.Timestamp,
    ) -> str:

        return timestamp.strftime(
            "%Y%m%d%H%M%S"
        )

    def fetch(
        self,
        ticker: str,
        company_name: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        max_records: int,
    ) -> list[dict]:

        params = {
            "query": f'"{company_name}"',
            "mode": "artlist",
            "format": "json",
            "maxrecords": min(
                max_records,
                250,
            ),
            "sort": "datedesc",
            "startdatetime":
                self._format_date(start_time),
            "enddatetime":
                self._format_date(end_time),
        }

        response = self.session.get(
            GDELT_URL,
            params=params,
            timeout=60,
        )

        response.raise_for_status()

        payload = response.json()

        rows = []

        for article in payload.get(
            "articles",
            [],
        ):

            rows.append(
                {
                    "ticker": ticker,
                    "company_name": company_name,
                    "title": article.get(
                        "title"
                    ),
                    "url": article.get(
                        "url"
                    ),
                    "domain": article.get(
                        "domain"
                    ),
                    "language": article.get(
                        "language"
                    ),
                    "source_country":
                        article.get(
                            "sourcecountry"
                        ),
                    "published_at":
                        article.get(
                            "seendate"
                        ),
                    "provider": "gdelt",
                }
            )

        return rows


class GoogleNewsRSSProvider(NewsProvider):

    def fetch(
        self,
        ticker: str,
        company_name: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        max_records: int,
    ) -> list[dict]:

        query = quote_plus(
            f'"{company_name}" stock'
        )

        url = (
            "https://news.google.com/rss/search"
            f"?q={query}"
            "&hl=en-US"
            "&gl=US"
            "&ceid=US:en"
        )

        feed = feedparser.parse(url)

        rows = []

        for entry in feed.entries[
            :max_records
        ]:

            rows.append(
                {
                    "ticker": ticker,
                    "company_name":
                        company_name,
                    "title":
                        entry.get(
                            "title"
                        ),
                    "url":
                        entry.get(
                            "link"
                        ),
                    "domain": None,
                    "language": "English",
                    "source_country": None,
                    "published_at":
                        entry.get(
                            "published"
                        ),
                    "provider":
                        "google_news_rss",
                }
            )

        return rows