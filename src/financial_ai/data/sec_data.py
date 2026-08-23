import json
from pathlib import Path

import pandas as pd
import requests

from financial_ai.config.settings import settings


SEC_TICKER_URL = (
    "https://www.sec.gov/files/company_tickers.json"
)

SEC_SUBMISSIONS_URL = (
    "https://data.sec.gov/submissions/"
    "CIK{cik}.json"
)

SEC_ARCHIVES_BASE = (
    "https://www.sec.gov/Archives/edgar/data"
)


def _get_headers() -> dict:
    return {
        "User-Agent": settings.SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }


def get_company_ticker_map(
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download SEC ticker -> CIK mapping.
    """

    output_dir = settings.DATA_DIR / "metadata"
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        output_dir / "sec_company_tickers.parquet"
    )

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    response = requests.get(
        SEC_TICKER_URL,
        headers=_get_headers(),
        timeout=30,
    )

    response.raise_for_status()

    raw = response.json()

    rows = []

    for value in raw.values():
        rows.append(
            {
                "ticker": value["ticker"].upper(),
                "company_name": value["title"],
                "cik": int(value["cik_str"]),
            }
        )

    df = pd.DataFrame(rows)

    df.to_parquet(
        cache_path,
        index=False,
    )

    return df


def get_cik(
    ticker: str,
) -> int:
    """
    Resolve stock ticker to SEC CIK.
    """

    ticker = ticker.upper()

    ticker_map = get_company_ticker_map()

    match = ticker_map[
        ticker_map["ticker"].eq(ticker)
    ]

    if match.empty:
        raise ValueError(
            f"Ticker {ticker} not found "
            f"in SEC company ticker mapping."
        )

    return int(match.iloc[0]["cik"])


def _format_cik(cik: int) -> str:
    """
    SEC submissions API expects a 10 digit CIK.
    """

    return str(cik).zfill(10)


def download_company_submissions(
    ticker: str,
    use_cache: bool = True,
) -> dict:
    """
    Download the company's SEC submissions JSON.
    """

    ticker = ticker.upper()
    cik = get_cik(ticker)

    output_dir = (
        settings.RAW_DATA_DIR
        / "sec"
        / ticker
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        output_dir / "submissions.json"
    )

    if use_cache and cache_path.exists():
        with open(
            cache_path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    url = SEC_SUBMISSIONS_URL.format(
        cik=_format_cik(cik)
    )

    response = requests.get(
        url,
        headers=_get_headers(),
        timeout=30,
    )

    response.raise_for_status()

    submissions = response.json()

    with open(
        cache_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            submissions,
            f,
            indent=2,
        )

    return submissions

def build_filing_metadata(
    ticker: str,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build filing metadata from SEC submissions.

    Important:
    available_at = filed_date
    """

    ticker = ticker.upper()

    output_dir = (
        settings.RAW_DATA_DIR
        / "sec"
        / ticker
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        output_dir
        / "filings_metadata.parquet"
    )

    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)

        df["filed_date"] = pd.to_datetime(
            df["filed_date"]
        )

        df["reporting_period"] = pd.to_datetime(
            df["reporting_period"],
            errors="coerce",
        )

        df["available_at"] = pd.to_datetime(
            df["available_at"]
        )

        return df

    submissions = download_company_submissions(
        ticker=ticker,
        use_cache=use_cache,
    )

    recent = submissions["filings"]["recent"]

    df = pd.DataFrame(recent)

    required_cols = [
        "accessionNumber",
        "filingDate",
        "reportDate",
        "form",
        "primaryDocument",
    ]

    missing_cols = [
        col
        for col in required_cols
        if col not in df.columns
    ]

    if missing_cols:
        raise ValueError(
            f"Missing SEC fields: {missing_cols}"
        )

    cik = get_cik(ticker)

    df = df[
        required_cols
    ].copy()

    df = df.rename(
        columns={
            "accessionNumber":
                "accession_number",
            "filingDate":
                "filed_date",
            "reportDate":
                "reporting_period",
            "primaryDocument":
                "primary_document",
        }
    )

    df["ticker"] = ticker
    df["cik"] = cik

    df["filed_date"] = pd.to_datetime(
        df["filed_date"]
    )

    df["reporting_period"] = pd.to_datetime(
        df["reporting_period"],
        errors="coerce",
    )

    # This is the timestamp that matters
    # for point-in-time correctness.
    df["available_at"] = df["filed_date"]

    df["document_url"] = df.apply(
        lambda row: build_filing_url(
            cik=cik,
            accession_number=(
                row["accession_number"]
            ),
            primary_document=(
                row["primary_document"]
            ),
        ),
        axis=1,
    )

    df = df[
        [
            "ticker",
            "cik",
            "form",
            "filed_date",
            "reporting_period",
            "available_at",
            "accession_number",
            "primary_document",
            "document_url",
        ]
    ]

    df = df.sort_values(
        "filed_date",
        ascending=False,
    ).reset_index(drop=True)

    df.to_parquet(
        cache_path,
        index=False,
    )

    return df


def build_filing_url(
    cik: int,
    accession_number: str,
    primary_document: str,
) -> str:

    accession_clean = (
        accession_number.replace("-", "")
    )

    return (
        f"{SEC_ARCHIVES_BASE}/"
        f"{cik}/"
        f"{accession_clean}/"
        f"{primary_document}"
    )
    
def download_filing_document(
    filing: pd.Series,
    overwrite: bool = False,
) -> Path:
    """
    Download the selected SEC filing document.
    """

    ticker = filing["ticker"]
    form = filing["form"]

    filed_date = pd.Timestamp(
        filing["filed_date"]
    ).strftime("%Y-%m-%d")

    accession = filing[
        "accession_number"
    ].replace("-", "")

    output_dir = (
        settings.RAW_DATA_DIR
        / "sec"
        / ticker
        / form.lower().replace("-", "")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        f"{filed_date}_"
        f"{accession}.html"
    )

    output_path = (
        output_dir / filename
    )

    if (
        output_path.exists()
        and not overwrite
    ):
        return output_path

    response = requests.get(
        filing["document_url"],
        headers=_get_headers(),
        timeout=30,
    )

    response.raise_for_status()

    output_path.write_bytes(
        response.content
    )

    return output_path
    
    