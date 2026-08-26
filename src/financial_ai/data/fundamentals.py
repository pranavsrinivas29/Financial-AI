import json
import logging

import pandas as pd
import requests

from financial_ai.config.settings import settings
from financial_ai.data.sec_data import (
    get_cik,
)


logger = logging.getLogger(__name__)


SEC_COMPANY_FACTS_URL = (
    "https://data.sec.gov/api/xbrl/"
    "companyfacts/CIK{cik}.json"
)


def _headers() -> dict:

    return {
        "User-Agent":
            settings.SEC_USER_AGENT,

        "Accept-Encoding":
            "gzip, deflate",
    }


def download_company_facts(
    ticker: str,
    use_cache: bool = True,
) -> dict:

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
        output_dir
        / "companyfacts.json"
    )

    if (
        use_cache
        and cache_path.exists()
    ):

        with open(
            cache_path,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    cik_formatted = str(cik).zfill(10)

    url = SEC_COMPANY_FACTS_URL.format(
        cik=cik_formatted
    )

    logger.info(
        "Downloading SEC company facts "
        "for %s",
        ticker,
    )

    response = requests.get(
        url,
        headers=_headers(),
        timeout=60,
    )

    response.raise_for_status()

    payload = response.json()

    with open(
        cache_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            payload,
            f,
        )

    return payload

def _extract_concept(
    payload: dict,
    concepts: list[str],
    unit: str = "USD",
) -> pd.DataFrame:
    """
    Try multiple XBRL concept names and
    return the first one present.
    """

    us_gaap = (
        payload
        .get("facts", {})
        .get("us-gaap", {})
    )

    selected = None

    for concept in concepts:

        if concept in us_gaap:

            selected = concept
            break

    if selected is None:

        return pd.DataFrame()

    units = (
        us_gaap[selected]
        .get("units", {})
    )

    if unit not in units:

        return pd.DataFrame()

    df = pd.DataFrame(
        units[unit]
    )

    if df.empty:

        return df

    df["concept"] = selected

    required = [
        "end",
        "filed",
        "form",
        "val",
    ]

    for col in required:

        if col not in df.columns:

            return pd.DataFrame()

    df["period_end"] = pd.to_datetime(
        df["end"],
        errors="coerce",
    )

    df["available_at"] = pd.to_datetime(
        df["filed"],
        errors="coerce",
    )

    df["value"] = pd.to_numeric(
        df["val"],
        errors="coerce",
    )

    df = df[
        df["form"].isin(
            ["10-Q", "10-K"]
        )
    ].copy()

    return df[
        [
            "concept",
            "period_end",
            "available_at",
            "form",
            "value",
        ]
    ]
    
FUNDAMENTAL_CONCEPTS = {

    "revenue": [
        (
            "RevenueFromContractWith"
            "CustomerExcludingAssessedTax"
        ),
        "Revenues",
        "SalesRevenueNet",
    ],

    "operating_income": [
        "OperatingIncomeLoss",
    ],

    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],

    "assets_current": [
        "AssetsCurrent",
    ],

    "liabilities_current": [
        "LiabilitiesCurrent",
    ],

    "liabilities": [
        "Liabilities",
    ],

    "equity": [
        "StockholdersEquity",
        (
            "StockholdersEquityIncluding"
            "PortionAttributableTo"
            "NoncontrollingInterest"
        ),
    ],

    "cash": [
        (
            "CashAndCashEquivalents"
            "AtCarryingValue"
        ),
        (
            "CashCashEquivalents"
            "RestrictedCashAnd"
            "RestrictedCashEquivalents"
        ),
    ],
}


def _latest_point_in_time_value(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> tuple[float, pd.Timestamp | None]:

    if df.empty:

        return float("nan"), None

    eligible = df[
        df["available_at"]
        <= as_of_date
    ].copy()

    if eligible.empty:

        return float("nan"), None

    # If duplicate facts exist for the same
    # reporting period, use the most recently
    # filed version available at that time.

    eligible = (
        eligible
        .sort_values(
            [
                "period_end",
                "available_at",
            ]
        )
        .drop_duplicates(
            subset=["period_end"],
            keep="last",
        )
    )

    latest = (
        eligible
        .sort_values(
            "period_end"
        )
        .iloc[-1]
    )

    return (
        float(latest["value"]),
        latest["available_at"],
    )
    
def get_fundamental_features(
    ticker: str,
    as_of_date: str,
) -> dict:

    ticker = ticker.upper()

    cutoff = pd.Timestamp(
        as_of_date
    ).normalize()

    payload = download_company_facts(
        ticker
    )

    extracted = {}

    availability_dates = []

    for (
        feature_name,
        concepts,
    ) in FUNDAMENTAL_CONCEPTS.items():

        df = _extract_concept(
            payload=payload,
            concepts=concepts,
        )

        value, available_at = (
            _latest_point_in_time_value(
                df=df,
                as_of_date=cutoff,
            )
        )

        extracted[feature_name] = value

        if available_at is not None:

            availability_dates.append(
                available_at
            )

    revenue = extracted.get(
        "revenue"
    )

    operating_income = extracted.get(
        "operating_income"
    )

    net_income = extracted.get(
        "net_income"
    )

    current_assets = extracted.get(
        "assets_current"
    )

    current_liabilities = extracted.get(
        "liabilities_current"
    )

    liabilities = extracted.get(
        "liabilities"
    )

    equity = extracted.get(
        "equity"
    )

    cash = extracted.get(
        "cash"
    )

    # -----------------------------
    # Derived ratios
    # -----------------------------

    operating_margin = (
        operating_income / revenue
        if (
            revenue
            and pd.notna(revenue)
            and revenue != 0
            and pd.notna(
                operating_income
            )
        )
        else float("nan")
    )

    net_margin = (
        net_income / revenue
        if (
            revenue
            and pd.notna(revenue)
            and revenue != 0
            and pd.notna(net_income)
        )
        else float("nan")
    )

    current_ratio = (
        current_assets
        / current_liabilities
        if (
            pd.notna(current_assets)
            and pd.notna(
                current_liabilities
            )
            and current_liabilities != 0
        )
        else float("nan")
    )

    liabilities_to_equity = (
        liabilities / equity
        if (
            pd.notna(liabilities)
            and pd.notna(equity)
            and equity != 0
        )
        else float("nan")
    )

    cash_to_current_liabilities = (
        cash / current_liabilities
        if (
            pd.notna(cash)
            and pd.notna(
                current_liabilities
            )
            and current_liabilities != 0
        )
        else float("nan")
    )

    latest_available_at = (
        max(availability_dates)
        if availability_dates
        else None
    )

    # Hard temporal guarantee.
    if latest_available_at is not None:

        assert (
            latest_available_at
            <= cutoff
        )

    revenue_df = _extract_concept(
        payload,
        FUNDAMENTAL_CONCEPTS[
            "revenue"
        ],
    )

    revenue_growth = _latest_growth(
        revenue_df,
        cutoff,
    )

    return {

        "revenue":
            revenue,

        "operating_income":
            operating_income,

        "net_income":
            net_income,

        "operating_margin":
            operating_margin,

        "net_margin":
            net_margin,

        "current_ratio":
            current_ratio,

        "liabilities_to_equity":
            liabilities_to_equity,

        "cash_to_current_liabilities":
            cash_to_current_liabilities,

        "fundamental_available_at":
            latest_available_at,
            
        "revenue_growth":
            revenue_growth,
    }
    
def _latest_growth(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp,
) -> float:

    if df.empty:

        return float("nan")

    eligible = df[
        df["available_at"]
        <= as_of_date
    ].copy()

    if eligible.empty:

        return float("nan")

    eligible = (
        eligible
        .sort_values(
            [
                "period_end",
                "available_at",
            ]
        )
        .drop_duplicates(
            subset=["period_end"],
            keep="last",
        )
        .sort_values(
            "period_end"
        )
    )

    if len(eligible) < 2:

        return float("nan")

    current = float(
        eligible.iloc[-1][
            "value"
        ]
    )

    previous = float(
        eligible.iloc[-2][
            "value"
        ]
    )

    if previous == 0:

        return float("nan")

    return (
        current / previous - 1
    )