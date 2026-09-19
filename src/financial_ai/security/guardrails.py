import re


MAX_QUERY_LENGTH = 2000

TICKER_PATTERN = re.compile(
    r"^[A-Z]{1,5}$"
)


PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"reveal\s+(the\s+)?system\s+prompt",
    r"show\s+(the\s+)?system\s+prompt",
    r"print\s+(the\s+)?system\s+prompt",
    r"override\s+(the\s+)?system",
    r"developer\s+message",
]


class GuardrailViolation(
    ValueError
):
    pass


def validate_query(
    query: str,
) -> str:

    if not query:
        raise GuardrailViolation(
            "Query cannot be empty."
        )

    query = " ".join(
        query.split()
    )

    if len(query) > MAX_QUERY_LENGTH:
        raise GuardrailViolation(
            "Query exceeds allowed length."
        )

    lowercase_query = query.lower()

    for pattern in (
        PROMPT_INJECTION_PATTERNS
    ):
        if re.search(
            pattern,
            lowercase_query,
        ):
            raise GuardrailViolation(
                "Potential prompt injection detected."
            )

    return query


def validate_ticker(
    ticker: str,
) -> str:

    ticker = ticker.strip().upper()

    if not TICKER_PATTERN.fullmatch(
        ticker
    ):
        raise GuardrailViolation(
            "Invalid stock ticker."
        )

    return ticker


def validate_financial_request(
    query: str,
    ticker: str,
):
    return (
        validate_query(query),
        validate_ticker(ticker),
    )


def sanitize_output(
    text: str,
) -> str:

    if not text:
        return ""

    return str(text).strip()[:10000]