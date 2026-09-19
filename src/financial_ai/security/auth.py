import hmac
import os

from fastapi import (
    HTTPException,
    Security,
)
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)


def require_api_key(
    api_key: str | None = Security(
        api_key_header
    ),
):
    expected_key = os.getenv(
        "APP_API_KEY"
    )

    # For local development, API key
    # protection can remain disabled.
    if not expected_key:
        return True

    if (
        api_key is None
        or not hmac.compare_digest(
            api_key,
            expected_key,
        )
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
        )

    return True