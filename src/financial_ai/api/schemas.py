from typing import Any

from pydantic import (
    BaseModel,
    Field,
)


class AnalyzeRequest(BaseModel):

    ticker: str = Field(
        ...,
        min_length=1,
        max_length=10,
        examples=["AAPL"],
    )

    as_of_date: str = Field(
        ...,
        examples=["2026-07-15"],
    )

    query: str = Field(
        ...,
        min_length=3,
        examples=[
            (
                "Analyze Apple using ML, "
                "news sentiment and SEC filings"
            )
        ],
    )


class AgentErrors(BaseModel):

    ml: str | None = None
    nlp: str | None = None
    rag: str | None = None


class AnalyzeResponse(BaseModel):

    ticker: str
    as_of_date: str
    query: str

    ml_result: dict[str, Any] | None = None
    nlp_result: dict[str, Any] | None = None
    rag_result: dict[str, Any] | None = None

    final_answer: str | None = None

    errors: AgentErrors