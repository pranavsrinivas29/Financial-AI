from datetime import datetime

from fastapi import (
    APIRouter,
    HTTPException,
)

from financial_ai.agents.graph import (
    analyze_financial_query,
)

from financial_ai.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    AgentErrors,
)


router = APIRouter()


@router.get(
    "/health",
    tags=["Health"],
)
def health_check():

    return {
        "status": "ok",
        "service": "financial-ai",
    }


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    tags=["Analysis"],
)
def analyze(
    request: AnalyzeRequest,
):

    try:

        # Validate date format
        datetime.strptime(
            request.as_of_date,
            "%Y-%m-%d",
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail=(
                "as_of_date must use "
                "YYYY-MM-DD format"
            ),
        )

    try:

        result = analyze_financial_query(
            ticker=request.ticker,
            as_of_date=request.as_of_date,
            query=request.query,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Financial analysis failed: "
                f"{str(exc)}"
            ),
        )

    return AnalyzeResponse(
        ticker=request.ticker.upper(),
        as_of_date=request.as_of_date,
        query=request.query,

        ml_result=result.get(
            "ml_result"
        ),

        nlp_result=result.get(
            "nlp_result"
        ),

        rag_result=result.get(
            "rag_result"
        ),

        final_answer=result.get(
            "final_answer"
        ),

        errors=AgentErrors(
            ml=result.get(
                "ml_error"
            ),
            nlp=result.get(
                "nlp_error"
            ),
            rag=result.get(
                "rag_error"
            ),
        ),
    )