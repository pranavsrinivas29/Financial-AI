from typing import TypedDict, Optional, Any


class FinancialAgentState(TypedDict, total=False):

    # Request
    query: str
    ticker: str
    as_of_date: str

    # Routing decisions
    use_ml: bool
    use_nlp: bool
    use_rag: bool

    # Outputs from independent agents
    ml_result: Optional[dict[str, Any]]
    nlp_result: Optional[dict[str, Any]]
    rag_result: Optional[dict[str, Any]]

    # Errors
    ml_error: Optional[str]
    nlp_error: Optional[str]
    rag_error: Optional[str]

    # Final answer
    final_answer: Optional[str]