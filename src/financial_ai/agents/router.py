from financial_ai.agents.state import (
    FinancialAgentState,
)


def supervisor_node(
    state: FinancialAgentState,
) -> FinancialAgentState:

    query = state["query"].lower()

    use_ml = any(
        word in query
        for word in [
            "outlook",
            "prediction",
            "predict",
            "direction",
            "volatility",
            "forecast",
        ]
    )

    use_nlp = any(
        word in query
        for word in [
            "news",
            "sentiment",
            "headline",
            "market reaction",
        ]
    )

    use_rag = any(
        word in query
        for word in [
            "filing",
            "10-q",
            "8-k",
            "risk",
            "revenue",
            "sec",
            "company says",
            "reported",
        ]
    )

    # Broad financial-analysis requests
    # should use all available evidence.
    broad_queries = [
        "financial outlook",
        "overall outlook",
        "analyze",
        "analysis",
        "investment outlook",
        "company outlook",
    ]

    if any(
        phrase in query
        for phrase in broad_queries
    ):
        use_ml = True
        use_nlp = True
        use_rag = True

    # If nothing was explicitly detected,
    # let RAG answer factual financial questions.
    if not any(
        [
            use_ml,
            use_nlp,
            use_rag,
        ]
    ):
        use_rag = True

    return {
        **state,
        "use_ml": use_ml,
        "use_nlp": use_nlp,
        "use_rag": use_rag,
    }