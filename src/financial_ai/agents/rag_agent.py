from financial_ai.agents.state import (
    FinancialAgentState,
)


def rag_agent_node(
    state: FinancialAgentState,
) -> FinancialAgentState:

    if not state.get("use_rag", False):
        return state

    try:
        from financial_ai.rag.pipeline import (
            ask_financial_rag,
        )

        result = ask_financial_rag(
            ticker=state["ticker"],
            as_of_date=state["as_of_date"],
            question=state["query"],
        )

        return {
            **state,
            "rag_result": result,
            "rag_error": None,
        }

    except Exception as exc:

        return {
            **state,
            "rag_result": None,
            "rag_error": str(exc),
        }