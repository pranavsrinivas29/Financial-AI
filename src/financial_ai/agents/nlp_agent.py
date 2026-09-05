from financial_ai.agents.state import (
    FinancialAgentState,
)

def nlp_agent_node(state):

    if not state.get("use_nlp", False):
        return state

    try:
        print("NLP: starting")

        from financial_ai.nlp.sentiment import (
            get_sentiment_analysis,
        )

        print("NLP: imported sentiment")

        result = get_sentiment_analysis(
            ticker=state["ticker"],
            as_of_date=state["as_of_date"],
        )

        print(
    "NLP: sentiment complete",
    flush=True,
)

        return {
            **state,
            "nlp_result": result,
            "nlp_error": None,
        }

    except Exception as exc:

        print(
            "NLP ERROR:",
            repr(exc),
        )

        return {
            **state,
            "nlp_result": None,
            "nlp_error": str(exc),
        }