from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from financial_ai.agents.state import (
    FinancialAgentState,
)

from financial_ai.agents.router import (
    supervisor_node,
)

from financial_ai.agents.ml_agent import (
    ml_agent_node,
)

from financial_ai.agents.nlp_agent import (
    nlp_agent_node,
)

from financial_ai.agents.rag_agent import (
    rag_agent_node,
)

from financial_ai.agents.synthesis_agent import (
    synthesis_agent_node,
)


def build_financial_agent_graph():

    graph = StateGraph(
        FinancialAgentState
    )

    graph.add_node(
        "supervisor",
        supervisor_node,
    )

    graph.add_node(
        "ml_agent",
        ml_agent_node,
    )

    graph.add_node(
        "nlp_agent",
        nlp_agent_node,
    )

    graph.add_node(
        "rag_agent",
        rag_agent_node,
    )

    graph.add_node(
        "synthesis",
        synthesis_agent_node,
    )

    graph.add_edge(
        START,
        "supervisor",
    )

    graph.add_edge(
        "supervisor",
        "ml_agent",
    )

    graph.add_edge(
        "ml_agent",
        "nlp_agent",
    )

    graph.add_edge(
        "nlp_agent",
        "rag_agent",
    )

    graph.add_edge(
        "rag_agent",
        "synthesis",
    )

    graph.add_edge(
        "synthesis",
        END,
    )

    return graph.compile()

_FINANCIAL_GRAPH = None


def get_financial_graph():

    global _FINANCIAL_GRAPH

    if _FINANCIAL_GRAPH is None:

        _FINANCIAL_GRAPH = (
            build_financial_agent_graph()
        )

    return _FINANCIAL_GRAPH


def analyze_financial_query(
    ticker: str,
    as_of_date: str,
    query: str,
) -> dict:

    graph = get_financial_graph()

    result = graph.invoke(
        {
            "ticker":
                ticker.upper(),

            "as_of_date":
                as_of_date,

            "query":
                query,

            "ml_result":
                None,

            "nlp_result":
                None,

            "rag_result":
                None,
        }
    )

    return result