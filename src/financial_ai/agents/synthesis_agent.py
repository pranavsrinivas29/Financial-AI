import json
import requests

from financial_ai.agents.state import (
    FinancialAgentState,
)


OLLAMA_URL = (
    "http://127.0.0.1:11434/api/chat"
)

MODEL_NAME = "qwen2.5:7b"


SYSTEM_PROMPT = """
You are a financial analysis orchestration agent.

You receive outputs from independent systems:

1. Predictive ML
2. Financial news sentiment
3. SEC filing RAG

Rules:

- Never invent missing values.
- Clearly distinguish predictions from factual disclosures.
- ML predictions are estimates, not facts.
- Sentiment describes news tone, not future returns.
- SEC filing evidence represents company disclosures.
- Mention uncertainty or disagreement between sources.
- Do not provide personalized investment advice.
- Preserve SEC source references when supplied.
"""


def synthesis_agent_node(
    state: FinancialAgentState,
) -> FinancialAgentState:

    evidence = {
        "ticker":
            state["ticker"],

        "as_of_date":
            state["as_of_date"],

        "question":
            state["query"],

        "ml":
            state.get("ml_result"),

        "nlp":
            state.get("nlp_result"),

        "rag":
            state.get("rag_result"),

        "errors": {
            "ml":
                state.get("ml_error"),

            "nlp":
                state.get("nlp_error"),

            "rag":
                state.get("rag_error"),
        },
    }

    user_prompt = f"""
Analyze the following evidence and answer the
user's question.

Evidence:

{json.dumps(
    evidence,
    indent=2,
    default=str,
)}

Return a concise financial analysis with:

1. Overall assessment
2. ML outlook, if available
3. News sentiment, if available
4. SEC filing evidence, if available
5. Risks / contradictions
6. Sources, where available
"""

    payload = {
        "model": MODEL_NAME,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    }

    print(
    "SYNTHESIS: preparing evidence",
    flush=True,
)
    print(
    "SYNTHESIS: calling Ollama",
    flush=True,
)
    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    answer = (
        response.json()
        ["message"]
        ["content"]
    )

    return {
        **state,
        "final_answer": answer,
    }