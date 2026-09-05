import requests

from financial_ai.rag.prompts import (
    FINANCIAL_RAG_SYSTEM_PROMPT,
)


OLLAMA_URL = (
    "http://localhost:11434/api/chat"
)


def generate_answer(
    question: str,
    context: str,
    model: str = "qwen2.5:7b",
) -> str:

    user_prompt = f"""
Question:

{question}

SEC filing context:

{context}

Answer the question based only on the context.
"""

    payload = {
        "model": model,

        "stream": False,

        "messages": [
            {
                "role": "system",
                "content":
                    FINANCIAL_RAG_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content":
                    user_prompt,
            },
        ],
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    return (
        response.json()[
            "message"
        ][
            "content"
        ]
    )