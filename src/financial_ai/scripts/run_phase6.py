from financial_ai.agents.graph import (
    analyze_financial_query,
)


def main():

    result = analyze_financial_query(
        ticker="AAPL",
        as_of_date="2026-07-15",
        query=(
            "Give me an overall financial "
            "outlook for Apple."
        ),
    )

    print(
        "\n================================="
    )

    print(
        "AGENTIC FINANCIAL AI"
    )

    print(
        "================================="
    )

    print(
        f"\nTicker: "
        f"{result['ticker']}"
    )

    print(
        f"\nAs of: "
        f"{result['as_of_date']}"
    )

    print(
        "\nAgents selected:"
    )

    print(
        f"ML  : "
        f"{result.get('use_ml')}"
    )

    print(
        f"NLP : "
        f"{result.get('use_nlp')}"
    )

    print(
        f"RAG : "
        f"{result.get('use_rag')}"
    )

    print(
        "\n================================="
    )

    print(
        "FINAL ANALYSIS"
    )

    print(
        "=================================\n"
    )

    print(
        result["final_answer"]
    )
    
    print("\n--- DEBUG ---")

    print("\nML RESULT:")
    print(result.get("ml_result"))

    print("\nML ERROR:")
    print(result.get("ml_error"))

    print("\nNLP RESULT:")
    print(result.get("nlp_result"))

    print("\nNLP ERROR:")
    print(result.get("nlp_error"))

    print("\nRAG ERROR:")
    print(result.get("rag_error"))


if __name__ == "__main__":
    main()