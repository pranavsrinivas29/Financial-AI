import pandas as pd

from financial_ai.data.sec_data import (
    download_filing_document,
)

from financial_ai.data.temporal import (
    get_latest_10q,
    get_latest_8k,
)

from financial_ai.rag.chunking import (
    build_rag_chunks,
)

from financial_ai.rag.llm import (
    generate_answer,
)

from financial_ai.rag.parser import (
    parse_sec_html,
)

from financial_ai.rag.retrieval import (
    hybrid_search,deduplicate_chunks
)

from financial_ai.rag.reranker import (
    rerank_chunks,
)

from financial_ai.rag.vector_store import (
    index_chunks,
)


def prepare_point_in_time_filings(
    ticker: str,
    as_of_date: str,
):

    # =========================================
    # REUSE PHASE 2
    # =========================================

    latest_10q = get_latest_10q(
        ticker=ticker,
        as_of_date=as_of_date,
    )

    latest_8k = get_latest_8k(
        ticker=ticker,
        as_of_date=as_of_date,
    )

    filings = [
        latest_10q,
        latest_8k,
    ]

    all_chunks = []

    for filing in filings:

        # Hard temporal assertion.
        assert (
            pd.Timestamp(
                filing[
                    "filed_date"
                ]
            )
            <= pd.Timestamp(
                as_of_date
            )
        )

        path = (
            download_filing_document(
                filing
            )
        )

        text = parse_sec_html(
            path
        )

        chunks = build_rag_chunks(
            text=text,

            ticker=ticker.upper(),

            filing_type=filing[
                "form"
            ],

            filed_date=str(
                filing[
                    "filed_date"
                ].date()
            ),

            accession_number=(
                filing[
                    "accession_number"
                ]
            ),

            source_url=(
                filing[
                    "document_url"
                ]
            ),
        )

        all_chunks.extend(
            chunks
        )

    index_chunks(
        all_chunks
    )

    return all_chunks

def ask_financial_rag(
    ticker: str,
    as_of_date: str,
    question: str,
) -> dict:

    chunks = (
        prepare_point_in_time_filings(
            ticker=ticker,
            as_of_date=as_of_date,
        )
    )

    retrieved = hybrid_search(
        query=question,
        ticker=ticker,
        all_chunks=chunks,
        top_k=20,
    )

    retrieved = deduplicate_chunks(
        retrieved
    )
    reranked = rerank_chunks(
        query=question,
        chunks=retrieved,
        top_k=5,
    )

    contexts = []

    sources = []

    for index, chunk in enumerate(
        reranked,
        start=1,
    ):

        contexts.append(
            f"""
[Source {index}]
Filing: {chunk.filing_type}
Filed: {chunk.filed_date}
Section: {chunk.section}

{chunk.text}
"""
        )

        sources.append(
            {
                "source_id":
                    index,

                "filing_type":
                    chunk.filing_type,

                "filed_date":
                    chunk.filed_date,

                "section":
                    chunk.section,

                "source_url":
                    chunk.source_url,

                "chunk_id":
                    chunk.chunk_id,
            }
        )

    context = "\n\n".join(
        contexts
    )

    answer = generate_answer(
        question=question,
        context=context,
    )

    return {
        "ticker":
            ticker.upper(),

        "as_of_date":
            as_of_date,

        "question":
            question,

        "answer":
            answer,

        "sources":
            sources,
    }