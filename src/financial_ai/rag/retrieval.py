from financial_ai.rag.bm25_store import (
    BM25Store,
)

from financial_ai.rag.schemas import (
    RAGChunk,
)

from financial_ai.rag.vector_store import (
    vector_search,
)


def deduplicate_chunks(
    chunks: list[RAGChunk],
) -> list[RAGChunk]:

    seen = set()
    output = []

    for chunk in chunks:

        # Filing + section + beginning of text
        # gives us a practical near-duplicate key.
        key = (
            chunk.accession_number,
            chunk.section.lower().strip(),
            " ".join(
                chunk.text
                .lower()
                .split()[:40]
            ),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(chunk)

    return output

def _qdrant_to_chunk(
    point,
) -> RAGChunk:

    payload = point.payload

    return RAGChunk(
        chunk_id=str(
            point.id
        ),
        ticker=payload[
            "ticker"
        ],
        filing_type=payload[
            "filing_type"
        ],
        filed_date=payload[
            "filed_date"
        ],
        accession_number=payload[
            "accession_number"
        ],
        section=payload[
            "section"
        ],
        text=payload[
            "text"
        ],
        source_url=payload[
            "source_url"
        ],
        chunk_index=payload[
            "chunk_index"
        ],
    )


def hybrid_search(
    query: str,
    ticker: str,
    all_chunks: list[RAGChunk],
    top_k: int = 20,
) -> list[RAGChunk]:
    """
    Combine dense + BM25 retrieval using
    Reciprocal Rank Fusion.
    """

    vector_results = vector_search(
        query=query,
        ticker=ticker,
        top_k=top_k,
    )

    bm25 = BM25Store(
        all_chunks
    )

    bm25_results = bm25.search(
        query=query,
        top_k=top_k,
    )

    scores = {}

    chunk_map = {}

    # Reciprocal Rank Fusion constant.
    k = 60

    for rank, point in enumerate(
        vector_results,
        start=1,
    ):

        chunk = (
            _qdrant_to_chunk(
                point
            )
        )

        chunk_map[
            chunk.chunk_id
        ] = chunk

        scores[
            chunk.chunk_id
        ] = (
            scores.get(
                chunk.chunk_id,
                0.0,
            )
            +
            1 / (k + rank)
        )

    for rank, (
        chunk,
        _,
    ) in enumerate(
        bm25_results,
        start=1,
    ):

        chunk_map[
            chunk.chunk_id
        ] = chunk

        scores[
            chunk.chunk_id
        ] = (
            scores.get(
                chunk.chunk_id,
                0.0,
            )
            +
            1 / (k + rank)
        )

    ranked = sorted(
        scores.items(),
        key=lambda item:
            item[1],
        reverse=True,
    )

    return [
        chunk_map[
            chunk_id
        ]
        for (
            chunk_id,
            _
        ) in ranked[
            :top_k
        ]
    ]