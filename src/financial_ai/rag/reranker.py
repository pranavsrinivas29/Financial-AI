from functools import lru_cache

from sentence_transformers import (
    CrossEncoder,
)

from financial_ai.rag.schemas import (
    RAGChunk,
)


RERANK_MODEL = (
    "cross-encoder/"
    "ms-marco-MiniLM-L-6-v2"
)


@lru_cache(maxsize=1)
def get_reranker():

    return CrossEncoder(
        RERANK_MODEL
    )


def rerank_chunks(
    query: str,
    chunks: list[RAGChunk],
    top_k: int = 5,
) -> list[RAGChunk]:

    if not chunks:
        return []

    model = get_reranker()

    pairs = [
        (
            query,
            chunk.text,
        )
        for chunk in chunks
    ]

    scores = model.predict(
        pairs
    )

    ranked = sorted(
        zip(
            chunks,
            scores,
        ),
        key=lambda item:
            float(item[1]),
        reverse=True,
    )

    return [
        chunk
        for (
            chunk,
            _
        ) in ranked[
            :top_k
        ]
    ]