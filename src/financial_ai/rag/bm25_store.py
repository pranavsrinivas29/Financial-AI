import re

from rank_bm25 import BM25Okapi

from financial_ai.rag.schemas import (
    RAGChunk,
)


TOKEN_PATTERN = re.compile(
    r"\b\w+\b"
)


def tokenize(
    text: str,
) -> list[str]:

    return TOKEN_PATTERN.findall(
        text.lower()
    )


class BM25Store:

    def __init__(
        self,
        chunks: list[RAGChunk],
    ):

        self.chunks = chunks

        tokenized = [
            tokenize(
                chunk.text
            )
            for chunk in chunks
        ]

        self.index = BM25Okapi(
            tokenized
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
    ):

        query_tokens = tokenize(
            query
        )

        scores = (
            self.index
            .get_scores(
                query_tokens
            )
        )

        ranked_indices = (
            scores
            .argsort()[::-1]
            [:top_k]
        )

        results = []

        for index in ranked_indices:

            results.append(
                (
                    self.chunks[
                        index
                    ],
                    float(
                        scores[index]
                    ),
                )
            )

        return results