from pathlib import Path

from qdrant_client import (
    QdrantClient,
)

from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from financial_ai.config.settings import (
    settings,
)

from financial_ai.rag.embeddings import (
    embed_documents,
    embed_query,
)

from financial_ai.rag.schemas import (
    RAGChunk,
)


COLLECTION_NAME = (
    "financial_sec_filings"
)


def get_qdrant_client():

    path = (
        settings.DATA_DIR
        / "rag"
        / "qdrant"
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return QdrantClient(
        path=str(path)
    )


def ensure_collection():

    client = get_qdrant_client()

    collections = (
        client.get_collections()
        .collections
    )

    existing = {
        item.name
        for item in collections
    }

    if (
        COLLECTION_NAME
        not in existing
    ):

        client.create_collection(
            collection_name=(
                COLLECTION_NAME
            ),
            vectors_config=(
                VectorParams(
                    size=768,
                    distance=(
                        Distance.COSINE
                    ),
                )
            ),
        )

    return client


def index_chunks(
    chunks: list[RAGChunk],
):

    if not chunks:
        return

    client = ensure_collection()

    texts = [
        chunk.text
        for chunk in chunks
    ]

    vectors = embed_documents(
        texts
    )

    points = []

    for chunk, vector in zip(
        chunks,
        vectors,
    ):

        payload = {
            "ticker":
                chunk.ticker,

            "filing_type":
                chunk.filing_type,

            "filed_date":
                chunk.filed_date,

            "accession_number":
                chunk.accession_number,

            "section":
                chunk.section,

            "text":
                chunk.text,

            "source_url":
                chunk.source_url,

            "chunk_index":
                chunk.chunk_index,
        }

        points.append(
            PointStruct(
                id=chunk.chunk_id,
                vector=(
                    vector.tolist()
                ),
                payload=payload,
            )
        )

    client.upsert(
        collection_name=(
            COLLECTION_NAME
        ),
        points=points,
    )
    
def vector_search(
    query: str,
    ticker: str,
    top_k: int = 20,
):

    from qdrant_client.models import (
        FieldCondition,
        Filter,
        MatchValue,
    )

    client = ensure_collection()

    vector = embed_query(
        query
    )

    result = client.query_points(
        collection_name=(
            COLLECTION_NAME
        ),
        query=vector.tolist(),
        query_filter=Filter(
            must=[
                FieldCondition(
                    key="ticker",
                    match=MatchValue(
                        value=ticker.upper()
                    ),
                )
            ]
        ),
        limit=top_k,
        with_payload=True,
    )

    return result.points