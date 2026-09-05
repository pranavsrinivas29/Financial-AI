from functools import lru_cache

from sentence_transformers import (
    SentenceTransformer,
)


EMBEDDING_MODEL = (
    "BAAI/bge-base-en-v1.5"
)


@lru_cache(maxsize=1)
def get_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


def embed_documents(
    texts: list[str],
):

    model = get_embedding_model()

    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


def embed_query(
    query: str,
):

    model = get_embedding_model()

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    return embedding[0]