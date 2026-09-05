from dataclasses import dataclass
from typing import Optional


@dataclass
class RAGChunk:
    chunk_id: str
    ticker: str
    filing_type: str
    filed_date: str
    accession_number: str
    section: str
    text: str
    source_url: str
    chunk_index: int


@dataclass
class RetrievedChunk:
    chunk: RAGChunk
    score: float
    vector_score: Optional[float] = None
    bm25_score: Optional[float] = None
    rerank_score: Optional[float] = None