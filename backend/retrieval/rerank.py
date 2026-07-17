"""Voyage rerank-2 reranker: fused candidates → top cited evidence chunks (F04)."""

from __future__ import annotations

from dataclasses import dataclass

import voyageai

from backend.config import Settings, get_settings
from backend.retrieval.hybrid_query import RetrievedChunk

RERANK_MODEL = "rerank-2"
TOP_N = 5
MIN_RERANK_SCORE = 0.3


@dataclass(frozen=True)
class RankedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    locator: str
    locator_type: str
    section: str
    text: str
    token_count: int
    rerank_score: float


def rerank(
    query: str,
    candidates: list[RetrievedChunk],
    *,
    settings: Settings | None = None,
    top_n: int = TOP_N,
    min_score: float = MIN_RERANK_SCORE,
) -> list[RankedChunk]:
    if not candidates:
        return []

    resolved = settings or get_settings()
    client = voyageai.Client(api_key=resolved.voyage_api_key)

    documents = [c.text for c in candidates]
    result = client.rerank(query, documents, model=RERANK_MODEL, top_k=top_n)

    ranked: list[RankedChunk] = []
    for item in result.results:
        if item.relevance_score < min_score:
            continue
        candidate = candidates[item.index]
        ranked.append(
            RankedChunk(
                chunk_id=candidate.chunk_id,
                doc_id=candidate.doc_id,
                doc_title=candidate.doc_title,
                locator=candidate.locator,
                locator_type=candidate.locator_type,
                section=candidate.section,
                text=candidate.text,
                token_count=candidate.token_count,
                rerank_score=item.relevance_score,
            )
        )
    return ranked
