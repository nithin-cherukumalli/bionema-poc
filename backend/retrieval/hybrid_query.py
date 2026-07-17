"""Hybrid retrieval: dense + sparse fusion via Qdrant RRF (F04)."""

from __future__ import annotations

from dataclasses import dataclass

import voyageai
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

from backend.config import Settings, get_settings
from backend.ingest.embed_upsert import (
    COLLECTION_NAME,
    SPARSE_MODEL_NAME,
    VOYAGE_EMBED_MODEL,
    build_qdrant_client,
)

TOP_K_PREFETCH = 16
VOYAGE_INPUT_TYPE = "query"

_sparse_model: SparseTextEmbedding | None = None


def _get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    return _sparse_model


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    locator: str
    locator_type: str
    section: str
    text: str
    token_count: int
    score: float


def embed_query_dense(query: str, settings: Settings) -> list[float]:
    client = voyageai.Client(api_key=settings.voyage_api_key)
    result = client.embed([query], model=VOYAGE_EMBED_MODEL, input_type=VOYAGE_INPUT_TYPE)
    return result.embeddings[0]


def embed_query_sparse(query: str) -> models.SparseVector:
    model = _get_sparse_model()
    embeddings = list(model.embed([query]))
    emb = embeddings[0]
    return models.SparseVector(
        indices=emb.indices.tolist(),
        values=emb.values.tolist(),
    )


def hybrid_search(
    query: str,
    *,
    settings: Settings | None = None,
    collection_name: str = COLLECTION_NAME,
    top_k: int = TOP_K_PREFETCH,
) -> list[RetrievedChunk]:
    resolved = settings or get_settings()
    client = build_qdrant_client(resolved)

    dense_vec = embed_query_dense(query, resolved)
    sparse_vec = embed_query_sparse(query)

    results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=dense_vec,
                using="dense",
                limit=top_k,
            ),
            models.Prefetch(
                query=sparse_vec,
                using="sparse",
                limit=top_k,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )

    chunks: list[RetrievedChunk] = []
    for point in results.points:
        p = point.payload or {}
        chunks.append(
            RetrievedChunk(
                chunk_id=p.get("chunk_id", ""),
                doc_id=p.get("doc_id", ""),
                doc_title=p.get("doc_title", ""),
                locator=p.get("locator", ""),
                locator_type=p.get("locator_type", ""),
                section=p.get("section", ""),
                text=p.get("text", ""),
                token_count=p.get("token_count", 0),
                score=point.score,
            )
        )
    return chunks
