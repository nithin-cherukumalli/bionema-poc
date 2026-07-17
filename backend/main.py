"""FastAPI application: /query, /health endpoints (F07a)."""

from __future__ import annotations

import time
from collections import OrderedDict
from copy import deepcopy

import voyageai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

from backend.config import get_settings
from backend.ingest.embed_upsert import COLLECTION_NAME, build_qdrant_client
from backend.retrieval.hybrid_query import hybrid_search
from backend.retrieval.rerank import RankedChunk, rerank
from backend.synthesis.synthesize import Citation, SynthesisResult, synthesize

settings = get_settings()
QUERY_CACHE_TTL_SECONDS = 60 * 60
QUERY_CACHE_MAX_ITEMS = 128
RANKED_CACHE_TTL_SECONDS = 5 * 60
_query_cache: OrderedDict[str, tuple[float, QueryResponse]] = OrderedDict()
_ranked_cache: OrderedDict[str, tuple[float, list[RankedChunk]]] = OrderedDict()

app = FastAPI(
    title="Bionema Retrieval POC",
    description="Hybrid RAG for patent/regulatory documents with clause-level citations.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
        "http://localhost:3000",
        "http://localhost:5173",
        "https://*.vercel.app",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    paragraph_id: str
    section: str
    quote: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    confidence: str
    citations: list[CitationOut]


def _normalize_cache_key(question: str) -> str:
    return " ".join(question.casefold().split())


def _get_cached_response(cache_key: str) -> QueryResponse | None:
    cached = _query_cache.get(cache_key)
    if cached is None:
        return None

    created_at, response = cached
    if time.monotonic() - created_at > QUERY_CACHE_TTL_SECONDS:
        _query_cache.pop(cache_key, None)
        return None

    _query_cache.move_to_end(cache_key)
    return deepcopy(response)


def _set_cached_response(cache_key: str, response: QueryResponse) -> None:
    _query_cache[cache_key] = (time.monotonic(), deepcopy(response))
    _query_cache.move_to_end(cache_key)
    while len(_query_cache) > QUERY_CACHE_MAX_ITEMS:
        _query_cache.popitem(last=False)


def _should_cache_response(response: QueryResponse) -> bool:
    if response.citations:
        return True
    return response.confidence == "not_found"


def _get_cached_ranked(cache_key: str) -> list[RankedChunk] | None:
    cached = _ranked_cache.get(cache_key)
    if cached is None:
        return None

    created_at, ranked = cached
    if time.monotonic() - created_at > RANKED_CACHE_TTL_SECONDS:
        _ranked_cache.pop(cache_key, None)
        return None

    _ranked_cache.move_to_end(cache_key)
    return deepcopy(ranked)


def _set_cached_ranked(cache_key: str, ranked: list[RankedChunk]) -> None:
    _ranked_cache[cache_key] = (time.monotonic(), deepcopy(ranked))
    _ranked_cache.move_to_end(cache_key)
    while len(_ranked_cache) > QUERY_CACHE_MAX_ITEMS:
        _ranked_cache.popitem(last=False)


def _get_or_build_ranked(question: str, cache_key: str) -> list[RankedChunk]:
    ranked = _get_cached_ranked(cache_key)
    if ranked is not None:
        return ranked

    candidates = hybrid_search(question, settings=settings)
    ranked = rerank(question, candidates, settings=settings)
    _set_cached_ranked(cache_key, ranked)
    return ranked


def _short_quote(text: str, max_chars: int = 360) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _retrieval_fallback_result(ranked: list[RankedChunk]) -> SynthesisResult:
    """Return citable evidence when the synthesis provider is temporarily unavailable."""
    if not ranked:
        return SynthesisResult(
            answer="not found",
            confidence="not_found",
            citations=[],
        )

    top_chunks = ranked[:3]
    citations = [
        Citation(
            paragraph_id=chunk.locator,
            section=chunk.section,
            quote=_short_quote(chunk.text),
            score=chunk.rerank_score,
        )
        for chunk in top_chunks
    ]
    evidence = " ".join(f"{citation.quote} {citation.paragraph_id}" for citation in citations)

    return SynthesisResult(
        answer=(
            "Kimi synthesis is temporarily unavailable. Retrieved evidence from the indexed "
            f"documents: {evidence}"
        ),
        confidence="partial",
        citations=citations,
    )


def _evidence_preview_result(ranked: list[RankedChunk]) -> SynthesisResult:
    if not ranked:
        return SynthesisResult(
            answer="not found",
            confidence="not_found",
            citations=[],
        )

    top_chunks = ranked[:3]
    citations = [
        Citation(
            paragraph_id=chunk.locator,
            section=chunk.section,
            quote=_short_quote(chunk.text),
            score=chunk.rerank_score,
        )
        for chunk in top_chunks
    ]
    evidence = " ".join(f"{citation.quote} {citation.paragraph_id}" for citation in citations)
    return SynthesisResult(
        answer=f"Retrieved evidence while the final cited answer is being synthesized: {evidence}",
        confidence="partial",
        citations=citations,
    )


def _to_query_response(result: SynthesisResult) -> QueryResponse:
    return QueryResponse(
        answer=result.answer,
        confidence=result.confidence,
        citations=[
            CitationOut(
                paragraph_id=c.paragraph_id,
                section=c.section,
                quote=c.quote,
                score=c.score,
            )
            for c in result.citations
        ],
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    cache_key = _normalize_cache_key(question)
    cached_response = _get_cached_response(cache_key)
    if cached_response is not None:
        return cached_response

    ranked = _get_or_build_ranked(question, cache_key)
    try:
        result: SynthesisResult = synthesize(question, ranked, settings=settings)
    except Exception:
        result = _retrieval_fallback_result(ranked)
    else:
        if ranked and result.confidence == "partial" and not result.citations:
            result = _evidence_preview_result(ranked)

    response = _to_query_response(result)
    if _should_cache_response(response):
        _set_cached_response(cache_key, response)
    return response


@app.post("/query/evidence", response_model=QueryResponse)
async def query_evidence(request: QueryRequest) -> QueryResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    cache_key = _normalize_cache_key(question)
    cached_response = _get_cached_response(cache_key)
    if cached_response is not None:
        return cached_response

    ranked = _get_or_build_ranked(question, cache_key)
    return _to_query_response(_evidence_preview_result(ranked))


@app.get("/health")
async def health() -> dict:
    status: dict[str, str] = {"status": "ok"}

    # Qdrant
    try:
        client = build_qdrant_client(settings)
        client.get_collection(COLLECTION_NAME)
        status["qdrant"] = "ok"
    except Exception as exc:
        status["qdrant"] = f"error: {exc}"
        status["status"] = "degraded"

    # Voyage
    try:
        voyage_client = voyageai.Client(api_key=settings.voyage_api_key)
        voyage_client.embed(["health"], model="voyage-3-large")
        status["voyage"] = "ok"
    except Exception as exc:
        status["voyage"] = f"error: {exc}"
        status["status"] = "degraded"

    # Kimi
    try:
        kimi_client = OpenAI(api_key=settings.kimi_api_key, base_url=settings.kimi_base_url)
        kimi_client.chat.completions.create(
            model=settings.kimi_model,
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        status["kimi"] = "ok"
    except Exception as exc:
        status["kimi"] = f"error: {exc}"
        status["status"] = "degraded"

    return status
