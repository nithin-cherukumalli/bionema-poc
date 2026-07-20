"""Kimi-backed synthesis client (F05).

Calls Kimi API with citation-constrained prompt and parses the structured response.
Every factual claim must trace to a retrieved chunk or the model says "not found."
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from openai import OpenAI

logger = logging.getLogger(__name__)

from backend.config import Settings, get_settings
from backend.retrieval.rerank import RankedChunk
from backend.synthesis.prompt import SYSTEM_PROMPT, build_user_message

NOT_FOUND_ANSWER = "The provided documents do not contain sufficient information to answer this question."
KIMI_MAX_TOKENS = 600
# The deployed Kimi model accepts only its provider default, 1. Passing a
# lower value causes a 400 before synthesis can begin.
KIMI_TEMPERATURE = 1

_LOCATOR_RE = re.compile(r"\[\d{4}\]")


@dataclass(frozen=True)
class Citation:
    paragraph_id: str
    section: str
    quote: str
    score: float = 0.0


@dataclass(frozen=True)
class SynthesisResult:
    answer: str
    confidence: str
    citations: list[Citation]


def create_kimi_client(settings: Settings | None = None) -> OpenAI:
    resolved_settings = settings or get_settings()
    return OpenAI(
        api_key=resolved_settings.kimi_api_key,
        base_url=resolved_settings.kimi_base_url,
    )


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    # Try markdown code fences first
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        return json.loads(match.group(1))
    # Try bare JSON parse
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    # Extract first {...} block from text that has prose around it
    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if brace_match:
        return json.loads(brace_match.group())
    raise ValueError("No JSON found in response")


def synthesize(
    question: str,
    chunks: list[RankedChunk],
    *,
    settings: Settings | None = None,
) -> SynthesisResult:
    if not chunks:
        return SynthesisResult(
            answer=NOT_FOUND_ANSWER,
            confidence="not_found",
            citations=[],
        )

    resolved = settings or get_settings()
    client = create_kimi_client(resolved)
    user_message = build_user_message(question, chunks)

    response = client.chat.completions.create(
        model=resolved.kimi_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=KIMI_TEMPERATURE,
        max_tokens=KIMI_MAX_TOKENS,
    )

    raw = response.choices[0].message.content or ""
    logger.info("Kimi raw response (first 500 chars): %s", raw[:500])

    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        logger.warning("Failed to parse Kimi JSON. Raw: %s", raw[:500])
        return SynthesisResult(
            answer=raw or NOT_FOUND_ANSWER,
            confidence="partial",
            citations=[],
        )

    answer = data.get("answer", NOT_FOUND_ANSWER)
    confidence = data.get("confidence", "partial")
    if confidence not in {"high", "partial", "not_found"}:
        confidence = "partial"

    # Build citation objects enriched with rerank scores
    chunk_by_locator = {c.locator: c for c in chunks}
    score_by_locator = {c.locator: c.rerank_score for c in chunks}

    raw_citations = data.get("citations") or []
    citations: list[Citation] = []
    for raw_cit in raw_citations:
        pid = raw_cit.get("paragraph_id", "")
        citations.append(
            Citation(
                paragraph_id=pid,
                section=raw_cit.get("section", ""),
                quote=raw_cit.get("quote", ""),
                score=score_by_locator.get(pid, 0.0),
            )
        )

    # Recovery: if citations is empty but the answer is real, scan the answer text
    # for [NNNN] locator patterns and build citations from the matched ranked chunks.
    # This handles the case where Kimi includes inline markers but forgets the array.
    if not citations and answer and NOT_FOUND_ANSWER not in answer:
        seen: set[str] = set()
        for match in _LOCATOR_RE.finditer(answer):
            pid = match.group()
            if pid in seen:
                continue
            seen.add(pid)
            chunk = chunk_by_locator.get(pid)
            if chunk:
                citations.append(
                    Citation(
                        paragraph_id=pid,
                        section=chunk.section,
                        quote=chunk.text[:120],
                        score=chunk.rerank_score,
                    )
                )

    return SynthesisResult(answer=answer, confidence=confidence, citations=citations)
