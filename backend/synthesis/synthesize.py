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
KIMI_MAX_TOKENS = 1400

_LOCATOR_RE = re.compile(r"\[\d{4}\]")
_BOLD_LOCATOR_RE = re.compile(r"\*\*(\[\d{4}\])\*\*")


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

    def clean_strings(value):
        if isinstance(value, str):
            return _BOLD_LOCATOR_RE.sub(r"\1", value)
        if isinstance(value, list):
            return [clean_strings(item) for item in value]
        if isinstance(value, dict):
            return {key: clean_strings(item) for key, item in value.items()}
        return value

    def parse_json(candidate: str) -> dict:
        candidate = _BOLD_LOCATOR_RE.sub(r"\1", candidate.strip())
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = json.loads(candidate, strict=False)
        return clean_strings(parsed)

    # Try markdown code fences first
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        return parse_json(match.group(1))
    # Try bare JSON parse
    try:
        return parse_json(raw)
    except (json.JSONDecodeError, ValueError):
        pass
    # Extract first {...} block from text that has prose around it
    brace_match = re.search(r"\{[\s\S]*\}", raw)
    if brace_match:
        return parse_json(brace_match.group())
    raise ValueError("No JSON found in response")


def _normalize_locator(value: str) -> str:
    value = value.strip()
    value = _BOLD_LOCATOR_RE.sub(r"\1", value)
    match = _LOCATOR_RE.search(value)
    return match.group() if match else value


def _clean_answer_text(value: str) -> str:
    value = value.strip()
    fence_match = re.search(r"```(?:json|text)?\s*([\s\S]+?)\s*```", value)
    if fence_match:
        value = fence_match.group(1).strip()
    return _BOLD_LOCATOR_RE.sub(r"\1", value).strip()


def _answer_text_from_raw(raw: str) -> str:
    raw = raw.strip()
    if not raw:
        raise ValueError("Kimi returned empty content")

    # Backward compatibility for any provider response that still comes back as JSON.
    try:
        data = _extract_json(raw)
    except (json.JSONDecodeError, ValueError):
        return _clean_answer_text(raw)

    return _clean_answer_text(data.get("answer", raw))


def _fallback_citations(chunks: list[RankedChunk], limit: int = 3) -> list[Citation]:
    return [
        Citation(
            paragraph_id=chunk.locator,
            section=chunk.section,
            quote=chunk.text[:360],
            score=chunk.rerank_score,
        )
        for chunk in chunks[:limit]
    ]


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
        max_tokens=KIMI_MAX_TOKENS,
        extra_body={"thinking": {"type": "disabled"}},
    )

    raw = response.choices[0].message.content or ""
    logger.info("Kimi raw response (first 500 chars): %r", raw[:500])

    try:
        answer = _answer_text_from_raw(raw)
    except ValueError as exc:
        logger.warning("Kimi response could not be used: %s", exc)
        raise

    # Build citation objects enriched with rerank scores
    chunk_by_locator = {c.locator: c for c in chunks}
    score_by_locator = {c.locator: c.rerank_score for c in chunks}

    citations: list[Citation] = []
    seen: set[str] = set()
    for match in _LOCATOR_RE.finditer(answer):
        pid = _normalize_locator(match.group())
        if pid in seen:
            continue
        seen.add(pid)
        chunk = chunk_by_locator.get(pid)
        if chunk:
            citations.append(
                Citation(
                    paragraph_id=pid,
                    section=chunk.section,
                    quote=chunk.text[:220],
                    score=score_by_locator.get(pid, chunk.rerank_score),
                )
            )

    if NOT_FOUND_ANSWER in answer:
        return SynthesisResult(answer=NOT_FOUND_ANSWER, confidence="not_found", citations=[])

    if not citations:
        logger.warning("Kimi answer omitted required citation locators. Raw: %r", raw[:500])
        return SynthesisResult(
            answer=NOT_FOUND_ANSWER,
            confidence="not_found",
            citations=[],
        )

    return SynthesisResult(answer=answer, confidence="high", citations=citations)
