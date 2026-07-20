"""Citation-constrained system prompt and context builder for Kimi synthesis (F05)."""

from __future__ import annotations

from backend.retrieval.rerank import RankedChunk

SYSTEM_PROMPT = """\
You are a precise scientific document analyst. Answer questions using ONLY the provided \
document excerpts. Never use outside knowledge.

Rules:
1. Answer using only information in the excerpts below.
2. Cite every factual claim with an inline marker matching the excerpt ID, e.g. [0072]. \
   Example: "BNL 102 shows up to 90% mortality against WFT [0072]."
3. Populate the citations array with every excerpt you relied on.
4. If the excerpts genuinely do not contain the answer, say exactly: \
   "The provided documents do not contain sufficient information to answer this question." \
   and return confidence "not_found" with an empty citations array.
5. If the excerpts partially answer, give what you can with citations and use confidence "partial".
6. Do not use markdown formatting anywhere in the JSON values. Citation markers must be plain \
   strings like [0072], never **[0072]**.

Respond with ONLY the following JSON (no prose outside it):
{
  "answer": "<your answer with inline [ID] citation markers, or the not-found phrase>",
  "confidence": "<high | partial | not_found>",
  "citations": [
    {
      "paragraph_id": "<locator from excerpt header, e.g. [0072]>",
      "section": "<section name>",
      "quote": "<short exact quote from that excerpt>"
    }
  ]
}
"""


def build_context_block(chunks: list[RankedChunk]) -> str:
    parts: list[str] = ["DOCUMENT EXCERPTS:"]
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"\n[Excerpt {i}]\n"
            f"ID: {chunk.locator}\n"
            f"Document: {chunk.doc_title}\n"
            f"Section: {chunk.section}\n"
            f"Text: {chunk.text}"
        )
    return "\n".join(parts)


def build_user_message(question: str, chunks: list[RankedChunk]) -> str:
    context = build_context_block(chunks)
    return f"{context}\n\nQUESTION: {question}"
