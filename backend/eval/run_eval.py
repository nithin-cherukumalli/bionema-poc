"""Evaluation harness: Recall@5, MRR, groundedness (F06).

Runs 25 hand-labeled Q/A pairs against the live retrieval + synthesis pipeline
and writes a timestamped results report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.config import get_settings  # noqa: E402
from backend.retrieval.hybrid_query import hybrid_search  # noqa: E402
from backend.retrieval.rerank import rerank  # noqa: E402
from backend.synthesis.synthesize import NOT_FOUND_ANSWER, synthesize  # noqa: E402

QA_SET_PATH = Path(__file__).parent / "qa_set.json"
TOP_K = 5


def load_qa_set(path: Path = QA_SET_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def recall_at_k(retrieved_locators: list[str], expected_locators: list[str], k: int = 5) -> float:
    if not expected_locators:
        return 0.0
    top_k = set(retrieved_locators[:k])
    return 1.0 if any(loc in top_k for loc in expected_locators) else 0.0


def reciprocal_rank(retrieved_locators: list[str], expected_locators: list[str]) -> float:
    if not expected_locators:
        return 0.0
    for i, loc in enumerate(retrieved_locators, 1):
        if loc in expected_locators:
            return 1.0 / i
    return 0.0


def groundedness_judge(answer: str, citations: list[dict], question: str) -> float:
    """Heuristic groundedness check (no LLM call to keep eval cost low).

    Scores:
    - 1.0: at least one citation's paragraph_id appears inline in the answer text
    - 0.8: citations array non-empty but no inline marker (model cited sources, partial format)
    - 0.0: citations empty and answer is a real answer (hallucination risk)
    - not scored here: unanswerable questions (handled separately in run_evaluation)
    """
    # Only treat it as a genuine "not found" if the answer IS the not-found message
    # (exact match, stripped). A substring check fires too eagerly on partial answers.
    if answer.strip() == NOT_FOUND_ANSWER.strip():
        return 0.0  # not_found — scored in unanswerable path

    if not citations:
        return 0.0  # answered but no citations — hallucination risk

    # Best case: locator appears inline in the answer text
    cited_ids = {c.get("paragraph_id", "") for c in citations}
    for cid in cited_ids:
        if cid and cid in answer:
            return 1.0

    # Citations returned but not embedded inline
    return 0.8


# Voyage free tier: 3 RPM. Each eval question makes 2 calls (embed + rerank).
# Sleep 25 s between questions so we stay within 3 RPM regardless of call ordering.
VOYAGE_INTER_QUESTION_SLEEP = 25
TRANSIENT_ERROR_MARKERS = (
    "429",
    "rate limit",
    "ratelimit",
    "engine_overloaded",
    "engine is currently overloaded",
    "overloaded",
)
DEFAULT_RETRIES = 3
DEFAULT_RETRY_SLEEP = 60


def is_transient_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in TRANSIENT_ERROR_MARKERS)


def run_with_retries(label: str, fn, *, retries: int, retry_sleep: float, verbose: bool):
    for attempt in range(1, retries + 2):
        try:
            return fn()
        except Exception as exc:
            if attempt > retries + 1 or not is_transient_error(exc):
                raise
            if verbose:
                print(
                    f"    transient error during {label}: {exc}; "
                    f"retrying in {retry_sleep}s ({attempt}/{retries})"
                )
            time.sleep(retry_sleep)


def run_evaluation(
    qa_items: list[dict],
    *,
    settings=None,
    verbose: bool = False,
    inter_question_sleep: float = VOYAGE_INTER_QUESTION_SLEEP,
    retries: int = DEFAULT_RETRIES,
    retry_sleep: float = DEFAULT_RETRY_SLEEP,
    retrieval_only: bool = False,
) -> dict:
    resolved = settings or get_settings()
    results = []

    for item in qa_items:
        qid = item["id"]
        question = item["question"]
        expected_locs = item.get("expected_locators") or []
        category = item.get("category", "unknown")

        if verbose:
            print(f"  [{qid}] {question[:60]}...")

        try:
            candidates = run_with_retries(
                "hybrid_search",
                lambda: hybrid_search(question, settings=resolved, top_k=20),
                retries=retries,
                retry_sleep=retry_sleep,
                verbose=verbose,
            )
            ranked = run_with_retries(
                "rerank",
                lambda: rerank(question, candidates, settings=resolved, top_n=TOP_K),
                retries=retries,
                retry_sleep=retry_sleep,
                verbose=verbose,
            )

            retrieved_locs = [c.locator for c in ranked]

            if retrieval_only:
                rec_at_5 = recall_at_k(retrieved_locs, expected_locs, k=TOP_K)
                mrr = reciprocal_rank(retrieved_locs, expected_locs)
                ground = 0.0
                confidence = "retrieval_only"
                answer_snippet = ""
            else:
                synthesis = run_with_retries(
                    "synthesis",
                    lambda: synthesize(question, ranked, settings=resolved),
                    retries=retries,
                    retry_sleep=retry_sleep,
                    verbose=verbose,
                )
                citations_out = [
                    {
                        "paragraph_id": c.paragraph_id,
                        "section": c.section,
                        "quote": c.quote,
                    }
                    for c in synthesis.citations
                ]

                if category == "unanswerable":
                    # Correct if confidence is not_found
                    correct_refusal = synthesis.confidence == "not_found"
                    rec_at_5 = 1.0 if correct_refusal else 0.0
                    mrr = 1.0 if correct_refusal else 0.0
                    ground = 0.5 if correct_refusal else 0.0
                else:
                    rec_at_5 = recall_at_k(retrieved_locs, expected_locs, k=TOP_K)
                    mrr = reciprocal_rank(retrieved_locs, expected_locs)
                    ground = groundedness_judge(synthesis.answer, citations_out, question)
                confidence = synthesis.confidence
                answer_snippet = synthesis.answer[:120]

            if category == "unanswerable" and retrieval_only:
                # Retrieval-only mode cannot prove a correct refusal because no
                # synthesis/refusal decision is made.
                rec_at_5 = 0.0
                mrr = 0.0
                ground = 0.0
            elif category == "unanswerable":
                # Correct if confidence is not_found
                pass

            results.append(
                {
                    "id": qid,
                    "question": question,
                    "category": category,
                    "expected_locators": expected_locs,
                    "retrieved_locators": retrieved_locs,
                    "confidence": confidence,
                    "recall_at_5": rec_at_5,
                    "mrr": mrr,
                    "groundedness": ground,
                    "answer_snippet": answer_snippet,
                    "error": None,
                }
            )
            if verbose:
                print(f"    recall@5={rec_at_5:.2f} mrr={mrr:.2f} ground={ground:.2f}")

        except Exception as exc:
            results.append(
                {
                    "id": qid,
                    "question": question,
                    "category": category,
                    "expected_locators": expected_locs,
                    "retrieved_locators": [],
                    "confidence": "error",
                    "recall_at_5": 0.0,
                    "mrr": 0.0,
                    "groundedness": 0.0,
                    "answer_snippet": "",
                    "error": str(exc),
                }
            )
            if verbose:
                print(f"    ERROR: {exc}")

        # Respect Voyage free-tier rate limit (3 RPM) between questions
        if inter_question_sleep > 0 and item is not qa_items[-1]:
            if verbose:
                print(f"    sleeping {inter_question_sleep}s (Voyage rate limit)...")
            time.sleep(inter_question_sleep)

    n = len(results)
    answerable = [r for r in results if r["category"] != "unanswerable"]
    unanswerable = [r for r in results if r["category"] == "unanswerable"]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": n,
        "answerable_n": len(answerable),
        "unanswerable_n": len(unanswerable),
        "recall_at_5": sum(r["recall_at_5"] for r in answerable) / max(len(answerable), 1),
        "mrr": sum(r["mrr"] for r in answerable) / max(len(answerable), 1),
        "groundedness": sum(r["groundedness"] for r in results) / max(n, 1),
        "correct_refusals": sum(r["recall_at_5"] for r in unanswerable),
        "results": results,
    }


def write_report(summary: dict, report_path: Path) -> None:
    lines = [
        "# Eval Results",
        "",
        f"**Timestamp**: {summary['timestamp']}",
        f"**Total questions**: {summary['total']}",
        f"**Answerable**: {summary['answerable_n']} | **Unanswerable**: {summary['unanswerable_n']}",
        "",
        "## Summary Metrics",
        "",
        f"| Metric | Score |",
        f"|--------|-------|",
        f"| Recall@5 | {summary['recall_at_5']:.3f} |",
        f"| MRR | {summary['mrr']:.3f} |",
        f"| Groundedness | {summary['groundedness']:.3f} |",
        f"| Correct refusals | {int(summary['correct_refusals'])}/{summary['unanswerable_n']} |",
        "",
        "## Per-Question Results",
        "",
        "| ID | Category | Recall@5 | MRR | Grounded | Confidence | Answer snippet |",
        "|----|----------|----------|-----|----------|------------|----------------|",
    ]

    for r in summary["results"]:
        err = f" ⚠ {r['error'][:40]}" if r["error"] else ""
        lines.append(
            f"| {r['id']} | {r['category']} | {r['recall_at_5']:.2f} | "
            f"{r['mrr']:.2f} | {r['groundedness']:.2f} | {r['confidence']} | "
            f"{r['answer_snippet'][:60]}{err} |"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run retrieval + synthesis evaluation.")
    parser.add_argument("--report", type=Path, default=Path("backend/eval/results.md"))
    parser.add_argument("--qa-set", type=Path, default=QA_SET_PATH)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--cross-doc-check", action="store_true",
                        help="Flag to indicate cross-doc eval mode (currently informational).")
    parser.add_argument("--sleep", type=float, default=VOYAGE_INTER_QUESTION_SLEEP,
                        help="Seconds to sleep between questions (default 25 for Voyage 3 RPM free tier).")
    parser.add_argument("--limit", type=int, help="Run only the first N questions from the QA set.")
    parser.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                        help="Retries for transient 429/overload errors.")
    parser.add_argument("--retry-sleep", type=float, default=DEFAULT_RETRY_SLEEP,
                        help="Seconds to wait before retrying transient errors.")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="Skip Kimi synthesis and groundedness; report retrieval Recall@5/MRR only.")
    args = parser.parse_args()

    qa_items = load_qa_set(args.qa_set)
    if args.limit is not None:
        qa_items = qa_items[:args.limit]
    print(f"[*] running eval on {len(qa_items)} questions (sleep={args.sleep}s between)...")

    summary = run_evaluation(
        qa_items,
        verbose=args.verbose,
        inter_question_sleep=args.sleep,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
        retrieval_only=args.retrieval_only,
    )
    write_report(summary, args.report)

    print(f"[ok] Recall@5={summary['recall_at_5']:.3f} | MRR={summary['mrr']:.3f} | "
          f"Groundedness={summary['groundedness']:.3f}")
    print(f"     report -> {args.report}")

    if summary["recall_at_5"] < 0.85:
        print(f"[WARN] Recall@5 {summary['recall_at_5']:.3f} < target 0.85", file=sys.stderr)
    if summary["groundedness"] < 0.90:
        print(f"[WARN] Groundedness {summary['groundedness']:.3f} < target 0.90", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
