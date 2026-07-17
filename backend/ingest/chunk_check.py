"""Verification command for F02 chunking (chunk_check.py).

Runs the chunker against all parsed documents and writes a report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.ingest.chunk import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    MIN_CHUNK_TOKENS,
    IndexChunk,
    chunks_from_all_parsed,
)
from backend.ingest.parse import DEFAULT_OUTPUT_DIR  # noqa: E402

MAX_TARGET_TOKENS = 300
MIN_TARGET_TOKENS = 150


def write_report(chunks: list[IndexChunk], report_path: Path) -> None:
    token_counts = [c.token_count for c in chunks]
    in_range = [c for c in chunks if MIN_TARGET_TOKENS <= c.token_count <= MAX_TARGET_TOKENS]
    docs: dict[str, list[IndexChunk]] = {}
    for c in chunks:
        docs.setdefault(c.doc_id, []).append(c)

    lines = [
        "# Chunk Report",
        "",
        f"Total chunks: {len(chunks)}",
        f"Token range: {min(token_counts)}–{max(token_counts)}",
        f"Avg tokens: {sum(token_counts) / len(token_counts):.0f}",
        f"In target range ({MIN_TARGET_TOKENS}–{MAX_TARGET_TOKENS}): "
        f"{len(in_range)} ({100 * len(in_range) / len(chunks):.1f}%)",
        "",
        "## Per-document breakdown",
        "",
    ]
    for doc_id, doc_chunks in sorted(docs.items()):
        dt = [c.token_count for c in doc_chunks]
        lines.append(f"### {doc_id}")
        lines.append(f"- chunks: {len(doc_chunks)}")
        lines.append(f"- token range: {min(dt)}–{max(dt)}")
        lines.append(f"- avg tokens: {sum(dt) / len(dt):.0f}")
        lines.append("")

    lines.append("## Sample chunks (first 3)")
    lines.append("")
    for chunk in chunks[:3]:
        lines.append(f"**{chunk.doc_id} / {chunk.locator}** [{chunk.token_count} tokens]")
        lines.append(f"> {chunk.text[:200]}...")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify F02 chunking output.")
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=Path("backend/eval/chunk_report.md"))
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--min-chunks", type=int, default=40,
                        help="Minimum number of chunks to consider the run valid.")
    args = parser.parse_args()

    chunks = chunks_from_all_parsed(args.parsed_dir, chunk_size=args.chunk_size)

    errors: list[str] = []
    if not chunks:
        errors.append("no chunks produced")
    elif len(chunks) < args.min_chunks:
        errors.append(f"expected at least {args.min_chunks} chunks, got {len(chunks)}")

    oversized = [c for c in chunks if c.token_count > MAX_TARGET_TOKENS]
    if oversized:
        pct = 100 * len(oversized) / len(chunks)
        if pct > 5:
            errors.append(
                f"{len(oversized)} chunks ({pct:.1f}%) exceed {MAX_TARGET_TOKENS} tokens"
            )

    if not errors and chunks:
        write_report(chunks, args.report)
        token_counts = [c.token_count for c in chunks]
        print(
            f"[ok] {len(chunks)} chunks, "
            f"token range {min(token_counts)}–{max(token_counts)}, "
            f"report -> {args.report}"
        )
        return 0
    elif chunks:
        write_report(chunks, args.report)

    for err in errors:
        print(f"[FAIL] {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
