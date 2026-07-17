"""Chonkie-based chunking for the Bionema retrieval POC (F02).

Takes ParsedUnit objects from F01 and produces IndexChunk objects ready for
embedding. Each ParsedUnit (paragraph) is a hard boundary — Chonkie only
splits within a unit when it exceeds the token limit.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from chonkie import RecursiveChunker

from backend.ingest.parse import DEFAULT_OUTPUT_DIR, ParsedUnit, parsed_document_from_dict

DEFAULT_CHUNK_SIZE = 300  # tokens (gpt2 tokenizer)
MIN_CHUNK_TOKENS = 15


@dataclass(frozen=True)
class IndexChunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    locator: str
    locator_type: str
    section: str
    text: str
    token_count: int


def _chunk_id(doc_id: str, locator: str, index: int) -> str:
    raw = f"{doc_id}:{locator}:{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def chunk_units(
    units: Iterable[ParsedUnit],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[IndexChunk]:
    chunker = RecursiveChunker(tokenizer="gpt2", chunk_size=chunk_size)
    results: list[IndexChunk] = []
    for unit in units:
        sub_chunks = chunker.chunk(unit.text)
        for i, sub in enumerate(sub_chunks):
            text = sub.text.strip()
            if not text or sub.token_count < MIN_CHUNK_TOKENS:
                continue
            results.append(
                IndexChunk(
                    chunk_id=_chunk_id(unit.doc_id, unit.locator, i),
                    doc_id=unit.doc_id,
                    doc_title=unit.doc_title,
                    locator=unit.locator,
                    locator_type=unit.locator_type,
                    section=unit.section,
                    text=text,
                    token_count=sub.token_count,
                )
            )
    return results


def chunks_from_parsed_file(
    parsed_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[IndexChunk]:
    with parsed_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    parsed = parsed_document_from_dict(payload)
    return chunk_units(parsed.units, chunk_size=chunk_size)


def chunks_from_all_parsed(
    parsed_dir: Path = DEFAULT_OUTPUT_DIR,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> list[IndexChunk]:
    all_chunks: list[IndexChunk] = []
    for path in sorted(parsed_dir.glob("*.parsed.json")):
        all_chunks.extend(chunks_from_parsed_file(path, chunk_size=chunk_size))
    return all_chunks


def main() -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Chunk parsed documents into IndexChunks.")
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    chunks = chunks_from_all_parsed(args.parsed_dir, chunk_size=args.chunk_size)
    if not chunks:
        print("[FAIL] no chunks produced", file=sys.stderr)
        return 1

    token_counts = [c.token_count for c in chunks]
    print(f"[ok] {len(chunks)} chunks total")
    print(f"     token range: {min(token_counts)}–{max(token_counts)}")
    print(f"     avg tokens: {sum(token_counts) / len(token_counts):.0f}")
    docs = {c.doc_id for c in chunks}
    for doc_id in sorted(docs):
        doc_chunks = [c for c in chunks if c.doc_id == doc_id]
        print(f"     {doc_id}: {len(doc_chunks)} chunks")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main())
