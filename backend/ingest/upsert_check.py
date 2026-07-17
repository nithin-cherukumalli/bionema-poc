"""Verification command for F03 — checks Qdrant collection after upsert."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.config import get_settings  # noqa: E402
from backend.ingest.embed_upsert import COLLECTION_NAME, build_qdrant_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Qdrant collection after F03 upsert.")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--expect-min-points", type=int, default=40)
    args = parser.parse_args()

    settings = get_settings()
    client = build_qdrant_client(settings)

    if not client.collection_exists(args.collection):
        print(f"[FAIL] collection {args.collection!r} does not exist", file=sys.stderr)
        return 1

    info = client.get_collection(args.collection)
    count = info.points_count or 0

    if count < args.expect_min_points:
        print(
            f"[FAIL] {args.collection}: expected >= {args.expect_min_points} points, "
            f"found {count}",
            file=sys.stderr,
        )
        return 1

    vectors_config = info.config.params.vectors
    has_dense = "dense" in (vectors_config or {})
    sparse_config = info.config.params.sparse_vectors
    has_sparse = sparse_config is not None and "sparse" in sparse_config

    if not has_dense:
        print(f"[FAIL] collection missing 'dense' vector config", file=sys.stderr)
        return 1
    if not has_sparse:
        print(f"[FAIL] collection missing 'sparse' vector config", file=sys.stderr)
        return 1

    # Sample one point to verify payload schema
    results = client.scroll(
        collection_name=args.collection,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if results[0]:
        payload = results[0][0].payload or {}
        required_fields = {"chunk_id", "doc_id", "locator", "section", "text", "token_count"}
        missing = required_fields - payload.keys()
        if missing:
            print(f"[FAIL] sample point missing payload fields: {missing}", file=sys.stderr)
            return 1

    print(
        f"[ok] {args.collection}: {count} points, "
        f"dense={has_dense}, sparse={has_sparse}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
