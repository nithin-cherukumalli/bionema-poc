"""Embedding + Qdrant upsert pipeline for the Bionema retrieval POC (F03).

Embeds IndexChunk objects with Voyage AI (dense) and FastEmbed BM25 (sparse),
then upserts them into a Qdrant hybrid collection.
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator

import voyageai
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.config import Settings, get_settings
from backend.ingest.chunk import IndexChunk, chunks_from_all_parsed
from backend.ingest.parse import DEFAULT_OUTPUT_DIR

COLLECTION_NAME = "bionema_poc_v1"
DENSE_DIM = 1024
VOYAGE_EMBED_MODEL = "voyage-3-large"
VOYAGE_BATCH_SIZE = 32
VOYAGE_BATCH_TOKEN_BUDGET = 3000
VOYAGE_REQUEST_INTERVAL_SECONDS = 22
UPSERT_BATCH_SIZE = 64
SPARSE_MODEL_NAME = "Qdrant/bm25"


def build_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )


def recreate_collection(client: QdrantClient, collection_name: str = COLLECTION_NAME) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        },
    )


def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _voyage_batches(chunks: list[IndexChunk]) -> Iterator[list[IndexChunk]]:
    batch: list[IndexChunk] = []
    token_total = 0
    for chunk in chunks:
        would_exceed_size = len(batch) >= VOYAGE_BATCH_SIZE
        would_exceed_tokens = batch and token_total + chunk.token_count > VOYAGE_BATCH_TOKEN_BUDGET
        if would_exceed_size or would_exceed_tokens:
            yield batch
            batch = []
            token_total = 0
        batch.append(chunk)
        token_total += chunk.token_count
    if batch:
        yield batch


def embed_dense(chunks: list[IndexChunk], settings: Settings) -> list[list[float]]:
    client = voyageai.Client(api_key=settings.voyage_api_key)
    all_embeddings: list[list[float]] = []
    batches = list(_voyage_batches(chunks))
    for index, batch in enumerate(batches, start=1):
        if index > 1:
            time.sleep(VOYAGE_REQUEST_INTERVAL_SECONDS)
        token_total = sum(chunk.token_count for chunk in batch)
        print(
            f"  dense batch {index}/{len(batches)}: "
            f"{len(batch)} chunks, ~{token_total} tokens"
        )
        result = client.embed([chunk.text for chunk in batch], model=VOYAGE_EMBED_MODEL)
        all_embeddings.extend(result.embeddings)
    return all_embeddings


def embed_sparse(texts: list[str]) -> list[models.SparseVector]:
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    results: list[models.SparseVector] = []
    for embedding in sparse_model.embed(texts):
        results.append(
            models.SparseVector(
                indices=embedding.indices.tolist(),
                values=embedding.values.tolist(),
            )
        )
    return results


def upsert_chunks(
    chunks: list[IndexChunk],
    *,
    settings: Settings | None = None,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False,
) -> int:
    resolved = settings or get_settings()
    client = build_qdrant_client(resolved)

    if recreate:
        recreate_collection(client, collection_name)
    else:
        ensure_collection(client, collection_name)

    print(f"  embedding {len(chunks)} chunks with {VOYAGE_EMBED_MODEL} (dense)...")
    dense_vecs = embed_dense(chunks, resolved)

    print(f"  computing BM25 sparse vectors ({SPARSE_MODEL_NAME})...")
    texts = [c.text for c in chunks]
    sparse_vecs = embed_sparse(texts)

    points: list[models.PointStruct] = []
    for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                vector={"dense": dense, "sparse": sparse},
                payload={
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "locator": chunk.locator,
                    "locator_type": chunk.locator_type,
                    "section": chunk.section,
                    "text": chunk.text,
                    "token_count": chunk.token_count,
                },
            )
        )

    total_upserted = 0
    for batch in _batched(points, UPSERT_BATCH_SIZE):
        client.upsert(collection_name=collection_name, points=batch)
        total_upserted += len(batch)
        print(f"  upserted {total_upserted}/{len(points)}")

    return total_upserted


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed and upsert chunks into Qdrant.")
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--recreate", action="store_true",
                        help="Drop and recreate the Qdrant collection before upserting.")
    args = parser.parse_args()

    settings = get_settings()
    chunks = chunks_from_all_parsed(args.parsed_dir)
    if not chunks:
        print("[FAIL] no chunks to upsert", file=sys.stderr)
        return 1

    print(f"[*] upserting {len(chunks)} chunks -> {args.collection}")
    count = upsert_chunks(chunks, settings=settings, collection_name=args.collection, recreate=args.recreate)
    print(f"[ok] upserted {count} points to {args.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
