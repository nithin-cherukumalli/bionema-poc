"""Tests for hybrid retrieval and reranking (F04)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.retrieval.hybrid_query import RetrievedChunk, hybrid_search
from backend.retrieval.rerank import RankedChunk, MIN_RERANK_SCORE, rerank


def _make_retrieved(locator: str = "[0072]", text: str = "test text", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="abc123",
        doc_id="WO2020053603A1",
        doc_title="Test Patent",
        locator=locator,
        locator_type="explicit_bracketed",
        section="DETAILED DESCRIPTION",
        text=text,
        token_count=50,
        score=score,
    )


class TestRerank:
    def test_empty_candidates_returns_empty(self):
        result = rerank("any query", [])
        assert result == []

    def test_rerank_filters_below_threshold(self):
        candidates = [_make_retrieved("[0001]"), _make_retrieved("[0002]")]

        mock_result_item_low = MagicMock()
        mock_result_item_low.index = 0
        mock_result_item_low.relevance_score = 0.1  # below threshold

        mock_result_item_high = MagicMock()
        mock_result_item_high.index = 1
        mock_result_item_high.relevance_score = 0.9

        mock_rerank_result = MagicMock()
        mock_rerank_result.results = [mock_result_item_low, mock_result_item_high]

        with patch("backend.retrieval.rerank.voyageai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank.return_value = mock_rerank_result
            mock_client_cls.return_value = mock_client

            result = rerank("test query", candidates)

        assert len(result) == 1
        assert result[0].locator == "[0002]"
        assert result[0].rerank_score == 0.9

    def test_rerank_returns_ranked_chunks(self):
        candidates = [_make_retrieved("[0010]", "fungi text"), _make_retrieved("[0020]", "other text")]

        item1 = MagicMock()
        item1.index = 0
        item1.relevance_score = 0.95

        item2 = MagicMock()
        item2.index = 1
        item2.relevance_score = 0.75

        mock_result = MagicMock()
        mock_result.results = [item1, item2]

        with patch("backend.retrieval.rerank.voyageai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client.rerank.return_value = mock_result
            mock_client_cls.return_value = mock_client

            result = rerank("fungi efficacy", candidates)

        assert len(result) == 2
        assert isinstance(result[0], RankedChunk)
        assert result[0].rerank_score == 0.95
        assert result[0].locator == "[0010]"


class TestHybridSearch:
    def test_hybrid_search_returns_retrieved_chunks(self):
        mock_point = MagicMock()
        mock_point.score = 0.85
        mock_point.payload = {
            "chunk_id": "abc",
            "doc_id": "WO2020053603A1",
            "doc_title": "Test Patent",
            "locator": "[0072]",
            "locator_type": "explicit_bracketed",
            "section": "DETAILED DESCRIPTION",
            "text": "BNL 102 showed 90% mortality against WFT.",
            "token_count": 30,
        }

        mock_query_result = MagicMock()
        mock_query_result.points = [mock_point]

        mock_dense_embedding = MagicMock()
        mock_dense_embedding.embeddings = [[0.1] * 1024]

        mock_sparse_embedding = MagicMock()
        mock_sparse_embedding.indices = np.array([1, 2, 3])
        mock_sparse_embedding.values = np.array([0.5, 0.3, 0.2])

        with (
            patch("backend.retrieval.hybrid_query.voyageai.Client") as mock_voyage,
            patch("backend.retrieval.hybrid_query._get_sparse_model") as mock_sparse_fn,
            patch("backend.retrieval.hybrid_query.build_qdrant_client") as mock_qdrant,
        ):
            mock_voyage.return_value.embed.return_value = mock_dense_embedding
            mock_sparse_fn.return_value.embed.return_value = iter([mock_sparse_embedding])
            mock_qdrant.return_value.query_points.return_value = mock_query_result

            results = hybrid_search("WFT mortality rate")

        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].locator == "[0072]"
        assert results[0].score == 0.85
