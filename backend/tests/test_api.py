"""Tests for FastAPI endpoints (F07a)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.retrieval.hybrid_query import RetrievedChunk
from backend.retrieval.rerank import RankedChunk
from backend.synthesis.synthesize import Citation, SynthesisResult


def _make_ranked_chunk(locator: str = "[0072]") -> RankedChunk:
    return RankedChunk(
        chunk_id="abc",
        doc_id="WO2020053603A1",
        doc_title="Insect-Pathogenic Fungus Patent",
        locator=locator,
        locator_type="explicit_bracketed",
        section="DETAILED DESCRIPTION",
        text="BNL 102 demonstrated 90% mortality against vine weevil larvae.",
        token_count=20,
        rerank_score=0.92,
    )


@pytest.fixture()
def client():
    with (
        patch("backend.main.hybrid_search") as mock_search,
        patch("backend.main.rerank") as mock_rerank,
        patch("backend.main.synthesize") as mock_synthesize,
        patch("backend.main.build_qdrant_client"),
        patch("backend.main.voyageai.Client"),
        patch("backend.main.OpenAI"),
    ):
        mock_search.return_value = [
            RetrievedChunk(
                chunk_id="abc",
                doc_id="WO2020053603A1",
                doc_title="Test",
                locator="[0072]",
                locator_type="explicit_bracketed",
                section="DETAILED DESCRIPTION",
                text="sample text",
                token_count=10,
                score=0.8,
            )
        ]
        mock_rerank.return_value = [_make_ranked_chunk()]
        mock_synthesize.return_value = SynthesisResult(
            answer="BNL 102 showed 90% mortality [0072].",
            confidence="high",
            citations=[
                Citation(
                    paragraph_id="[0072]",
                    section="DETAILED DESCRIPTION",
                    quote="90% mortality",
                    score=0.92,
                )
            ],
        )

        from backend import main as main_module

        main_module._query_cache.clear()
        main_module._ranked_cache.clear()
        app = main_module.app
        with TestClient(app) as c:
            yield c, mock_search, mock_rerank, mock_synthesize


class TestQueryEndpoint:
    def test_query_returns_200(self, client):
        c, *_ = client
        response = c.post("/query", json={"question": "What is the efficacy of BNL 102?"})
        assert response.status_code == 200

    def test_query_response_schema(self, client):
        c, *_ = client
        response = c.post("/query", json={"question": "What is BNL 102?"})
        data = response.json()
        assert "answer" in data
        assert "confidence" in data
        assert "citations" in data
        assert data["confidence"] in {"high", "partial", "not_found"}

    def test_query_empty_question_returns_400(self, client):
        c, *_ = client
        response = c.post("/query", json={"question": ""})
        assert response.status_code == 400

    def test_query_calls_pipeline_in_order(self, client):
        c, mock_search, mock_rerank, mock_synthesize = client
        c.post("/query", json={"question": "What fungi are used?"})
        mock_search.assert_called_once()
        mock_rerank.assert_called_once()
        mock_synthesize.assert_called_once()

    def test_query_reuses_cached_response_for_same_question(self, client):
        c, mock_search, mock_rerank, mock_synthesize = client

        first = c.post("/query", json={"question": "What is BNL 102?"})
        second = c.post("/query", json={"question": "  what   is bnl 102?  "})

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json() == second.json()
        mock_search.assert_called_once()
        mock_rerank.assert_called_once()
        mock_synthesize.assert_called_once()

    def test_query_replaces_partial_response_without_citations(self, client):
        c, mock_search, mock_rerank, mock_synthesize = client
        mock_synthesize.return_value = SynthesisResult(
            answer="The provided documents do not contain sufficient information to answer this question.",
            confidence="partial",
            citations=[],
        )

        first = c.post("/query", json={"question": "What is the strain name?"})
        second = c.post("/query", json={"question": "what is the strain name?"})

        assert first.json() == second.json()
        assert first.json()["citations"][0]["paragraph_id"] == "[0072]"
        assert mock_search.call_count == 1
        assert mock_rerank.call_count == 1
        assert mock_synthesize.call_count == 1

    def test_query_citation_fields(self, client):
        c, *_ = client
        response = c.post("/query", json={"question": "Efficacy test"})
        data = response.json()
        assert len(data["citations"]) == 1
        cit = data["citations"][0]
        assert "paragraph_id" in cit
        assert "section" in cit
        assert "quote" in cit
        assert "score" in cit

    def test_query_returns_retrieved_evidence_when_synthesis_fails(self, client):
        c, _, _, mock_synthesize = client
        mock_synthesize.side_effect = RuntimeError("engine overloaded")

        response = c.post("/query", json={"question": "What is BNL 102?"})

        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] == "partial"
        assert "Kimi synthesis is temporarily unavailable" in data["answer"]
        assert data["citations"][0]["paragraph_id"] == "[0072]"
        assert "90% mortality" in data["citations"][0]["quote"]

    def test_query_evidence_returns_sources_without_synthesis(self, client):
        c, mock_search, mock_rerank, mock_synthesize = client

        response = c.post("/query/evidence", json={"question": "What is BNL 102?"})

        assert response.status_code == 200
        data = response.json()
        assert data["confidence"] == "partial"
        assert "Retrieved evidence" in data["answer"]
        assert data["citations"][0]["paragraph_id"] == "[0072]"
        mock_search.assert_called_once()
        mock_rerank.assert_called_once()
        mock_synthesize.assert_not_called()

    def test_query_reuses_ranked_cache_after_evidence_preview(self, client):
        c, mock_search, mock_rerank, mock_synthesize = client

        c.post("/query/evidence", json={"question": "What is BNL 102?"})
        final = c.post("/query", json={"question": "what is bnl 102?"})

        assert final.status_code == 200
        mock_search.assert_called_once()
        mock_rerank.assert_called_once()
        mock_synthesize.assert_called_once()


class TestHealthEndpoint:
    def test_health_returns_200(self):
        with (
            patch("backend.main.build_qdrant_client") as mock_qdrant,
            patch("backend.main.voyageai.Client") as mock_voyage,
            patch("backend.main.OpenAI") as mock_kimi,
        ):
            mock_qdrant.return_value.get_collection.return_value = MagicMock()
            mock_voyage.return_value.embed.return_value = MagicMock()
            mock_kimi.return_value.chat.completions.create.return_value = MagicMock()

            from backend.main import app
            with TestClient(app) as c:
                response = c.get("/health")

        assert response.status_code == 200

    def test_health_response_schema(self):
        with (
            patch("backend.main.build_qdrant_client") as mock_qdrant,
            patch("backend.main.voyageai.Client") as mock_voyage,
            patch("backend.main.OpenAI") as mock_kimi,
        ):
            mock_qdrant.return_value.get_collection.return_value = MagicMock()
            mock_voyage.return_value.embed.return_value = MagicMock()
            mock_kimi.return_value.chat.completions.create.return_value = MagicMock()

            from backend.main import app
            with TestClient(app) as c:
                response = c.get("/health")

        data = response.json()
        assert "status" in data
        assert "qdrant" in data
        assert "voyage" in data
        assert "kimi" in data
