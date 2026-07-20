"""Tests for citation-constrained synthesis (F05)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.retrieval.rerank import RankedChunk
from backend.synthesis.prompt import build_context_block, build_user_message
from backend.synthesis.synthesize import (
    NOT_FOUND_ANSWER,
    Citation,
    SynthesisResult,
    _extract_json,
    synthesize,
)


def _make_ranked(locator: str = "[0072]", text: str = "BNL 102 showed 90% mortality.", score: float = 0.9) -> RankedChunk:
    return RankedChunk(
        chunk_id="abc",
        doc_id="WO2020053603A1",
        doc_title="Insect-Pathogenic Fungus Patent",
        locator=locator,
        locator_type="explicit_bracketed",
        section="DETAILED DESCRIPTION",
        text=text,
        token_count=20,
        rerank_score=score,
    )


class TestPromptBuilding:
    def test_context_block_contains_locator(self):
        chunks = [_make_ranked("[0072]", "BNL 102 showed 90% mortality.")]
        block = build_context_block(chunks)
        assert "[0072]" in block
        assert "BNL 102 showed 90% mortality." in block

    def test_context_block_multiple_chunks(self):
        chunks = [
            _make_ranked("[0072]", "Text one."),
            _make_ranked("[0080]", "Text two."),
        ]
        block = build_context_block(chunks)
        assert "Excerpt 1" in block
        assert "Excerpt 2" in block

    def test_user_message_contains_question(self):
        chunks = [_make_ranked()]
        msg = build_user_message("What is the efficacy of BNL 102?", chunks)
        assert "What is the efficacy of BNL 102?" in msg
        assert "[0072]" in msg


class TestExtractJson:
    def test_plain_json(self):
        raw = '{"answer": "test", "confidence": "high", "citations": []}'
        result = _extract_json(raw)
        assert result["answer"] == "test"

    def test_json_in_code_fence(self):
        raw = '```json\n{"answer": "test", "confidence": "high", "citations": []}\n```'
        result = _extract_json(raw)
        assert result["answer"] == "test"

    def test_json_in_plain_fence(self):
        raw = '```\n{"answer": "ok", "confidence": "partial", "citations": []}\n```'
        result = _extract_json(raw)
        assert result["confidence"] == "partial"

    def test_json_in_fence_with_bold_locators(self):
        raw = '''```json
{
  "answer": "BNL 101 and BNL 102 are both deposited strains [0010]**[0011]**.",
  "confidence": "partial",
  "citations": [
    {
      "paragraph_id": "**[0010]**",
      "section": "SUMMARY",
      "quote": "BNL 101 deposited... BNL 102 deposited..."
    }
  ]
}
```'''
        result = _extract_json(raw)
        assert result["answer"] == "BNL 101 and BNL 102 are both deposited strains [0010][0011]."
        assert result["citations"][0]["paragraph_id"] == "[0010]"


class TestSynthesize:
    def test_empty_chunks_returns_not_found(self):
        result = synthesize("any question", [])
        assert result.confidence == "not_found"
        assert result.answer == NOT_FOUND_ANSWER
        assert result.citations == []

    def test_successful_synthesis(self):
        chunks = [_make_ranked("[0072]", "BNL 102 showed 90% mortality against WFT.")]

        mock_response_content = "BNL 102 showed 90% mortality [0072]."
        mock_choice = MagicMock()
        mock_choice.message.content = mock_response_content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("backend.synthesis.synthesize.create_kimi_client") as mock_kimi:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_kimi.return_value = mock_client

            result = synthesize("What is the efficacy of BNL 102?", chunks)

        assert result.confidence == "high"
        assert "[0072]" in result.answer
        assert len(result.citations) == 1
        assert result.citations[0].paragraph_id == "[0072]"
        assert result.citations[0].score == 0.9
        assert "temperature" not in mock_client.chat.completions.create.call_args.kwargs
        assert mock_client.chat.completions.create.call_args.kwargs["extra_body"] == {
            "thinking": {"type": "disabled"}
        }

    def test_synthesis_tolerates_legacy_json_response(self):
        chunks = [_make_ranked("[0072]", "BNL 102 showed 90% mortality against WFT.")]

        mock_response_content = (
            '{"answer": "BNL 102 showed 90% mortality [0072].", '
            '"confidence": "high", '
            '"citations": [{"paragraph_id": "[0072]", "section": "DETAILED DESCRIPTION", '
            '"quote": "BNL 102 showed 90% mortality"}]}'
        )
        mock_choice = MagicMock()
        mock_choice.message.content = mock_response_content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("backend.synthesis.synthesize.create_kimi_client") as mock_kimi:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_kimi.return_value = mock_client

            result = synthesize("What is the efficacy of BNL 102?", chunks)

        assert result.confidence == "high"
        assert result.answer == "BNL 102 showed 90% mortality [0072]."
        assert result.citations[0].paragraph_id == "[0072]"

    def test_synthesis_normalizes_bold_locator_markers(self):
        chunks = [_make_ranked("[0010]", "BNL 101 and BNL 102 deposited strains.")]

        mock_response_content = (
            '```json\n{"answer": "Both strains are deposited **[0010]**.", '
            '"confidence": "partial", '
            '"citations": [{"paragraph_id": "**[0010]**", "section": "SUMMARY", '
            '"quote": "Both strains are deposited"}]}\n```'
        )
        mock_choice = MagicMock()
        mock_choice.message.content = mock_response_content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("backend.synthesis.synthesize.create_kimi_client") as mock_kimi:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_kimi.return_value = mock_client

            result = synthesize("Compare strains", chunks)

        assert result.answer == "Both strains are deposited [0010]."
        assert result.citations[0].paragraph_id == "[0010]"

    def test_uncited_response_is_rejected(self):
        chunks = [_make_ranked()]

        mock_choice = MagicMock()
        mock_choice.message.content = "I cannot parse this as JSON."
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("backend.synthesis.synthesize.create_kimi_client") as mock_kimi:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_kimi.return_value = mock_client

            result = synthesize("test question", chunks)

        assert result.confidence == "not_found"
        assert result.answer == NOT_FOUND_ANSWER
        assert result.citations == []

    def test_not_found_confidence(self):
        chunks = [_make_ranked()]

        mock_response_content = (
            '{"answer": "' + NOT_FOUND_ANSWER + '", '
            '"confidence": "not_found", "citations": []}'
        )
        mock_choice = MagicMock()
        mock_choice.message.content = mock_response_content
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        with patch("backend.synthesis.synthesize.create_kimi_client") as mock_kimi:
            mock_client = MagicMock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_kimi.return_value = mock_client

            result = synthesize("an unanswerable question", chunks)

        assert result.confidence == "not_found"
        assert result.citations == []
