"""Tests for tnmi.local_llm.GemmaAnalyzer.

These tests do NOT require Ollama to be running. We mock the Ollama client so
the analyser logic — JSON-mode parsing, code-fence stripping, backfill of
missing/null fields, gate enforcement — is verifiable in CI.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
    SourceType,
    Stance,
)
from tnmi.local_llm import (
    GemmaAnalyzer,
    GemmaAnalyzerUnavailable,
    _backfill_defaults,
    _strip_code_fences,
)


def _make_item(
    *,
    title: str = "தமிழக அரசு புதிய நலத்திட்டத்தை அறிவித்துள்ளது",
    body: str = (
        "சென்னை: தமிழக அரசு இன்று புதிய நலத்திட்டத்தை அறிவித்தது. "
        "முதலமைச்சர் ஸ்டாலின் இத்திட்டத்தை வரவேற்றுள்ளார். "
        "இத்திட்டம் தமிழ்நாட்டின் அனைத்து 38 மாவட்டங்களிலும் "
        "செயல்படுத்தப்படும் என்று அரசு தெரிவித்துள்ளது."
    ),
) -> NormalizedItem:
    return NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Test",
        source_url="https://example.com/a",
        language="ta",
        title=title,
        raw_text_original=body,
        clean_text_original=body,
    )


def test_strip_code_fences_removes_json_fence():
    raw = "```json\n{\"stance\": \"positive\"}\n```"
    assert _strip_code_fences(raw) == '{"stance": "positive"}'


def test_strip_code_fences_removes_generic_fence():
    raw = "```\n{\"a\": 1}\n```"
    assert _strip_code_fences(raw) == '{"a": 1}'


def test_strip_code_fences_passes_through_clean_json():
    raw = '{"stance": "negative"}'
    assert _strip_code_fences(raw) == raw


def test_backfill_coerces_none_strings_to_empty():
    item = _make_item()
    payload = {
        "government_relevance": "high",
        "stance_toward_government": "positive",
        "sentiment": "positive",
        "people_impact": None,        # Gemma's "I don't know" pattern
        "recommended_step": "None",   # Gemma's literal-string pattern
        "root_cause": "N/A",
    }
    _backfill_defaults(payload, item)
    assert payload["people_impact"] == ""
    assert payload["recommended_step"] == ""
    assert payload["root_cause"] == ""


def test_backfill_coerces_null_lists_to_empty_arrays():
    item = _make_item()
    payload = {
        "positive_points": None,
        "negative_points": None,
        "evidence_quotes_original": None,
    }
    _backfill_defaults(payload, item)
    assert payload["positive_points"] == []
    assert payload["negative_points"] == []
    assert payload["evidence_quotes_original"] == []


def test_backfill_coerces_confidence_from_string():
    item = _make_item()
    payload = {"confidence": "0.85"}
    _backfill_defaults(payload, item)
    assert payload["confidence"] == 0.85


def test_backfill_preserves_existing_values():
    item = _make_item()
    payload = {
        "stance_toward_government": "negative",
        "summary_original": "Actual summary text in Tamil",
        "party_action": "TVK MLAs protested in Assembly",
    }
    _backfill_defaults(payload, item)
    assert payload["stance_toward_government"] == "negative"
    assert payload["summary_original"] == "Actual summary text in Tamil"
    assert payload["party_action"] == "TVK MLAs protested in Assembly"


def _make_gemma_with_fake_client(client: MagicMock) -> GemmaAnalyzer:
    """Construct a GemmaAnalyzer with a pre-baked fake Ollama client. Skips
    the lazy import + daemon check so we can unit-test analyse() logic."""
    analyser = GemmaAnalyzer.__new__(GemmaAnalyzer)
    analyser._client = client
    analyser._model = "gemma2:2b"
    analyser.model_name = "ollama/gemma2:2b"
    analyser._daemon_checked = True
    return analyser


def test_analyze_listing_page_gated_before_llm_call():
    """RSS shells must never reach the LLM — saves Gemma latency + avoids
    hallucinated stances on noise."""
    client = MagicMock()
    g = _make_gemma_with_fake_client(client)
    short_item = _make_item(title="news", body="x")  # < 160 chars
    result = g.analyze(short_item)
    assert result.government_relevance == GovernmentRelevance.NONE
    client.generate.assert_not_called()


def test_analyze_out_of_scope_gated_before_llm_call():
    """Articles with no TN keyword anywhere must be gated before Gemma is
    invoked."""
    client = MagicMock()
    g = _make_gemma_with_fake_client(client)
    foreign = _make_item(
        title="Mumbai stock market closes higher",
        body=(
            "BSE Sensex rose 200 points after RBI announced a new bond auction "
            "this morning. Markets in Mumbai responded positively to the latest "
            "inflation forecast released earlier today by the central bank."
        ),
    )
    result = g.analyze(foreign)
    assert result.government_relevance == GovernmentRelevance.NONE
    client.generate.assert_not_called()


def test_analyze_parses_well_formed_json():
    """The happy path: TN-relevant article + valid JSON response → AIAnalysis."""
    client = MagicMock()
    client.generate.return_value = SimpleNamespace(
        response=(
            '{"government_relevance": "high",'
            ' "stance_toward_government": "positive",'
            ' "sentiment": "positive",'
            ' "target": "Tamil Nadu Government",'
            ' "department": "social welfare",'
            ' "district": "statewide",'
            ' "scheme": "New welfare scheme",'
            ' "topic": "TN welfare scheme launch",'
            ' "issue_category": "welfare",'
            ' "severity": "low",'
            ' "summary_original": "தமிழக அரசு புதிய நலத்திட்டத்தை அறிவித்தது.",'
            ' "summary_english": "Tamil Nadu announced a new welfare scheme.",'
            ' "party_action": "",'
            ' "people_impact": "Benefits to people across 38 districts.",'
            ' "root_cause": "Government welfare initiative.",'
            ' "recommended_step": "Track rollout progress.",'
            ' "positive_points": ["State-wide reach"],'
            ' "negative_points": [],'
            ' "evidence_quotes_original": [],'
            ' "evidence_quotes_english": [],'
            ' "confidence": 0.85,'
            ' "needs_human_review": false}'
        )
    )
    g = _make_gemma_with_fake_client(client)
    result = g.analyze(_make_item())
    assert isinstance(result, AIAnalysis)
    assert result.stance_toward_government == Stance.POSITIVE
    assert result.government_relevance == GovernmentRelevance.HIGH
    assert result.summary_english.startswith("Tamil Nadu announced")
    assert result.confidence == 0.85
    client.generate.assert_called_once()


def test_analyze_recovers_from_code_fenced_json():
    """Gemma sometimes wraps JSON in ```json ... ``` despite format='json'."""
    client = MagicMock()
    client.generate.return_value = SimpleNamespace(
        response=(
            "```json\n"
            '{"government_relevance": "low",'
            ' "stance_toward_government": "neutral",'
            ' "sentiment": "neutral",'
            ' "target": "Public matter",'
            ' "department": "general",'
            ' "district": "unspecified",'
            ' "scheme": null,'
            ' "topic": "Test",'
            ' "issue_category": "general",'
            ' "severity": "low",'
            ' "summary_original": "Test.",'
            ' "summary_english": "Test.",'
            ' "confidence": 0.5,'
            ' "needs_human_review": false}\n'
            "```"
        )
    )
    g = _make_gemma_with_fake_client(client)
    result = g.analyze(_make_item())
    assert result.stance_toward_government == Stance.NEUTRAL


def test_analyze_raises_unavailable_on_malformed_json():
    """When Gemma returns garbage that isn't JSON, raise so the cascade
    falls through to LocalTamilAnalyzer."""
    client = MagicMock()
    client.generate.return_value = SimpleNamespace(
        response="I'm sorry, I cannot generate that."
    )
    g = _make_gemma_with_fake_client(client)
    with pytest.raises(GemmaAnalyzerUnavailable):
        g.analyze(_make_item())


def test_analyze_raises_unavailable_on_empty_response():
    client = MagicMock()
    client.generate.return_value = SimpleNamespace(response="")
    g = _make_gemma_with_fake_client(client)
    with pytest.raises(GemmaAnalyzerUnavailable):
        g.analyze(_make_item())
