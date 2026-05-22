from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from tnmi import __version__
from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
    Sentiment,
    Severity,
    SourceType,
    Stance,
)


def test_package_imports():
    assert __version__ == "0.1.0"


def test_normalized_item_accepts_tamil_news_article():
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example Tamil Daily",
        source_url="https://example.com/article",
        published_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        language="ta",
        title="தமிழக அரசு புதிய திட்டம் அறிவிப்பு",
        raw_text_original="தமிழக அரசு இன்று புதிய திட்டத்தை அறிவித்தது.",
        clean_text_original="தமிழக அரசு இன்று புதிய திட்டத்தை அறிவித்தது.",
        metadata={"section": "politics"},
    )

    assert item.source_type == SourceType.NEWS
    assert item.content_hash_input().startswith('{"clean_text_original":')


def test_content_hash_input_distinguishes_pipe_collision_values():
    first = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example Tamil Daily",
        source_url="https://example.com/a",
        published_at=None,
        language="ta",
        title="b|c",
        raw_text_original="d",
        clean_text_original="d",
    )
    second = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example Tamil Daily",
        source_url="https://example.com/a|b",
        published_at=None,
        language="ta",
        title="c",
        raw_text_original="d",
        clean_text_original="d",
    )

    pipe_join_first = "news|https://example.com/a|b|c|d"
    pipe_join_second = "news|https://example.com/a|b|c|d"

    assert pipe_join_first == pipe_join_second
    assert first.content_hash_input() != second.content_hash_input()


def test_ai_analysis_schema_has_evidence_and_review_flag():
    analysis = AIAnalysis(
        government_relevance=GovernmentRelevance.HIGH,
        stance_toward_government=Stance.POSITIVE,
        sentiment=Sentiment.POSITIVE,
        target="Tamil Nadu Government",
        department="welfare",
        district="unknown",
        scheme=None,
        topic="new scheme",
        issue_category="welfare",
        severity=Severity.LOW,
        summary_original="அரசு திட்டம் குறித்து சாதகமான செய்தி.",
        summary_english="Positive coverage about a government scheme.",
        positive_points=["Scheme announcement was described favorably."],
        negative_points=[],
        evidence_quotes_original=["புதிய திட்டத்தை அறிவித்தது"],
        evidence_quotes_english=["announced a new scheme"],
        confidence=0.86,
        needs_human_review=False,
    )

    assert analysis.confidence == 0.86
    assert analysis.needs_human_review is False


def test_ai_analysis_rejects_invalid_sentiment():
    with pytest.raises(ValidationError):
        AIAnalysis(
            government_relevance=GovernmentRelevance.HIGH,
            stance_toward_government=Stance.POSITIVE,
            sentiment="excited",
            target="Tamil Nadu Government",
            department="welfare",
            district="unknown",
            scheme=None,
            topic="new scheme",
            issue_category="welfare",
            severity=Severity.LOW,
            summary_original="அரசு திட்டம் குறித்து சாதகமான செய்தி.",
            summary_english="Positive coverage about a government scheme.",
            positive_points=["Scheme announcement was described favorably."],
            negative_points=[],
            evidence_quotes_original=["புதிய திட்டத்தை அறிவித்தது"],
            evidence_quotes_english=["announced a new scheme"],
            confidence=0.86,
            needs_human_review=False,
        )


def test_ai_analysis_rejects_invalid_severity():
    with pytest.raises(ValidationError):
        AIAnalysis(
            government_relevance=GovernmentRelevance.HIGH,
            stance_toward_government=Stance.POSITIVE,
            sentiment="positive",
            target="Tamil Nadu Government",
            department="welfare",
            district="unknown",
            scheme=None,
            topic="new scheme",
            issue_category="welfare",
            severity="urgent",
            summary_original="அரசு திட்டம் குறித்து சாதகமான செய்தி.",
            summary_english="Positive coverage about a government scheme.",
            positive_points=["Scheme announcement was described favorably."],
            negative_points=[],
            evidence_quotes_original=["புதிய திட்டத்தை அறிவித்தது"],
            evidence_quotes_english=["announced a new scheme"],
            confidence=0.86,
            needs_human_review=False,
        )
