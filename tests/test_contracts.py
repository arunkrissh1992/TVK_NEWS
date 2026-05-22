from datetime import datetime, timezone

from tnmi import __version__
from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
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
    assert item.content_hash_input().startswith("news|https://example.com/article|")


def test_ai_analysis_schema_has_evidence_and_review_flag():
    analysis = AIAnalysis(
        government_relevance=GovernmentRelevance.HIGH,
        stance_toward_government=Stance.POSITIVE,
        sentiment="positive",
        target="Tamil Nadu Government",
        department="welfare",
        district="unknown",
        scheme=None,
        topic="new scheme",
        issue_category="welfare",
        severity="low",
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
