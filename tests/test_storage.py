from datetime import datetime, timezone

from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
    Sentiment,
    Severity,
    SourceType,
    Stance,
)
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item


def test_save_raw_item_is_idempotent_by_content_hash(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/a",
        published_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        language="ta",
        title="Title",
        raw_text_original="தமிழக அரசு செய்தி",
        clean_text_original="தமிழக அரசு செய்தி",
    )

    with session_factory() as session:
        first = save_raw_item(session, item)
        second = save_raw_item(session, item)
        session.commit()

    assert first.id == second.id


def test_save_ai_analysis_for_raw_item(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/a",
        published_at=None,
        language="en",
        title="Title",
        raw_text_original="Government announced a scheme",
        clean_text_original="Government announced a scheme",
    )
    analysis = AIAnalysis(
        government_relevance=GovernmentRelevance.HIGH,
        stance_toward_government=Stance.POSITIVE,
        sentiment=Sentiment.POSITIVE,
        target="Tamil Nadu Government",
        department="welfare",
        district="unknown",
        scheme=None,
        topic="scheme",
        issue_category="welfare",
        severity=Severity.LOW,
        summary_original="Positive item.",
        summary_english="Positive item.",
        positive_points=["positive"],
        negative_points=[],
        evidence_quotes_original=["announced a scheme"],
        evidence_quotes_english=["announced a scheme"],
        confidence=0.9,
        needs_human_review=False,
    )

    with session_factory() as session:
        raw = save_raw_item(session, item)
        saved = save_ai_analysis(session, raw.id, analysis, model_name="mock", prompt_version="v1")
        session.commit()

    assert saved.raw_item_id == raw.id
