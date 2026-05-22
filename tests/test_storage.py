from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
    Sentiment,
    Severity,
    SourceType,
    Stance,
)
from tnmi.storage import (
    AIAnalysisRecord,
    RawItemRecord,
    create_session_factory,
    init_db,
    save_ai_analysis,
    save_raw_item,
)


def make_item() -> NormalizedItem:
    return NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/a",
        published_at=None,
        language="en",
        title="Title",
        raw_text_original="Government announced a scheme",
        clean_text_original="Government announced a scheme",
    )


def make_analysis() -> AIAnalysis:
    return AIAnalysis(
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
    item = make_item()
    analysis = make_analysis()

    with session_factory() as session:
        raw = save_raw_item(session, item)
        saved = save_ai_analysis(session, raw.id, analysis, model_name="mock", prompt_version="v1")
        session.commit()

    assert saved.raw_item_id == raw.id


def test_save_ai_analysis_is_idempotent_for_raw_model_and_prompt(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    item = make_item()
    analysis = make_analysis()

    with session_factory() as session:
        raw = save_raw_item(session, item)
        first = save_ai_analysis(session, raw.id, analysis, model_name="mock", prompt_version="v1")
        second = save_ai_analysis(session, raw.id, analysis, model_name="mock", prompt_version="v1")
        row_count = session.scalar(select(func.count()).select_from(AIAnalysisRecord))
        session.commit()

    assert first.id == second.id
    assert row_count == 1


def test_save_ai_analysis_rejects_nonexistent_raw_item(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)

    with session_factory() as session:
        with pytest.raises(IntegrityError):
            save_ai_analysis(session, 999, make_analysis(), model_name="mock", prompt_version="v1")


def test_save_raw_item_recovers_from_duplicate_insert_conflict(tmp_path, monkeypatch):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    item = make_item()

    with session_factory() as session:
        existing = save_raw_item(session, item)
        session.commit()
        existing_id = existing.id

    with session_factory() as session:
        original_scalar = session.scalar
        scalar_calls = 0

        def scalar_with_race(*args, **kwargs):
            nonlocal scalar_calls
            scalar_calls += 1
            if scalar_calls == 1:
                return None
            return original_scalar(*args, **kwargs)

        monkeypatch.setattr(session, "scalar", scalar_with_race)

        saved = save_raw_item(session, item)
        session.commit()

    assert saved.id == existing_id
    assert scalar_calls == 2


def test_postgresql_ddl_includes_json_server_defaults():
    dialect = postgresql.dialect()
    raw_items_ddl = str(CreateTable(RawItemRecord.__table__).compile(dialect=dialect))
    ai_analysis_ddl = str(CreateTable(AIAnalysisRecord.__table__).compile(dialect=dialect))

    assert "metadata_json JSONB DEFAULT '{}'" in raw_items_ddl
    assert "positive_points JSONB DEFAULT '[]'" in ai_analysis_ddl
    assert "negative_points JSONB DEFAULT '[]'" in ai_analysis_ddl
    assert "evidence_quotes_original JSONB DEFAULT '[]'" in ai_analysis_ddl
    assert "evidence_quotes_english JSONB DEFAULT '[]'" in ai_analysis_ddl
    assert "CONSTRAINT uq_ai_analysis_raw_model_prompt UNIQUE (raw_item_id, model_name, prompt_version)" in ai_analysis_ddl
