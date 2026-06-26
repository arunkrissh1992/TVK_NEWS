"""Entity resolution — bilingual aliases, candidate queueing, idempotency."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select

from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
    Sentiment,
    Severity,
    SourceType,
    Stance,
)
from tnmi.resolver import (
    RESOLVER_VERSION,
    AliasIndex,
    normalize_surface,
    pick_best_analyses,
    resolve_all,
    slugify,
    sync_district_entities,
    sync_seed_entities,
)
from tnmi.storage import (
    AIAnalysisRecord,
    EntityRecord,
    ItemEntityRecord,
    create_session_factory,
    init_db,
    save_ai_analysis,
    save_raw_item,
)

SEED_PATH = "configs/entities.seed.yaml"


def _factory(tmp_path):
    factory = create_session_factory(f"sqlite:///{tmp_path / 'resolver.db'}")
    init_db(factory)
    return factory


def _item(url: str = "https://example.com/a", source_name: str = "The Hindu (Chennai)") -> NormalizedItem:
    return NormalizedItem(
        source_type=SourceType.NEWS,
        source_name=source_name,
        source_url=url,
        published_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
        language="ta",
        title="Title",
        raw_text_original="உள்ளடக்கம்",
        clean_text_original="உள்ளடக்கம்",
    )


def _analysis(
    *,
    actors: list[str] | None = None,
    district: str = "unspecified",
    department: str = "general",
    scheme: str | None = None,
) -> AIAnalysis:
    return AIAnalysis(
        government_relevance=GovernmentRelevance.HIGH,
        stance_toward_government=Stance.NEGATIVE,
        sentiment=Sentiment.NEGATIVE,
        target="government",
        political_actors=actors or [],
        department=department,
        district=district,
        scheme=scheme,
        topic="issue",
        issue_category="civic",
        severity=Severity.HIGH,
        summary_original="சுருக்கம்",
        summary_english="Summary.",
        confidence=0.9,
        needs_human_review=False,
    )


def test_normalize_surface_unifies_bilingual_variants():
    assert normalize_surface("தி.மு.க") == normalize_surface("திமுக")
    assert normalize_surface("M.K. Stalin") == normalize_surface("MK Stalin")
    assert normalize_surface("  Chief  Minister ") == "chief minister"


def test_seed_sync_is_idempotent(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        first = sync_seed_entities(session, SEED_PATH)
        second = sync_seed_entities(session, SEED_PATH)
        session.commit()
        assert first == second
        total = len(session.scalars(select(EntityRecord)).all())
        assert total == first  # no duplicate rows on re-sync


def test_alias_lookup_resolves_bilingual_and_mock_labels(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        sync_seed_entities(session, SEED_PATH)
        session.commit()
        index = AliasIndex(session)
        # Real-data surfaces (Gemma) and mock labels must land on the same nodes.
        assert index.lookup("DMK").slug == "dmk"
        assert index.lookup("DMK (opposition)").slug == "dmk"
        assert index.lookup("திமுக").slug == "dmk"
        assert index.lookup("Vijay").slug == "vijay"
        assert index.lookup("Vijay (CM)").slug == "vijay"
        assert index.lookup("Chief Minister").slug == "office-chief-minister"
        assert index.lookup("முதலமைச்சர்").slug == "office-chief-minister"
        assert index.lookup("எடப்பாடி பழனிசாமி").slug == "edappadi-palaniswami"
        assert index.lookup("EPS").slug == "edappadi-palaniswami"
        # Unseen parenthetical falls back to the stripped form.
        assert index.lookup("AIADMK (opposition)").slug == "aiadmk"


def test_district_sync_covers_all_38(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        count = sync_district_entities(session)
        session.commit()
        assert count == 38
        madurai = session.scalar(select(EntityRecord).where(EntityRecord.slug == "madurai"))
        assert madurai is not None
        assert madurai.name_ta  # Tamil display name picked up from the registry


def test_resolve_all_creates_edges_and_candidates(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, _item())
        save_ai_analysis(
            session,
            raw.id,
            _analysis(
                actors=["Vijay", "Chief Minister", "DMK", "Mystery Person"],
                district="மதுரை",
                department="Transport / road safety",
            ),
            model_name="ollama/gemma2:2b",
            prompt_version="tvk-portrayal-v16",
        )
        session.commit()

        stats = resolve_all(session, seed_path=SEED_PATH)
        session.commit()

        edges = session.scalars(select(ItemEntityRecord)).all()
        by_field = {}
        for edge in edges:
            by_field.setdefault(edge.mention_field, set()).add(edge.entity_id)
        entity = lambda eid: session.get(EntityRecord, eid)  # noqa: E731

        actor_slugs = {entity(eid).slug for eid in by_field["political_actors"]}
        assert {"vijay", "office-chief-minister", "dmk", "candidate-mystery-person"} == actor_slugs
        district_slugs = {entity(eid).slug for eid in by_field["district"]}
        assert district_slugs == {"madurai"}
        dept_slugs = {entity(eid).slug for eid in by_field["department"]}
        assert dept_slugs == {"dept-transport"}
        source_slugs = {entity(eid).slug for eid in by_field["source"]}
        assert source_slugs == {"source-the-hindu-chennai"}

        mystery = session.scalar(
            select(EntityRecord).where(EntityRecord.slug == "candidate-mystery-person")
        )
        assert mystery.status == "candidate"  # queued, never dropped
        assert stats.candidate_surfaces == 1
        assert stats.resolution_rate < 1.0
        assert all(edge.resolver_version == RESOLVER_VERSION for edge in edges)


def test_resolve_all_is_idempotent(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, _item())
        save_ai_analysis(
            session,
            raw.id,
            _analysis(actors=["Vijay", "TVK"], district="Chennai"),
            model_name="ollama/gemma2:2b",
            prompt_version="tvk-portrayal-v16",
        )
        session.commit()

        resolve_all(session, seed_path=SEED_PATH)
        session.commit()
        first_edges = len(session.scalars(select(ItemEntityRecord)).all())

        second = resolve_all(session, seed_path=SEED_PATH)
        session.commit()
        second_edges = len(session.scalars(select(ItemEntityRecord)).all())

        assert first_edges == second_edges
        assert second.mentions_created == 0


def test_better_analysis_replaces_old_edges(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, _item())
        save_ai_analysis(
            session,
            raw.id,
            _analysis(actors=["DMK"]),
            model_name="mock",
            prompt_version="tvk-portrayal-v15",
        )
        session.commit()
        resolve_all(session, seed_path=SEED_PATH)
        session.commit()

        save_ai_analysis(
            session,
            raw.id,
            _analysis(actors=["Vijay"]),
            model_name="ollama/gemma2:2b",
            prompt_version="tvk-portrayal-v16",
        )
        session.commit()
        resolve_all(session, seed_path=SEED_PATH)
        session.commit()

        actor_edges = session.scalars(
            select(ItemEntityRecord).where(ItemEntityRecord.mention_field == "political_actors")
        ).all()
        slugs = {session.get(EntityRecord, edge.entity_id).slug for edge in actor_edges}
        assert slugs == {"vijay"}  # mock-era DMK edge replaced by the real model's view


def test_pick_best_analyses_prefers_non_mock_then_latest(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, _item())
        mock_row = save_ai_analysis(
            session, raw.id, _analysis(), model_name="mock", prompt_version="v1"
        )
        real_row = save_ai_analysis(
            session, raw.id, _analysis(), model_name="ollama/gemma2:2b", prompt_version="v1"
        )
        session.commit()
        rows = session.scalars(select(AIAnalysisRecord)).all()
        best = pick_best_analyses(rows)
        assert best[raw.id].id == real_row.id
        assert best[raw.id].id != mock_row.id


def test_slugify_handles_tamil_and_punctuation():
    assert slugify("The Hindu (Chennai)") == "the-hindu-chennai"
    assert slugify("மதுரை") == "மதுரை"
    assert slugify("Transport / road safety") == "transport-road-safety"
