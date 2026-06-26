from datetime import datetime, timedelta, timezone

from tnmi.contracts import GovernmentRelevance, Sentiment, Severity, Stance
from tnmi.entity_api import (
    actor_scorecards,
    entity_dossier,
    favorability,
    invalidate_entity_cache,
    list_entities,
)
from tnmi.resolver import resolve_all
from tnmi.storage import (
    AIAnalysisRecord,
    EntityAliasRecord,
    EntityRecord,
    RawItemRecord,
    create_session_factory,
    init_db,
    save_ai_analysis,
    save_raw_item,
)

from tests.test_storage import make_analysis, make_item

SEED = "configs/entities.seed.yaml"


def _factory(tmp_path):
    factory = create_session_factory(f"sqlite:///{tmp_path / 'entity.db'}")
    init_db(factory)
    return factory


def _add(session, *, url, title, body, actors, portrayal, days_ago=0):
    item = make_item().model_copy(
        update={
            "source_url": url,
            "title": title,
            "raw_text_original": body,
            "clean_text_original": body,
            "published_at": datetime.now(timezone.utc) - timedelta(days=days_ago),
        }
    )
    raw = save_raw_item(session, item)
    analysis = make_analysis().model_copy(
        update={
            "political_actors": actors,
            "tvk_portrayal": portrayal,
            "stance_toward_government": portrayal,
            "government_relevance": GovernmentRelevance.HIGH,
        }
    )
    save_ai_analysis(session, raw.id, analysis, model_name="mock", prompt_version="v19")
    return raw


def test_favorability_scoring():
    assert favorability({"positive": 4, "negative": 0}) == 100
    assert favorability({"positive": 0, "negative": 4}) == 0
    assert favorability({"positive": 2, "negative": 2}) == 50
    assert favorability({"positive": 0, "negative": 0, "neutral": 9}) is None


def test_list_entities_ranks_by_mentions_with_portrayal_split(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        _add(session, url="https://e.com/1", title="Vijay scheme", body="Vijay announced a welfare scheme in Chennai.",
             actors=["Vijay (CM)"], portrayal=Stance.POSITIVE)
        _add(session, url="https://e.com/2", title="Vijay event", body="Vijay addressed a rally in Madurai.",
             actors=["Vijay (CM)"], portrayal=Stance.NEUTRAL)
        _add(session, url="https://e.com/3", title="DMK criticism", body="DMK leader Stalin criticised the move.",
             actors=["DMK (opposition)", "M.K. Stalin"], portrayal=Stance.NEGATIVE)
        session.commit()
        resolve_all(session, seed_path=SEED)
        session.commit()
        invalidate_entity_cache()

        entities = list_entities(session, limit=50)
        by_slug = {e["slug"]: e for e in entities}
        assert "vijay" in by_slug
        assert by_slug["vijay"]["mention_count"] == 2
        assert by_slug["vijay"]["portrayal_split"]["positive"] == 1
        # Ranked by mention volume.
        assert entities[0]["mention_count"] >= entities[-1]["mention_count"]


def test_entity_dossier_has_co_mentions_and_evidence(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        _add(session, url="https://e.com/1", title="Stalin and Udhayanidhi",
             body="DMK leaders Stalin and Udhayanidhi spoke in Chennai.",
             actors=["M.K. Stalin", "Udhayanidhi", "DMK (opposition)"], portrayal=Stance.NEGATIVE)
        session.commit()
        resolve_all(session, seed_path=SEED)
        session.commit()
        invalidate_entity_cache()

        dossier = entity_dossier(session, "mk-stalin")
        assert dossier is not None
        assert dossier["name"] == "M.K. Stalin"
        assert dossier["mention_count"] == 1
        co_slugs = {c["slug"] for c in dossier["co_mentions"]}
        assert "udhayanidhi-stalin" in co_slugs  # appeared together
        assert len(dossier["evidence"]) == 1
        assert dossier["evidence"][0]["portrayal"] == "negative"
        assert len(dossier["timeseries"]) == 8  # default weeks


def test_entity_dossier_unknown_slug_returns_none(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        invalidate_entity_cache()
        assert entity_dossier(session, "nobody") is None


def test_actor_scorecards_only_people_with_favorability(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        for i in range(3):
            _add(session, url=f"https://e.com/v{i}", title="Vijay good", body="Vijay launched a scheme.",
                 actors=["Vijay (CM)"], portrayal=Stance.POSITIVE)
        _add(session, url="https://e.com/eps1", title="EPS", body="Edappadi Palaniswami led a protest.",
             actors=["Edappadi K. Palaniswami", "AIADMK"], portrayal=Stance.NEGATIVE)
        _add(session, url="https://e.com/eps2", title="EPS2", body="Edappadi Palaniswami spoke again.",
             actors=["Edappadi K. Palaniswami"], portrayal=Stance.NEGATIVE)
        session.commit()
        resolve_all(session, seed_path=SEED)
        session.commit()
        invalidate_entity_cache()

        cards = actor_scorecards(session, min_mentions=2)
        slugs = {c["slug"] for c in cards}
        # Only person entities — never parties/offices/sources.
        assert "vijay" in slugs
        assert "edappadi-palaniswami" in slugs
        assert "aiadmk" not in slugs
        vijay = next(c for c in cards if c["slug"] == "vijay")
        assert vijay["favorability"] == 100  # 3/3 positive
        eps = next(c for c in cards if c["slug"] == "edappadi-palaniswami")
        assert eps["favorability"] == 0  # 2/2 negative
