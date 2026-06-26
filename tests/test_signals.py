from datetime import datetime, timedelta, timezone

from tnmi.contracts import GovernmentRelevance, Severity, Stance
from tnmi.entity_api import invalidate_entity_cache
from tnmi.resolver import resolve_all
from tnmi.signals import detect_spikes
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item

from tests.test_storage import make_analysis, make_item

SEED = "configs/entities.seed.yaml"


def _factory(tmp_path):
    factory = create_session_factory(f"sqlite:///{tmp_path / 'signals.db'}")
    init_db(factory)
    return factory


def _story(session, *, idx, actors, portrayal, days_ago, severity=Severity.MEDIUM):
    raw = save_raw_item(
        session,
        make_item().model_copy(
            update={
                "source_url": f"https://e.com/{idx}",
                "title": f"Story {idx}",
                "raw_text_original": "Body text mentioning the actor.",
                "clean_text_original": "Body text mentioning the actor.",
                "published_at": datetime.now(timezone.utc) - timedelta(days=days_ago),
            }
        ),
    )
    save_ai_analysis(
        session,
        raw.id,
        make_analysis().model_copy(
            update={
                "political_actors": actors,
                "tvk_portrayal": portrayal,
                "stance_toward_government": portrayal,
                "government_relevance": GovernmentRelevance.HIGH,
                "severity": severity,
            }
        ),
        model_name="mock",
        prompt_version="v19",
    )


def test_detect_spikes_flags_recent_negative_surge_as_threat(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        # Baseline: one Stalin story per day, 4–12 days ago (quiet).
        for d in range(4, 13):
            _story(session, idx=f"base{d}", actors=["M.K. Stalin"], portrayal=Stance.NEUTRAL, days_ago=d)
        # Recent surge: 6 negative Stalin stories in the last 2 days.
        for i in range(6):
            _story(session, idx=f"hot{i}", actors=["M.K. Stalin"], portrayal=Stance.NEGATIVE,
                   days_ago=i % 2, severity=Severity.HIGH)
        session.commit()
        resolve_all(session, seed_path=SEED)
        session.commit()
        invalidate_entity_cache()

        spikes = detect_spikes(session, recent_days=3, baseline_days=10)

    stalin = next((s for s in spikes if s["slug"] == "mk-stalin"), None)
    assert stalin is not None
    assert stalin["is_threat"] is True
    assert stalin["label"] == "Emerging threat"
    assert stalin["negative_share"] >= 0.3
    assert stalin["z_score"] >= 1.0
    assert stalin["headline"]["title"]  # a story to drill into


def test_detect_spikes_classifies_neutral_surge_as_developing_story(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        for i in range(6):
            _story(session, idx=f"pos{i}", actors=["Vijay (CM)"], portrayal=Stance.POSITIVE, days_ago=i % 2)
        session.commit()
        resolve_all(session, seed_path=SEED)
        session.commit()
        invalidate_entity_cache()

        spikes = detect_spikes(session, recent_days=3, baseline_days=10, min_recent=3)

    vijay = next((s for s in spikes if s["slug"] == "vijay"), None)
    assert vijay is not None
    assert vijay["is_threat"] is False
    assert vijay["label"] == "Developing story"


def test_threats_only_filter_drops_non_hostile_surges(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        for i in range(6):
            _story(session, idx=f"pos{i}", actors=["Vijay (CM)"], portrayal=Stance.POSITIVE, days_ago=i % 2)
        session.commit()
        resolve_all(session, seed_path=SEED)
        session.commit()
        invalidate_entity_cache()

        spikes = detect_spikes(session, recent_days=3, baseline_days=10, min_recent=3, threats_only=True)

    assert all(s["is_threat"] for s in spikes)
    assert not any(s["slug"] == "vijay" for s in spikes)
