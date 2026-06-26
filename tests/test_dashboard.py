from tests.test_storage import make_analysis, make_item
from tnmi.embeddings import HashEmbeddingProvider
from tnmi.rag import RAGIndexer
from tnmi.contracts import GovernmentRelevance, ReviewDecisionCreate, ReviewStatus, Severity, Stance
from tnmi.dashboard import (
    get_dashboard_summary,
    list_latest_items,
    list_review_queue,
    select_priority_alerts,
    summarize_briefing_categories,
)
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item, save_review_decision
from tnmi.vector_index import InMemoryVectorIndex


def test_dashboard_summary_counts_analysis_and_review_status(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'dashboard.db'}")
    init_db(session_factory)

    negative = make_analysis().model_copy(
        update={
            "stance_toward_government": Stance.NEGATIVE,
            "tvk_portrayal": Stance.NEGATIVE,
            "severity": Severity.HIGH,
            "government_relevance": GovernmentRelevance.HIGH,
            "needs_human_review": True,
            "department": "transport",
            "district": "Chennai",
            "summary_english": "Negative road issue.",
        }
    )
    positive = make_analysis().model_copy(
        update={
            "stance_toward_government": Stance.POSITIVE,
            "tvk_portrayal": Stance.POSITIVE,
            "severity": Severity.LOW,
            "needs_human_review": False,
            "department": "health",
            "district": "Madurai",
            "summary_english": "Positive health item.",
        }
    )

    with session_factory() as session:
        raw_one = save_raw_item(session, make_item().model_copy(update={"source_url": "https://example.com/one"}))
        analysis_one = save_ai_analysis(session, raw_one.id, negative, model_name="mock", prompt_version="v1")
        raw_two = save_raw_item(session, make_item().model_copy(update={"source_url": "https://example.com/two"}))
        save_ai_analysis(session, raw_two.id, positive, model_name="mock", prompt_version="v1")
        save_ai_analysis(session, raw_two.id, positive, model_name="gpt-5-mini", prompt_version="v1")
        RAGIndexer(
            embedding_provider=HashEmbeddingProvider(dimension=8),
            vector_index=InMemoryVectorIndex(dimension=8),
            max_chars=24,
            overlap_chars=6,
        ).index_raw_item(session, raw_two)
        save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=analysis_one.id,
                reviewer_name="analyst-1",
                status=ReviewStatus.ESCALATED,
                note="Needs department confirmation.",
            ),
        )
        summary = get_dashboard_summary(session)
        session.commit()

    assert summary["total_items"] == 2
    assert summary["total_analyses"] == 3
    assert summary["total_chunks"] >= 1
    assert summary["total_embeddings"] >= 1
    assert summary["openai_analyses"] == 1
    assert summary["semantic_analyses"] == 1
    assert summary["fallback_analyses"] == 1
    assert summary["keyword_analyses"] == 0
    assert summary["mock_analyses"] == 2
    assert summary["needs_human_review"] == 1
    assert summary["reviewed"] == 1
    assert summary["pending_review"] == 0
    # Stance counts dedupe to one analysis per raw item (prefer non-mock).
    # raw_one → negative (mock, only one). raw_two → positive (gpt-5-mini wins over mock).
    assert summary["stance_counts"] == {"negative": 1, "positive": 1}
    assert summary["severity_counts"]["high"] == 1
    assert summary["department_counts"]["transport"] == 1
    assert summary["district_counts"]["Chennai"] == 1
    assert summary["source_counts"]["Example"] == 2
    assert summary["analysis_model_counts"]["gpt-5-mini"] == 1
    assert "local/hash-embedding-v1" in summary["embedding_provider_counts"]


def test_dashboard_summary_counts_keyword_analyzer_as_fallback(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'dashboard.db'}")
    init_db(session_factory)

    with session_factory() as session:
        raw = save_raw_item(session, make_item().model_copy(update={"source_url": "https://example.com/local"}))
        save_ai_analysis(
            session,
            raw.id,
            make_analysis(),
            model_name="local-tamil-keywords",
            prompt_version="v1",
        )
        summary = get_dashboard_summary(session)
        session.commit()

    assert summary["openai_analyses"] == 0
    assert summary["semantic_analyses"] == 0
    assert summary["fallback_analyses"] == 1
    assert summary["keyword_analyses"] == 1


def test_review_queue_prioritizes_unreviewed_high_severity_items(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'dashboard.db'}")
    init_db(session_factory)

    critical = make_analysis().model_copy(
        update={
            "stance_toward_government": Stance.NEGATIVE,
            "severity": Severity.CRITICAL,
            "needs_human_review": True,
            "confidence": 0.55,
            "summary_english": "Critical allegation.",
        }
    )
    low = make_analysis().model_copy(
        update={
            "stance_toward_government": Stance.MIXED,
            "severity": Severity.LOW,
            "needs_human_review": True,
            "confidence": 0.4,
            "summary_english": "Low severity issue.",
        }
    )

    with session_factory() as session:
        raw_low = save_raw_item(session, make_item().model_copy(update={"source_url": "https://example.com/low"}))
        save_ai_analysis(session, raw_low.id, low, model_name="mock", prompt_version="v1")
        raw_critical = save_raw_item(
            session, make_item().model_copy(update={"source_url": "https://example.com/critical"})
        )
        analysis_critical = save_ai_analysis(session, raw_critical.id, critical, model_name="mock", prompt_version="v1")
        queue = list_review_queue(session, limit=10)
        session.commit()

    assert queue[0]["analysis_id"] == analysis_critical.id
    assert queue[0]["review_status"] == "pending"
    assert queue[0]["severity"] == "critical"
    assert queue[0]["stance"] == "negative"


def test_latest_items_returns_recent_analyzed_newspaper_items(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'dashboard.db'}")
    init_db(session_factory)

    with session_factory() as session:
        raw = save_raw_item(session, make_item().model_copy(update={"title": "Real newspaper item"}))
        analysis = save_ai_analysis(
            session,
            raw.id,
            make_analysis().model_copy(update={"summary_english": "Visible business demo summary."}),
            model_name="mock",
            prompt_version="v1",
        )
        latest = list_latest_items(session, limit=10)
        session.commit()

    assert latest[0]["raw_item_id"] == raw.id
    assert latest[0]["analysis_id"] == analysis.id
    assert latest[0]["title"] == "Real newspaper item"
    assert latest[0]["summary"] == "Visible business demo summary."
    assert latest[0]["model_name"] == "mock"
    assert latest[0]["stance_label"] == "Positive / நேர்மறை"
    assert latest[0]["portrayal_kind"] == "positive"
    assert latest[0]["evidence_original"]


def test_latest_items_displays_neutral_people_issue_as_people_issue(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'dashboard.db'}")
    init_db(session_factory)

    people_issue = make_analysis().model_copy(
        update={
            "stance_toward_government": Stance.NEUTRAL,
            "tvk_portrayal": Stance.NEUTRAL,
            "people_issue": True,
            "public_issue": "school safety incident",
            "severity": Severity.HIGH,
            "recommended_step": "Send the district team to verify injuries and response.",
            "action_owner": "District field team",
            "action_type": "field_verification",
        }
    )

    with session_factory() as session:
        raw = save_raw_item(
            session,
            make_item().model_copy(
                update={
                    "title": "சென்னை பள்ளி அருகே தீ விபத்து",
                    "source_url": "https://example.com/school-fire",
                }
            ),
        )
        save_ai_analysis(session, raw.id, people_issue, model_name="mock", prompt_version="v2")
        latest = list_latest_items(session, limit=10)
        session.commit()

    assert latest[0]["stance"] == "neutral"
    assert latest[0]["people_issue"] is True
    assert latest[0]["portrayal_kind"] == "people"
    assert latest[0]["stance_label"] == "People Issue / மக்கள் பிரச்சனை"
    assert latest[0]["public_issue"] == "school safety incident"


def test_latest_items_prefers_live_ai_analysis_over_mock_for_same_raw_item(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'dashboard.db'}")
    init_db(session_factory)

    with session_factory() as session:
        raw = save_raw_item(session, make_item().model_copy(update={"title": "Live AI item"}))
        save_ai_analysis(
            session,
            raw.id,
            make_analysis().model_copy(update={"summary_english": "Mock summary."}),
            model_name="mock",
            prompt_version="v1",
        )
        live_analysis = save_ai_analysis(
            session,
            raw.id,
            make_analysis().model_copy(update={"summary_english": "OpenAI summary."}),
            model_name="gpt-5-mini",
            prompt_version="v1",
        )
        latest = list_latest_items(session, limit=10)
        session.commit()

    assert len(latest) == 1
    assert latest[0]["analysis_id"] == live_analysis.id
    assert latest[0]["summary"] == "OpenAI summary."
    assert latest[0]["model_name"] == "gpt-5-mini"


def _alert_item(**overrides):
    base = {
        "raw_item_id": 1,
        "analysis_id": 1,
        "title": "Sample story",
        "source_name": "Daily Thanthi",
        "source_count": 1,
        "display_category": "negative",
        "stance_label": "Negative / எதிர்மறை",
        "portrayal_kind": "negative",
        "action_priority": "high",
        "severity": "high",
        "needs_human_review": False,
        "public_issue": "",
        "summary": "Something happened.",
        "risk_if_ignored": "Narrative hardens.",
        "recommended_step": "Verify locally first.",
        "published_at": None,
        "ingested_at": None,
    }
    base.update(overrides)
    return base


def test_select_priority_alerts_ranks_critical_above_high():
    items = [
        _alert_item(raw_item_id=1, action_priority="high", severity="high"),
        _alert_item(raw_item_id=2, action_priority="critical", severity="critical"),
    ]
    alerts = select_priority_alerts(items, limit=5)
    assert [a["raw_item_id"] for a in alerts] == [2, 1]


def test_select_priority_alerts_excludes_positive_and_low_priority():
    items = [
        _alert_item(raw_item_id=1, display_category="positive", portrayal_kind="positive"),
        _alert_item(raw_item_id=2, display_category="neutral", portrayal_kind="neutral"),
        _alert_item(
            raw_item_id=3,
            display_category="negative",
            action_priority="low",
            severity="low",
            needs_human_review=False,
        ),
    ]
    assert select_priority_alerts(items, limit=5) == []


def test_select_priority_alerts_includes_low_priority_when_awaiting_review():
    items = [
        _alert_item(
            raw_item_id=7,
            display_category="people",
            portrayal_kind="people",
            action_priority="low",
            severity="low",
            needs_human_review=True,
        )
    ]
    alerts = select_priority_alerts(items, limit=5)
    assert len(alerts) == 1
    assert alerts[0]["raw_item_id"] == 7
    assert alerts[0]["needs_human_review"] is True


def test_select_priority_alerts_respects_limit():
    items = [_alert_item(raw_item_id=i, action_priority="critical", severity="critical") for i in range(10)]
    assert len(select_priority_alerts(items, limit=3)) == 3


def test_briefing_category_counts_reconcile_with_cards():
    """The KPI deck is a filter over the cards, so its tiles must be derived
    from the cards and the buckets must sum to the total shown."""
    cards = [
        {"display_category": "positive"},
        {"display_category": "positive"},
        {"display_category": "negative"},
        {"display_category": "mixed"},
        {"display_category": "people"},
        {"display_category": "people"},
        {"display_category": "neutral"},
        {"display_category": None},  # defaults to neutral
    ]
    deck = summarize_briefing_categories(cards)
    assert deck["briefing_total"] == len(cards)
    assert deck["positive_count"] == 2
    assert deck["negative_count"] == 1
    assert deck["mixed_count"] == 1
    assert deck["people_issue_count"] == 2
    assert deck["neutral_count"] == 2
    # Every card lands in exactly one bucket — the headline always reconciles.
    bucketed = (
        deck["positive_count"]
        + deck["negative_count"]
        + deck["mixed_count"]
        + deck["people_issue_count"]
        + deck["neutral_count"]
    )
    assert bucketed == deck["briefing_total"]


def test_compose_brief_synthesises_ranked_lines():
    from tnmi.dashboard import compose_brief

    brief = compose_brief(
        summary={"positive_count": 30, "negative_count": 10},
        emerging_signals=[
            {"is_threat": True, "label": "Emerging threat", "name": "AIADMK",
             "entity_type": "party", "recent_mentions": 12, "z_score": 2.1, "slug": "aiadmk"},
        ],
        priority_alerts=[{"title": "Water protest", "risk_if_ignored": "spreads"}],
        district_summary={"tiles": [
            {"district": "Madurai", "total": 5, "negative": 3, "people": 1,
             "top_issues": [{"issue": "water shortage", "count": 2}]},
            {"district": "Salem", "total": 1, "negative": 0, "people": 0},
        ]},
        actors=[
            {"name": "Vijay", "party": "TVK", "is_tvk": True, "favorability": 70, "slug": "vijay"},
            {"name": "M.K. Stalin", "party": "DMK", "is_tvk": False, "favorability": 40,
             "momentum": -5, "slug": "mk-stalin"},
        ],
    )
    kinds = [l["kind"] for l in brief]
    assert kinds == ["standing", "signal", "alert", "hotspot", "rival"]
    standing = brief[0]
    assert "75/100" in standing["title"] and standing["tone"] == "good"  # (30-10)/40 → 75
    assert brief[1]["slug"] == "aiadmk" and brief[1]["tone"] == "bad"
    assert "Madurai" in brief[3]["title"] and "water shortage" in brief[3]["detail"]
    rival = brief[4]
    assert rival["name"] if False else "Stalin" in rival["title"]
    assert "↓" in rival["detail"]


def test_compose_brief_is_empty_without_signals():
    from tnmi.dashboard import compose_brief

    assert compose_brief(
        summary={"positive_count": 0, "negative_count": 0},
        emerging_signals=[], priority_alerts=[],
        district_summary={"tiles": []}, actors=[],
    ) == []


def test_send_brief_render_text():
    from pipelines.send_brief import render_text

    txt = render_text([
        {"tone": "good", "title": "TVK standing: 70/100", "detail": "all good"},
        {"tone": "bad", "title": "Act now: roof collapse"},
    ])
    assert "TVK Intelligence Brief" in txt
    assert "✅ TVK standing: 70/100 — all good" in txt
    assert "🔴 Act now: roof collapse" in txt
    assert render_text([]).rstrip().endswith("No notable signals today.")
