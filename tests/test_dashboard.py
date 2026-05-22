from tests.test_storage import make_analysis, make_item
from tnmi.contracts import GovernmentRelevance, ReviewDecisionCreate, ReviewStatus, Severity, Stance
from tnmi.dashboard import get_dashboard_summary, list_review_queue
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item, save_review_decision


def test_dashboard_summary_counts_analysis_and_review_status(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'dashboard.db'}")
    init_db(session_factory)

    negative = make_analysis().model_copy(
        update={
            "stance_toward_government": Stance.NEGATIVE,
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
    assert summary["total_analyses"] == 2
    assert summary["needs_human_review"] == 1
    assert summary["reviewed"] == 1
    assert summary["pending_review"] == 0
    assert summary["stance_counts"] == {"negative": 1, "positive": 1}
    assert summary["severity_counts"]["high"] == 1
    assert summary["department_counts"]["transport"] == 1
    assert summary["district_counts"]["Chennai"] == 1


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
