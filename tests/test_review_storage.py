from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from tests.test_storage import make_analysis, make_item
from tnmi.contracts import ReviewDecisionCreate, ReviewStatus, Stance
from tnmi.storage import (
    ReviewDecisionRecord,
    create_session_factory,
    get_latest_review_decision,
    init_db,
    save_ai_analysis,
    save_raw_item,
    save_review_decision,
)


def test_save_review_decision_records_operator_audit_row(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'review.db'}")
    init_db(session_factory)

    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        analysis = save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        decision = save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=analysis.id,
                reviewer_name="analyst-1",
                status=ReviewStatus.ESCALATED,
                note="Allegation requires department confirmation.",
                corrected_stance=Stance.MIXED,
                corrected_summary="Reviewed as mixed because the article praises one response and raises one concern.",
            ),
        )
        session.commit()

    assert decision.analysis_id == analysis.id
    assert decision.status == "escalated"
    assert decision.corrected_stance == "mixed"


def test_get_latest_review_decision_returns_newest_row(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'review.db'}")
    init_db(session_factory)

    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        analysis = save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=analysis.id,
                reviewer_name="analyst-1",
                status=ReviewStatus.APPROVED,
                note="Initial review.",
            ),
        )
        latest = save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=analysis.id,
                reviewer_name="lead-1",
                status=ReviewStatus.ESCALATED,
                note="Escalate after supervisor check.",
            ),
        )
        found = get_latest_review_decision(session, analysis.id)
        session.commit()

    assert found is not None
    assert found.id == latest.id
    assert found.reviewer_name == "lead-1"


def test_review_decision_postgresql_ddl_has_audit_fields():
    ddl = str(CreateTable(ReviewDecisionRecord.__table__).compile(dialect=postgresql.dialect()))

    assert "review_decisions" in ddl
    assert "analysis_id" in ddl
    assert "reviewer_name" in ddl
    assert "corrected_summary" in ddl
