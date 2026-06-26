from tnmi.contracts import AIAnalysis, NormalizedItem, SourceType, Stance
from tnmi.labeling import export_dataset
from tnmi.storage import (
    AIAnalysisRecord,
    create_session_factory,
    init_db,
    save_ai_analysis,
    save_raw_item,
)
from tnmi.validation import validate_analysis

from tests.test_storage import make_analysis, make_item


class FixedAnalyzer:
    """A stand-in teacher that always returns a preset analysis."""

    def __init__(self, analysis: AIAnalysis, model_name: str = "teacher-test"):
        self._analysis = analysis
        self.model_name = model_name

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        return self._analysis


def _teacher_item() -> NormalizedItem:
    return NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="X",
        source_url="https://e.com/a",
        language="ta",
        title="t",
        raw_text_original="body",
        clean_text_original="body",
    )


def _setup(tmp_path):
    factory = create_session_factory(f"sqlite:///{tmp_path / 'val.db'}")
    init_db(factory)
    return factory


def test_teacher_agreement_writes_silver_no_review(tmp_path):
    factory = _setup(tmp_path)
    student = make_analysis()
    teacher = FixedAnalyzer(make_analysis())  # identical → full agreement
    with factory() as session:
        raw = save_raw_item(session, make_item())
        rec = save_ai_analysis(session, raw.id, student, model_name="student", prompt_version="v1")
        session.commit()
        result = validate_analysis(
            session,
            raw_item_id=raw.id,
            student_analysis=student,
            student_model="student",
            analysis_id=rec.id,
            teacher=teacher,
            teacher_item=_teacher_item(),
        )
        session.commit()
        assert result.needs_review is False
        assert result.silver_written == len(result.agreed)
        assert not result.disagreed
        silver = export_dataset(session, tiers=("silver",))
        assert {r.field for r in silver}  # silver labels exist
        # The analysis was NOT pushed to review.
        assert session.get(AIAnalysisRecord, rec.id).needs_human_review is False


def test_teacher_disagreement_routes_to_review(tmp_path):
    factory = _setup(tmp_path)
    student = make_analysis()  # tvk_portrayal positive
    teacher = FixedAnalyzer(make_analysis().model_copy(update={"tvk_portrayal": Stance.NEGATIVE}))
    with factory() as session:
        raw = save_raw_item(session, make_item())
        rec = save_ai_analysis(session, raw.id, student, model_name="student", prompt_version="v1")
        session.commit()
        result = validate_analysis(
            session,
            raw_item_id=raw.id,
            student_analysis=student,
            student_model="student",
            analysis_id=rec.id,
            teacher=teacher,
            teacher_item=_teacher_item(),
        )
        session.commit()
        assert "tvk_portrayal" in result.disagreed
        assert result.needs_review is True
        # The disagreed field must NOT have a silver label.
        silver_fields = {r.field for r in export_dataset(session, tiers=("silver",))}
        assert "tvk_portrayal" not in silver_fields
        # Item was routed to the human review queue.
        assert session.get(AIAnalysisRecord, rec.id).needs_human_review is True


def test_no_teacher_high_confidence_becomes_silver(tmp_path):
    factory = _setup(tmp_path)
    student = make_analysis().model_copy(update={"confidence": 0.95})
    with factory() as session:
        raw = save_raw_item(session, make_item())
        rec = save_ai_analysis(session, raw.id, student, model_name="student", prompt_version="v1")
        session.commit()
        result = validate_analysis(
            session,
            raw_item_id=raw.id,
            student_analysis=student,
            student_model="student",
            analysis_id=rec.id,
            high_confidence_threshold=0.85,
        )
        session.commit()
        assert result.silver_written > 0
        assert result.needs_review is False
        provs = {r.provenance for r in export_dataset(session, tiers=("silver",))}
        assert provs == {"ai_high_conf"}


def test_no_teacher_low_confidence_routes_to_review(tmp_path):
    factory = _setup(tmp_path)
    student = make_analysis().model_copy(update={"confidence": 0.4})
    with factory() as session:
        raw = save_raw_item(session, make_item())
        rec = save_ai_analysis(session, raw.id, student, model_name="student", prompt_version="v1")
        session.commit()
        result = validate_analysis(
            session,
            raw_item_id=raw.id,
            student_analysis=student,
            student_model="student",
            analysis_id=rec.id,
            high_confidence_threshold=0.85,
        )
        session.commit()
        assert result.silver_written == 0
        assert result.needs_review is True
        assert session.get(AIAnalysisRecord, rec.id).needs_human_review is True
