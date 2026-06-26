from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
    ReviewDecisionCreate,
    ReviewStatus,
    Stance,
)
from tnmi.flywheel import harvest_labels, run_flywheel, train_and_gate
from tnmi.labeling import export_dataset
from tnmi.registry import get_live_model
from tnmi.storage import (
    create_session_factory,
    init_db,
    save_ai_analysis,
    save_raw_item,
    save_review_decision,
)
from tnmi.training import StubTrainer

from tests.test_storage import make_analysis, make_item


class FixedAnalyzer:
    def __init__(self, analysis: AIAnalysis, model_name: str = "fixed"):
        self._analysis = analysis
        self.model_name = model_name

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        return self._analysis


def _factory(tmp_path):
    factory = create_session_factory(f"sqlite:///{tmp_path / 'flywheel.db'}")
    init_db(factory)
    return factory


def _seed(session, *, n=2, confidence=0.95):
    """n raw items, each with a high-confidence analysis."""
    records = []
    for i in range(n):
        raw = save_raw_item(
            session, make_item().model_copy(update={"source_url": f"https://e.com/{i}"})
        )
        rec = save_ai_analysis(
            session,
            raw.id,
            make_analysis().model_copy(update={"confidence": confidence}),
            model_name="mock",
            prompt_version="v1",
        )
        records.append((raw, rec))
    session.commit()
    return records


def test_harvest_labels_writes_bronze_silver_and_gold(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        records = _seed(session, n=2)
        # One human correction → should become gold.
        save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=records[0][1].id,
                reviewer_name="analyst-1",
                status=ReviewStatus.CORRECTED,
                corrected_stance=Stance.NEGATIVE,
                corrected_relevance=GovernmentRelevance.HIGH,
            ),
        )
        session.commit()

        report = harvest_labels(session)
        session.commit()

        assert report.gold_promoted == 2  # stance + relevance corrections
        assert report.bronze_written > 0
        assert report.items_validated == 2
        assert report.silver_written > 0  # confidence 0.95 ≥ 0.85 threshold
        assert report.routed_to_review == 0
        gold = export_dataset(session, tiers=("gold",))
        assert {r.field for r in gold} == {"tvk_portrayal", "government_relevance"}


def test_train_and_gate_does_not_promote_without_gold_yardstick(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        _seed(session, n=2)
        harvest_labels(session)  # silver only — no human corrections
        session.commit()

        result, report = train_and_gate(
            session,
            trainer=StubTrainer(),
            candidate_analyzer=FixedAnalyzer(make_analysis()),
            output_dir=str(tmp_path / "artifacts"),
        )
        session.commit()

        assert result is not None
        assert report.training_examples > 0
        assert report.eval_total == 0
        assert report.promotion is not None
        assert report.promotion.promoted is False
        assert "no gold test labels" in report.promotion.reason
        # Registered but NOT live.
        assert get_live_model(session, "tvk-tamil-classifier") is None


def test_run_flywheel_full_pass_is_idempotent(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        records = _seed(session, n=3)
        save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=records[0][1].id,
                reviewer_name="analyst-1",
                status=ReviewStatus.CORRECTED,
                corrected_stance=Stance.NEGATIVE,
            ),
        )
        session.commit()

        first = run_flywheel(
            session,
            trainer=StubTrainer(),
            candidate_analyzer=FixedAnalyzer(make_analysis()),
            output_dir=str(tmp_path / "artifacts"),
        )
        session.commit()
        second = run_flywheel(
            session,
            trainer=StubTrainer(),
            candidate_analyzer=FixedAnalyzer(make_analysis()),
            output_dir=str(tmp_path / "artifacts"),
        )
        session.commit()

    # Same data → same trained version (dataset fingerprint), no label blowup.
    assert first.trained_version == second.trained_version
    assert first.label_stats["total_labels"] == second.label_stats["total_labels"]
    assert first.training_examples == second.training_examples


def test_model_health_reports_labels_and_live_model(tmp_path):
    from tnmi.flywheel import model_health
    from tnmi.registry import promote_if_better, register_model

    factory = _factory(tmp_path)
    with factory() as session:
        records = _seed(session, n=2)
        save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=records[0][1].id,
                reviewer_name="analyst-1",
                status=ReviewStatus.CORRECTED,
                corrected_stance=Stance.NEGATIVE,
            ),
        )
        session.commit()

        # Before any flywheel pass: nothing learned yet.
        before = model_health(session)
        assert before["has_run"] is False
        assert before["live_version"] == ""

        harvest_labels(session)
        register_model(session, model_name="tvk-tamil-classifier", version="v-test", primary_metric=0.8)
        promote_if_better(session, model_name="tvk-tamil-classifier", version="v-test")
        session.commit()

        after = model_health(session)
        assert after["has_run"] is True
        assert after["gold"] >= 1  # the human correction
        assert after["bronze"] >= 1
        assert after["live_version"] == "v-test"
        assert after["live_metric"] == 0.8
