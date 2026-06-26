from tnmi.eval import EvalReport, FieldMetrics
from tnmi.registry import (
    get_live_model,
    primary_metric_from_report,
    promote_if_better,
    register_model,
)
from tnmi.storage import create_session_factory, init_db


def _factory(tmp_path):
    factory = create_session_factory(f"sqlite:///{tmp_path / 'registry.db'}")
    init_db(factory)
    return factory


def test_primary_metric_from_report_averages_macro_f1():
    report = EvalReport(
        total=10,
        overall_accuracy=0.9,
        per_field={
            "a": FieldMetrics("a", 5, 0.8, 0.7, 0.7, 0.6),
            "b": FieldMetrics("b", 5, 0.9, 0.9, 0.9, 0.8),
        },
    )
    assert abs(primary_metric_from_report(report) - 0.7) < 1e-9


def test_first_model_is_promoted_unconditionally(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        register_model(session, model_name="clf", version="v1", primary_metric=0.5)
        decision = promote_if_better(session, model_name="clf", version="v1")
        session.commit()
        assert decision.promoted is True
        assert get_live_model(session, "clf").version == "v1"


def test_better_candidate_replaces_incumbent(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        register_model(session, model_name="clf", version="v1", primary_metric=0.60)
        promote_if_better(session, model_name="clf", version="v1")
        register_model(session, model_name="clf", version="v2", primary_metric=0.72)
        decision = promote_if_better(session, model_name="clf", version="v2")
        session.commit()
        assert decision.promoted is True
        assert get_live_model(session, "clf").version == "v2"


def test_worse_candidate_is_rejected_incumbent_stays_live(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        register_model(session, model_name="clf", version="v1", primary_metric=0.80)
        promote_if_better(session, model_name="clf", version="v1")
        register_model(session, model_name="clf", version="v2", primary_metric=0.75)
        decision = promote_if_better(session, model_name="clf", version="v2")
        session.commit()
        assert decision.promoted is False
        # The incumbent must remain the live model — no silent regression.
        assert get_live_model(session, "clf").version == "v1"


def test_min_delta_blocks_marginal_improvements(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        register_model(session, model_name="clf", version="v1", primary_metric=0.80)
        promote_if_better(session, model_name="clf", version="v1")
        register_model(session, model_name="clf", version="v2", primary_metric=0.805)
        decision = promote_if_better(session, model_name="clf", version="v2", min_delta=0.02)
        session.commit()
        assert decision.promoted is False
        assert get_live_model(session, "clf").version == "v1"
