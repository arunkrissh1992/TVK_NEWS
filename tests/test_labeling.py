from tnmi.contracts import (
    GovernmentRelevance,
    LabelProvenance,
    LabelTier,
    ReviewDecisionCreate,
    ReviewStatus,
    Stance,
)
from tnmi.labeling import (
    dataset_stats,
    export_dataset,
    promote_corrections_to_gold,
    record_bronze_from_analysis,
    record_label,
    split_bucket,
)
from tnmi.storage import (
    create_session_factory,
    init_db,
    save_ai_analysis,
    save_raw_item,
    save_review_decision,
)

from tests.test_storage import make_analysis, make_item


def _factory(tmp_path):
    factory = create_session_factory(f"sqlite:///{tmp_path / 'labels.db'}")
    init_db(factory)
    return factory


def test_split_bucket_is_deterministic_and_bounded():
    a = split_bucket(123, "tvk_portrayal")
    b = split_bucket(123, "tvk_portrayal")
    assert a == b
    assert 0 <= a < 100
    # Different field → (almost surely) different bucket, never out of range.
    assert 0 <= split_bucket(123, "people_issue") < 100


def test_record_label_upserts_same_item_field_tier(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, make_item())
        session.commit()
        first = record_label(
            session,
            raw_item_id=raw.id,
            field="tvk_portrayal",
            value="neutral",
            tier=LabelTier.BRONZE,
            provenance=LabelProvenance.AI,
        )
        second = record_label(
            session,
            raw_item_id=raw.id,
            field="tvk_portrayal",
            value="negative",
            tier=LabelTier.BRONZE,
            provenance=LabelProvenance.AI,
        )
        session.commit()
        assert first.id == second.id  # upsert, not a new row
        assert second.value == "negative"


def test_record_bronze_from_analysis_writes_core_fields(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, make_item())
        analysis = save_ai_analysis(
            session, raw.id, make_analysis(), model_name="mock", prompt_version="v1"
        )
        session.commit()
        labels = record_bronze_from_analysis(session, analysis)
        session.commit()
        by_field = {label.field: label.value for label in labels}
        assert by_field["tvk_portrayal"] == "positive"
        assert by_field["people_issue"] in {"true", "false"}
        assert by_field["government_relevance"] == "high"
        # Bronze provenance + validator captured.
        assert all(label.tier == "bronze" for label in labels)
        assert all(label.provenance == "ai" for label in labels)


def test_promote_corrections_to_gold(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, make_item())
        analysis = save_ai_analysis(
            session, raw.id, make_analysis(), model_name="mock", prompt_version="v1"
        )
        session.commit()
        save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=analysis.id,
                reviewer_name="analyst-1",
                status=ReviewStatus.CORRECTED,
                corrected_stance=Stance.NEGATIVE,
                corrected_relevance=GovernmentRelevance.HIGH,
            ),
        )
        session.commit()
        count = promote_corrections_to_gold(session)
        session.commit()
        assert count == 2
        rows = export_dataset(session, tiers=(LabelTier.GOLD.value,))
        by_field = {r.field: r for r in rows}
        assert by_field["tvk_portrayal"].value == "negative"
        assert by_field["tvk_portrayal"].provenance == "human"
        assert by_field["government_relevance"].value == "high"


def test_export_dataset_prefers_gold_over_bronze(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, make_item())
        session.commit()
        record_label(
            session,
            raw_item_id=raw.id,
            field="tvk_portrayal",
            value="neutral",
            tier=LabelTier.BRONZE,
            provenance=LabelProvenance.AI,
        )
        record_label(
            session,
            raw_item_id=raw.id,
            field="tvk_portrayal",
            value="negative",
            tier=LabelTier.GOLD,
            provenance=LabelProvenance.HUMAN,
        )
        session.commit()
        rows = export_dataset(session, tiers=(LabelTier.BRONZE.value, LabelTier.GOLD.value))
        assert len(rows) == 1
        assert rows[0].value == "negative"
        assert rows[0].tier == "gold"
        assert rows[0].text  # raw text joined in


def test_export_dataset_can_exclude_held_out_buckets(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, make_item())
        session.commit()
        label = record_label(
            session,
            raw_item_id=raw.id,
            field="tvk_portrayal",
            value="negative",
            tier=LabelTier.GOLD,
            provenance=LabelProvenance.HUMAN,
        )
        session.commit()
        held = range(label.split_bucket, label.split_bucket + 1)
        rows = export_dataset(
            session, tiers=(LabelTier.GOLD.value,), exclude_split_buckets=held
        )
        assert rows == []


def test_dataset_stats_counts_by_tier_and_provenance(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, make_item())
        analysis = save_ai_analysis(
            session, raw.id, make_analysis(), model_name="mock", prompt_version="v1"
        )
        session.commit()
        record_bronze_from_analysis(session, analysis)
        session.commit()
        stats = dataset_stats(session)
        assert stats["total_labels"] >= 1
        assert stats["by_tier"]["bronze"] >= 1
        assert "ai" in stats["by_provenance"]
