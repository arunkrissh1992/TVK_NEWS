import json

from tnmi.contracts import LABEL_FIELDS, LabelProvenance, LabelTier
from tnmi.eval import HELD_OUT_TEST_BUCKETS
from tnmi.labeling import record_label, split_bucket
from tnmi.storage import create_session_factory, init_db, save_raw_item
from tnmi.training import (
    StubTrainer,
    build_distillation_dataset,
    dataset_fingerprint,
    export_jsonl,
)

from tests.test_storage import make_item


def _factory(tmp_path):
    factory = create_session_factory(f"sqlite:///{tmp_path / 'train.db'}")
    init_db(factory)
    return factory


def _label_all_fields(session, raw_id, tier=LabelTier.SILVER, provenance=LabelProvenance.AI_HIGH_CONF):
    for fld in LABEL_FIELDS:
        record_label(
            session,
            raw_item_id=raw_id,
            field=fld,
            value="neutral" if "portrayal" in fld or "stance" in fld else "low",
            tier=tier,
            provenance=provenance,
        )


def test_build_dataset_excludes_held_out_test_buckets(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        ids = []
        for i in range(3):
            raw = save_raw_item(
                session, make_item().model_copy(update={"source_url": f"https://e.com/{i}"})
            )
            session.commit()
            _label_all_fields(session, raw.id)
            ids.append(raw.id)
        session.commit()

        held = build_distillation_dataset(session, exclude_held_out=True)
        full = build_distillation_dataset(session, exclude_held_out=False)

    held_pairs = {(ex.raw_item_id, fld) for ex in held for fld in ex.labels}
    expected_kept = {
        (rid, fld)
        for rid in ids
        for fld in LABEL_FIELDS
        if split_bucket(rid, fld) not in set(HELD_OUT_TEST_BUCKETS)
    }
    assert held_pairs == expected_kept
    # The full dataset keeps everything; held-out is a strict subset here.
    full_pairs = {(ex.raw_item_id, fld) for ex in full for fld in ex.labels}
    assert held_pairs <= full_pairs
    assert len(held_pairs) < len(full_pairs)


def test_build_dataset_groups_labels_per_item(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, make_item())
        session.commit()
        record_label(
            session, raw_item_id=raw.id, field="tvk_portrayal", value="negative",
            tier=LabelTier.GOLD, provenance=LabelProvenance.HUMAN,
        )
        record_label(
            session, raw_item_id=raw.id, field="people_issue", value="true",
            tier=LabelTier.GOLD, provenance=LabelProvenance.HUMAN,
        )
        session.commit()
        examples = build_distillation_dataset(session, exclude_held_out=False)

    assert len(examples) == 1
    assert examples[0].labels["tvk_portrayal"] == "negative"
    assert examples[0].labels["people_issue"] == "true"
    assert examples[0].text  # text joined from the raw item


def test_export_jsonl_writes_valid_lines(tmp_path):
    factory = _factory(tmp_path)
    out = tmp_path / "ds" / "train.jsonl"
    with factory() as session:
        raw = save_raw_item(session, make_item())
        session.commit()
        record_label(
            session, raw_item_id=raw.id, field="tvk_portrayal", value="negative",
            tier=LabelTier.GOLD, provenance=LabelProvenance.HUMAN,
        )
        session.commit()
        examples = build_distillation_dataset(session, exclude_held_out=False)
        n = export_jsonl(examples, out)

    assert n == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["labels"]["tvk_portrayal"] == "negative"
    assert "text" in row


def test_dataset_fingerprint_is_deterministic_and_label_sensitive(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, make_item())
        session.commit()
        record_label(
            session, raw_item_id=raw.id, field="tvk_portrayal", value="negative",
            tier=LabelTier.GOLD, provenance=LabelProvenance.HUMAN,
        )
        session.commit()
        examples = build_distillation_dataset(session, exclude_held_out=False)
        fp1 = dataset_fingerprint(examples)
        fp2 = dataset_fingerprint(examples)
        assert fp1 == fp2
        # Changing a label changes the fingerprint.
        examples[0].labels["tvk_portrayal"] = "positive"
        assert dataset_fingerprint(examples) != fp1


def test_stub_trainer_produces_deterministic_version(tmp_path):
    factory = _factory(tmp_path)
    with factory() as session:
        raw = save_raw_item(session, make_item())
        session.commit()
        record_label(
            session, raw_item_id=raw.id, field="tvk_portrayal", value="negative",
            tier=LabelTier.GOLD, provenance=LabelProvenance.HUMAN,
        )
        session.commit()
        examples = build_distillation_dataset(session, exclude_held_out=False)

    result = StubTrainer().train(examples, model_name="tvk-clf", output_dir=tmp_path / "artifacts")
    assert result.version == dataset_fingerprint(examples)
    assert result.num_examples == 1
    assert result.version in result.artifact_uri
    assert result.metadata["label_counts"]["tvk_portrayal"] == 1
