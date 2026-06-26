from tnmi.ai import MockAIAnalyzer
from tnmi.contracts import LabelProvenance, LabelTier, NormalizedItem, SourceType
from tnmi.eval import evaluate_classifier, extract_field_value, score_predictions
from tnmi.labeling import record_label
from tnmi.storage import create_session_factory, init_db, save_raw_item


def test_score_predictions_computes_per_field_accuracy():
    gold = [
        (1, "tvk_portrayal", "positive"),
        (2, "tvk_portrayal", "negative"),
        (3, "tvk_portrayal", "negative"),
    ]
    predictions = {
        (1, "tvk_portrayal"): "positive",  # correct
        (2, "tvk_portrayal"): "negative",  # correct
        (3, "tvk_portrayal"): "positive",  # wrong
    }
    report = score_predictions(gold, predictions)
    assert report.total == 3
    assert abs(report.overall_accuracy - 2 / 3) < 1e-9
    fm = report.per_field["tvk_portrayal"]
    assert fm.support == 3
    assert abs(fm.accuracy - 2 / 3) < 1e-9
    # macro F1 is between 0 and 1 and reflects the imperfect score.
    assert 0.0 < fm.macro_f1 < 1.0


def test_score_predictions_counts_missing_as_wrong():
    gold = [(1, "people_issue", "true")]
    report = score_predictions(gold, {})  # no prediction supplied
    assert report.per_field["people_issue"].accuracy == 0.0


def test_extract_field_value_stringifies_bool_and_enum():
    analysis = MockAIAnalyzer().analyze(
        NormalizedItem(
            source_type=SourceType.NEWS,
            source_name="X",
            source_url="https://e.com/a",
            language="ta",
            title="t",
            raw_text_original="b",
            clean_text_original="b",
        )
    )
    assert extract_field_value(analysis, "people_issue") in {"true", "false"}
    assert extract_field_value(analysis, "government_relevance") in {"high", "medium", "low", "none"}


def _scheme_item() -> NormalizedItem:
    body = (
        "தமிழக அரசு இன்று புதிய நலத்திட்டத்தை அறிவித்தது. "
        "முதலமைச்சர் இத்திட்டத்தை வரவேற்றுள்ளார். "
        "இத்திட்டம் மக்களுக்கு பெரும் பயன் தருகிறது என்று அரசு கூறுகிறது. "
        "தமிழ்நாட்டின் அனைத்து மாவட்டங்களிலும் இது செயல்படுத்தப்படும்."
    )
    return NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/scheme",
        language="ta",
        title="தமிழக அரசு புதிய நலத்திட்டத்தை அறிவித்துள்ளது",
        raw_text_original=body,
        clean_text_original=body,
    )


def test_evaluate_classifier_against_mock(tmp_path):
    factory = create_session_factory(f"sqlite:///{tmp_path / 'eval.db'}")
    init_db(factory)
    with factory() as session:
        raw = save_raw_item(session, _scheme_item())
        session.commit()
        # One gold label the mock should get right, one it should get wrong.
        record_label(
            session,
            raw_item_id=raw.id,
            field="tvk_portrayal",
            value="positive",
            tier=LabelTier.GOLD,
            provenance=LabelProvenance.HUMAN,
        )
        record_label(
            session,
            raw_item_id=raw.id,
            field="government_relevance",
            value="none",  # mock will say "high" → mismatch
            tier=LabelTier.GOLD,
            provenance=LabelProvenance.HUMAN,
        )
        session.commit()

        report = evaluate_classifier(session, MockAIAnalyzer(), held_out=range(0, 100))

    assert report.per_field["tvk_portrayal"].accuracy == 1.0
    assert report.per_field["government_relevance"].accuracy == 0.0
