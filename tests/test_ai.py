from tnmi.ai import MockAIAnalyzer
from tnmi.contracts import GovernmentRelevance, NormalizedItem, SourceType, Stance


def test_mock_ai_analyzer_returns_positive_for_scheme_news():
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/a",
        language="ta",
        title="தமிழக அரசு புதிய திட்டம்",
        raw_text_original="தமிழக அரசு இன்று புதிய நலத்திட்டத்தை அறிவித்தது.",
        clean_text_original="தமிழக அரசு இன்று புதிய நலத்திட்டத்தை அறிவித்தது.",
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert analysis.government_relevance == GovernmentRelevance.HIGH
    assert analysis.stance_toward_government == Stance.POSITIVE
    assert analysis.confidence >= 0.5
