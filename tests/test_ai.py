import pytest

from tnmi.ai import (
    AIAnalysisError,
    MockAIAnalyzer,
    OpenAIAnalyzer,
    build_classification_prompt,
)
from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
    Sentiment,
    Severity,
    SourceType,
    Stance,
)


def test_mock_ai_analyzer_returns_positive_for_scheme_news():
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/a",
        language="ta",
        title="தமிழக அரசு புதிய நலத்திட்டத்தை அறிவித்துள்ளது",
        # Body length is intentionally above the listing-page threshold so the
        # mock analyzer does not treat this as an RSS shell.
        raw_text_original=(
            "தமிழக அரசு இன்று புதிய நலத்திட்டத்தை அறிவித்தது. "
            "முதலமைச்சர் இத்திட்டத்தை வரவேற்றுள்ளார். "
            "இத்திட்டம் மக்களுக்கு பெரும் பயன் தருகிறது என்று அரசு கூறுகிறது. "
            "தமிழ்நாட்டின் அனைத்து மாவட்டங்களிலும் இது செயல்படுத்தப்படும் என்று அரசு தெரிவித்துள்ளது."
        ),
        clean_text_original=(
            "தமிழக அரசு இன்று புதிய நலத்திட்டத்தை அறிவித்தது. "
            "முதலமைச்சர் இத்திட்டத்தை வரவேற்றுள்ளார். "
            "இத்திட்டம் மக்களுக்கு பெரும் பயன் தருகிறது என்று அரசு கூறுகிறது. "
            "தமிழ்நாட்டின் அனைத்து மாவட்டங்களிலும் இது செயல்படுத்தப்படும் என்று அரசு தெரிவித்துள்ளது."
        ),
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert analysis.government_relevance == GovernmentRelevance.HIGH
    assert analysis.stance_toward_government == Stance.POSITIVE
    assert analysis.confidence >= 0.5


def test_build_classification_prompt_documents_scheme_contract():
    prompt = build_classification_prompt(_normalized_item())

    assert '"scheme": "string|null"' in prompt
    assert "Populate the scheme name only when explicitly mentioned; otherwise null." in prompt


def test_openai_analyzer_uses_structured_output_parse():
    expected = _analysis(scheme="Kalaignar Magalir Urimai Thogai")
    client = _FakeClient(output_parsed=expected)
    analyzer = _openai_analyzer(client)

    analysis = analyzer.analyze(_normalized_item())

    assert analysis == expected
    assert client.responses.parse_kwargs["model"] == "fake-model"
    assert client.responses.parse_kwargs["text_format"] is AIAnalysis
    assert isinstance(client.responses.parse_kwargs["input"], list)
    assert "Tamil Nadu Government" in str(client.responses.parse_kwargs["input"])
    assert "Kalaignar Magalir Urimai Thogai" in str(client.responses.parse_kwargs["input"])


def test_openai_analyzer_validates_dict_output_parsed():
    client = _FakeClient(output_parsed=_analysis().model_dump(mode="json"))
    analyzer = _openai_analyzer(client)

    analysis = analyzer.analyze(_normalized_item())

    assert isinstance(analysis, AIAnalysis)
    assert analysis.government_relevance == GovernmentRelevance.HIGH


def test_openai_analyzer_raises_when_parsed_output_missing():
    client = _FakeClient(
        output_parsed=None,
        refusal="Cannot classify the supplied item.",
        incomplete_reason="max_output_tokens",
    )
    analyzer = _openai_analyzer(client)

    with pytest.raises(AIAnalysisError) as exc_info:
        analyzer.analyze(_normalized_item())

    message = str(exc_info.value)
    assert "parsed output" in message
    assert "Cannot classify" in message
    assert "max_output_tokens" in message


def _normalized_item() -> NormalizedItem:
    return NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/a",
        language="ta",
        title="Kalaignar Magalir Urimai Thogai scheme expanded",
        raw_text_original=(
            "Tamil Nadu Government announced updates to Kalaignar Magalir Urimai Thogai."
        ),
        clean_text_original=(
            "Tamil Nadu Government announced updates to Kalaignar Magalir Urimai Thogai."
        ),
    )


def _analysis(scheme: str | None = None) -> AIAnalysis:
    return AIAnalysis(
        government_relevance=GovernmentRelevance.HIGH,
        stance_toward_government=Stance.POSITIVE,
        sentiment=Sentiment.POSITIVE,
        target="Tamil Nadu Government",
        department="unknown",
        district="unknown",
        scheme=scheme,
        topic="scheme update",
        issue_category="welfare",
        severity=Severity.LOW,
        summary_original="Scheme update.",
        summary_english="Scheme update.",
        positive_points=["Mentions a government scheme."],
        negative_points=[],
        evidence_quotes_original=["Tamil Nadu Government announced updates."],
        evidence_quotes_english=["Tamil Nadu Government announced updates."],
        confidence=0.9,
        needs_human_review=False,
    )


def _openai_analyzer(client: "_FakeClient") -> OpenAIAnalyzer:
    analyzer = OpenAIAnalyzer.__new__(OpenAIAnalyzer)
    analyzer.client = client
    analyzer.model_name = "fake-model"
    return analyzer


class _FakeClient:
    def __init__(
        self,
        output_parsed: AIAnalysis | dict | None,
        refusal: str | None = None,
        incomplete_reason: str | None = None,
    ) -> None:
        self.responses = _FakeResponses(output_parsed, refusal, incomplete_reason)


class _FakeResponses:
    def __init__(
        self,
        output_parsed: AIAnalysis | dict | None,
        refusal: str | None,
        incomplete_reason: str | None,
    ) -> None:
        self.output_parsed = output_parsed
        self.refusal = refusal
        self.incomplete_reason = incomplete_reason
        self.parse_kwargs: dict | None = None

    def parse(self, **kwargs):
        self.parse_kwargs = kwargs
        return _FakeResponse(self.output_parsed, self.refusal, self.incomplete_reason)


class _FakeResponse:
    def __init__(
        self,
        output_parsed: AIAnalysis | dict | None,
        refusal: str | None,
        incomplete_reason: str | None,
    ) -> None:
        self.output_parsed = output_parsed
        self.incomplete_details = _FakeIncompleteDetails(incomplete_reason)
        self.output = [_FakeOutput([_FakeContent(refusal)])]


class _FakeIncompleteDetails:
    def __init__(self, reason: str | None) -> None:
        self.reason = reason


class _FakeOutput:
    def __init__(self, content: list["_FakeContent"]) -> None:
        self.content = content


class _FakeContent:
    def __init__(self, refusal: str | None) -> None:
        self.refusal = refusal
