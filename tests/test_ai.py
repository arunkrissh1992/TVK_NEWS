import pytest

from tnmi.ai import (
    AIAnalysisError,
    MockAIAnalyzer,
    OpenAIAnalyzer,
    build_classification_prompt,
    _extract_political_actors,
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
from tnmi.local_models import LocalTamilAnalyzer


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


def test_mock_ai_analyzer_treats_government_scheme_as_tvk_positive():
    """TVK is the ruling party: a positive government/CM story reflects on TVK,
    so tvk_portrayal is positive (not neutral) and the CM is a TVK office-holder."""
    body = (
        "தமிழக அரசு இன்று புதிய நலத்திட்டத்தை அறிவித்தது. "
        "முதலமைச்சர் இத்திட்டத்தை வரவேற்றுள்ளார். "
        "இத்திட்டம் மக்களுக்கு பெரும் பயன் தருகிறது என்று அரசு கூறுகிறது. "
        "தமிழ்நாட்டின் அனைத்து மாவட்டங்களிலும் இது செயல்படுத்தப்படும்."
    )
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/scheme",
        language="ta",
        title="தமிழக அரசு புதிய நலத்திட்டத்தை அறிவித்துள்ளது",
        raw_text_original=body,
        clean_text_original=body,
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert analysis.tvk_portrayal == Stance.POSITIVE
    assert analysis.tvk_relevance == GovernmentRelevance.HIGH
    assert "Vijay (CM)" in analysis.political_actors


def test_mock_ai_analyzer_labels_dmk_as_opposition_actor():
    """Rival-party leaders must surface under the opposition bucket and must not
    flip tvk_portrayal on their own."""
    body = (
        "திமுக தலைவர் ஸ்டாலின் சென்னையில் ஒரு பொதுக்கூட்டம் நடத்தினார். "
        "எதிர்க்கட்சியைச் சேர்ந்தவர்கள் தமிழக அரசின் கொள்கைகளை விமர்சித்தனர். "
        "இந்நிகழ்வில் கட்சியின் பல மாவட்ட நிர்வாகிகள் கலந்து கொண்டனர். "
        "தமிழ்நாட்டின் பல்வேறு பகுதிகளில் இருந்து தொண்டர்கள் வந்திருந்தனர் என்று கூறப்படுகிறது."
    )
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/dmk",
        language="ta",
        title="திமுக கூட்டம்",
        raw_text_original=body,
        clean_text_original=body,
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert "DMK (opposition)" in analysis.political_actors
    assert "Vijay (CM)" not in analysis.political_actors


def test_actor_extraction_uses_word_boundaries_for_english_tokens():
    actors = _extract_political_actors(
        "AIADMK reacts to train incident in Vijayawada Division",
        (
            "The report mentioned AIADMK leaders and a rail incident near Andhra Pradesh; "
            "it did not name any ruling-party actor."
        ),
    )

    assert "AIADMK" in actors
    assert "DMK (opposition)" not in actors
    assert "Vijay (CM)" not in actors


def test_mock_ai_analyzer_flags_school_fire_as_people_issue_with_action():
    body = (
        "சென்னை பள்ளிக்கரணை பகுதியில் உள்ள மாநகராட்சி குப்பைக்கிடங்கில் தீ விபத்து ஏற்பட்டது. "
        "அருகில் உள்ள பள்ளி மாணவர்கள் பாதுகாப்பாக வெளியேற்றப்பட்டனர். "
        "தீயணைப்பு வீரர்கள் சம்பவ இடத்தில் பணியில் ஈடுபட்டனர். "
        "தமிழகத்தில் உள்ளூர் பாதுகாப்பு நடவடிக்கைகள் குறித்து பெற்றோர் கவலை தெரிவித்தனர்."
    )
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Polimer News",
        source_url="https://example.com/school-fire",
        language="ta",
        title="சென்னை பள்ளி அருகே தீ விபத்து",
        raw_text_original=body,
        clean_text_original=body,
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert analysis.tvk_portrayal == Stance.NEUTRAL
    assert analysis.people_issue is True
    assert analysis.public_issue == "school safety incident"
    assert analysis.severity == Severity.HIGH
    assert analysis.action_owner == "District field team"
    assert analysis.action_type == "field_verification"
    assert "சென்னை பள்ளி அருகே தீ விபத்து" in analysis.recommended_step
    assert "student safety" in analysis.recommended_step
    assert "verify" in analysis.recommended_step.lower()
    assert "appropriate department" not in analysis.recommended_step.lower()


def test_mock_ai_analyzer_does_not_treat_pallikaranai_as_school():
    body = (
        "சென்னை பள்ளிக்கரணை பகுதியில் உள்ள மாநகராட்சி குப்பைக்கிடங்கில் தீ விபத்து ஏற்பட்டது. "
        "அருகில் வசிக்கும் மக்கள் புகை காரணமாக அவதியடைந்தனர். "
        "தீயணைப்பு வீரர்கள் சம்பவ இடத்தில் பணியில் ஈடுபட்டனர். "
        "தமிழகத்தில் உள்ளூர் பாதுகாப்பு நடவடிக்கைகள் குறித்து மக்கள் கவலை தெரிவித்தனர்."
    )
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Polimer News",
        source_url="https://example.com/pallikaranai-fire",
        language="ta",
        title="சென்னை பள்ளிக்கரணை குப்பைக்கிடங்கில் பயங்கர தீ விபத்து",
        raw_text_original=body,
        clean_text_original=body,
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert analysis.people_issue is True
    assert analysis.public_issue == "public safety incident"
    assert analysis.issue_category == "public_safety"
    assert analysis.needs_human_review is True
    assert "பள்ளிக்கரணை" in analysis.recommended_step
    assert "casualties" in analysis.recommended_step


def test_mock_ai_analyzer_builds_action_playbook_for_people_issue():
    """Negative / people-issue rows must carry a ready-to-act playbook: risk,
    verification checklist, talking points, and a draft statement."""
    body = (
        "சென்னை பள்ளிக்கரணை பகுதியில் உள்ள மாநகராட்சி குப்பைக்கிடங்கில் தீ விபத்து ஏற்பட்டது. "
        "அருகில் உள்ள பள்ளி மாணவர்கள் பாதுகாப்பாக வெளியேற்றப்பட்டனர். "
        "தீயணைப்பு வீரர்கள் சம்பவ இடத்தில் பணியில் ஈடுபட்டனர். "
        "தமிழகத்தில் உள்ளூர் பாதுகாப்பு நடவடிக்கைகள் குறித்து பெற்றோர் கவலை தெரிவித்தனர்."
    )
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Polimer News",
        source_url="https://example.com/school-fire-playbook",
        language="ta",
        title="சென்னை பள்ளி அருகே தீ விபத்து",
        raw_text_original=body,
        clean_text_original=body,
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert analysis.people_issue is True
    assert analysis.risk_if_ignored != ""
    assert len(analysis.verification_checklist) >= 2
    assert analysis.draft_statement_original != ""


def test_mock_ai_analyzer_leaves_playbook_empty_for_positive_item():
    """Positive party news needs amplification, not a defensive playbook —
    the playbook fields must stay empty."""
    body = (
        "தலைவர் விஜய் தலைமையிலான தவெக அரசு புதிய மகளிர் நலத் திட்டத்தை அறிமுகப்படுத்தியது. "
        "இத்திட்டத்தை பொதுமக்கள் வரவேற்றனர் என்று செய்திகள் தெரிவிக்கின்றன. "
        "தமிழ்நாட்டின் பல மாவட்டங்களில் இத்திட்டம் செயல்படுத்தப்படும். "
        "முதலமைச்சர் இது குறித்து சட்டமன்றத்தில் அறிவித்தார்."
    )
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/positive-scheme",
        language="ta",
        title="தவெக மகளிர் நலத் திட்டம்",
        raw_text_original=body,
        clean_text_original=body,
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert analysis.tvk_portrayal == Stance.POSITIVE
    assert analysis.risk_if_ignored == ""
    assert analysis.talking_points == []
    assert analysis.verification_checklist == []
    assert analysis.draft_statement_english == ""
    assert analysis.draft_statement_original == ""


def test_build_classification_prompt_documents_action_playbook():
    prompt = build_classification_prompt(_normalized_item())

    assert "ACTION PLAYBOOK" in prompt
    assert '"risk_if_ignored": "string"' in prompt
    assert '"talking_points": ["string"]' in prompt
    assert '"verification_checklist": ["string"]' in prompt
    assert '"draft_statement_english": "string"' in prompt


def test_local_tamil_analyzer_does_not_treat_vijayawada_as_vijay():
    body = (
        "The All India Loco Running Staff Association has expressed shock over the incident "
        "reported in Vijayawada Division where the lookout glass of an express train broke. "
        "Train No 12616 New Delhi-Chennai Central came to a halt after the Assistant Locopilot "
        "suffered a bleeding injury. The incident needs railway safety follow-up in Tamil Nadu."
    )
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="The Hindu (Chennai)",
        source_url="https://example.com/train-safety",
        language="en",
        title="Lookout glass of train breaks after hit by food packet",
        raw_text_original=body,
        clean_text_original=body,
    )

    analysis = LocalTamilAnalyzer().analyze(item)

    assert analysis.tvk_relevance != GovernmentRelevance.HIGH
    assert analysis.tvk_portrayal == Stance.NEUTRAL
    assert analysis.party_action == ""
    assert analysis.target != "TVK leadership"


def test_build_classification_prompt_documents_scheme_contract():
    prompt = build_classification_prompt(_normalized_item())

    assert '"scheme": "string|null"' in prompt
    assert "Populate the scheme name only when explicitly mentioned; otherwise null." in prompt


def test_build_classification_prompt_frames_tvk_office_holders_and_two_axes():
    """The briefing lens must (1) count TVK office-holders (MLA/minister/CM) as
    TVK, (2) exclude rival parties from tvk_portrayal, and (3) keep
    tvk_portrayal and stance_toward_government as independent axes with
    tvk_portrayal as the headline."""
    prompt = build_classification_prompt(_normalized_item())

    # TVK roster includes its own office-holders.
    assert "office-holder" in prompt
    assert "Chief Minister" in prompt
    # Rival parties never drive tvk_portrayal.
    assert "are NOT TVK" in prompt
    # Two independent axes, with tvk_portrayal as the headline label.
    assert "Two independent judgement axes" in prompt
    assert "tvk_portrayal is the HEADLINE label" in prompt


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


def test_tn_gate_rejects_international_story_with_substring_traps():
    """'theni' in "strengthening" / 'salem' in "Jerusalem" must NOT pass the
    Tamil Nadu gate — this exact bug put a Pakistan/US-Iran wire story on the
    leadership dashboard as a TN people-issue."""
    body = (
        "Pakistan has accelerated efforts at strengthening ties with Iran. "
        "Islamabad seeks to sustain the ceasefire and restart dialogue with "
        "officials in Jerusalem and Tehran. The minister delivered a message "
        "from the army chief to the Supreme Leader on Friday, officials said."
    )
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Wire",
        source_url="https://example.com/intl",
        language="en",
        title="Pakistan tries to revive US-Iran talks",
        raw_text_original=body,
        clean_text_original=body,
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert analysis.government_relevance == GovernmentRelevance.NONE
    assert analysis.tvk_relevance == GovernmentRelevance.NONE
    assert analysis.people_issue is False

    local = LocalTamilAnalyzer().analyze(item)
    assert local.government_relevance == GovernmentRelevance.NONE


def test_mock_analyzer_extracts_district_from_text():
    body = (
        "மதுரை மாவட்டத்தில் குடிநீர் தட்டுப்பாடு குறித்து பொதுமக்கள் புகார் தெரிவித்தனர். "
        "தமிழக அரசு நடவடிக்கை எடுக்க வேண்டும் என்று கோரிக்கை வைத்தனர். "
        "மாநகராட்சி நிர்வாகம் தண்ணீர் விநியோகத்தை சீரமைக்கவில்லை என்றனர். "
        "வரும் வாரத்தில் போராட்டம் நடத்தப்படும் என அறிவிக்கப்பட்டுள்ளது."
    )
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/madurai-water",
        language="ta",
        title="மதுரையில் குடிநீர் தட்டுப்பாடு",
        raw_text_original=body,
        clean_text_original=body,
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert analysis.district == "Madurai"


def test_prompt_lens_is_per_tenant_subject():
    """The classifier prompt is the only per-tenant variable: swap the subject
    party and governing flag and the lens reframes — same engine, any party."""
    item = _normalized_item()
    # Default = TVK governing (unchanged behaviour).
    tvk = build_classification_prompt(item)
    assert "are NOT TVK" in tvk
    assert "When TVK runs the government" in tvk

    # An opposition tenant: positive flips to the subject's gain from govt failure.
    dmk = build_classification_prompt(item, subject="DMK", leader="Stalin", governing=False)
    assert "DMK party leadership office" in dmk
    assert "are NOT DMK" in dmk
    assert "in OPPOSITION" in dmk
    assert "DMK can capitalise on" in dmk
    assert "TVK" not in dmk  # no leakage of the default subject
