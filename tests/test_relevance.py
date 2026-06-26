"""Accuracy tests for the relevance/scope gate and people-issue detection.

These pin the behaviour the dashboard depends on: international sport, foreign
affairs and film gossip must NOT enter a Tamil Nadu government briefing, and a
"people issue" must be a genuine public-service grievance — not merely a
negative tone or a single common word like "water" or "death".
"""

import pytest

from tnmi.ai import (
    MockAIAnalyzer,
    _detect_people_issue,
    _is_hard_out_of_scope,
    _is_out_of_scope,
    _looks_like_tn_content,
)
from tnmi.contracts import GovernmentRelevance, NormalizedItem, SourceType
from tnmi.local_models import LocalTamilAnalyzer


# --- scope gate ------------------------------------------------------------

OUT_OF_SCOPE = [
    ("🏆 FIFA World Cup 2026: அமெரிக்கா அபார வெற்றி", "கால்பந்து உலகக் கோப்பை போட்டியில் அமெரிக்கா வென்றது.", "https://p.com/sportsnews/a"),
    ("The 4 Davids of FIFA WC 2026 take on the Goliaths", "FIFA World Cup 2026 underdogs preview ahead of the tournament kickoff.", "https://p.com/sportsnews/b"),
    ("Women's T20 World Cup 2026: England beat Sri Lanka", "A thrilling T20 World Cup contest decided in the final over.", "https://p.com/sportsnews/c"),
    ("தாய்லாந்து இளவரசி மரணம்", "மூன்று ஆண்டுகளாக கோமா நிலையில் இருந்த தாய்லாந்து இளவரசி உயிரிழந்தார்.", "https://p.com/worldnews/d"),
    ("பாகிஸ்தான் பாதுகாப்பு படை சுட்டதில் 30 பேர் பலி", "பாகிஸ்தானில் நடந்த துப்பாக்கிச் சூட்டில் 30 பேர் இறந்தனர் என தகவல்.", "https://p.com/worldnews/e"),
    ("PM Modi embarks on a visit to France and Slovakia", "The Prime Minister begins a multi-nation foreign tour this week.", "https://p.com/indianews/f"),
    ("Chennai Super Kings win IPL final", "CSK lifted the IPL trophy in a last-ball thriller at the stadium.", "https://p.com/sportsnews/g"),
    # Other Indian states are out of scope for a TAMIL NADU briefing.
    ("கேரளாவில் சிறுமி உயிரிழந்தார்; 15 ஆண்டுகளுக்குப் பிறகு குற்றவாளிகள் விடுதலை", "கேரள மாநிலத்தில் நடந்த சம்பவத்தில் சிறுமி உயிரிழந்தார்.", "https://p.com/news/k1"),
    ("Jewellery shop owner arrested in Kerala for fraud", "A jeweller in Kochi, Kerala was arrested for defrauding customers.", "https://p.com/news/k2"),
    ("பெங்களூருவில் கட்டிடம் இடிந்து விபத்து", "கர்நாடக மாநிலம் பெங்களூருவில் கட்டிடம் இடிந்தது.", "https://p.com/news/k3"),
]

IN_SCOPE = [
    ("முதலமைச்சர் விஜய் கொள்ளூர் கோவிலில் வழிபாடு", "தமிழக முதலமைச்சர் விஜய் கோவிலில் சிறப்பு வழிபாடு நடத்தினார் என்று செய்தி.", "https://p.com/latestnews/h"),
    ("Chennai faces acute drinking water shortage", "Residents across Chennai report no drinking water supply for over a week now.", "https://p.com/tamilnadunews/i"),
    ("சேலம் அரசு மருத்துவமனையில் மருந்து தட்டுப்பாடு", "சேலம் அரசு மருத்துவமனையில் மருந்து இல்லாமல் நோயாளிகள் பாதிக்கப்படுகின்றனர்.", "https://p.com/districtnews/j"),
    ("TN government meets China trade delegation in Chennai", "Tamil Nadu government signed an MoU with Chinese firms for Chennai investment.", "https://p.com/tamilnadunews/k"),
    # Cross-border disputes that name Tamil Nadu stay in scope.
    ("முல்லைப்பெரியாறு அணை: தமிழ்நாடு கேரளா இடையே பிரச்சனை", "தமிழ்நாடு கேரளா இடையே முல்லைப்பெரியாறு அணை நீர் பிரச்சனை தொடர்கிறது.", "https://p.com/news/x1"),
    ("Tamil Nadu CM raises Cauvery issue with Karnataka", "Tamil Nadu government pressed Karnataka on Cauvery water release.", "https://p.com/news/x2"),
]


@pytest.mark.parametrize("title,body,url", OUT_OF_SCOPE)
def test_out_of_scope_is_rejected(title, body, url):
    assert _looks_like_tn_content(title, body, url) is False


@pytest.mark.parametrize("title,body,url", IN_SCOPE)
def test_in_scope_is_accepted(title, body, url):
    assert _looks_like_tn_content(title, body, url) is True


def test_sport_film_is_hard_out_of_scope_even_with_tn_place():
    # A TN city name must not rescue sports copy into a government briefing.
    assert _is_hard_out_of_scope("Chennai Super Kings win IPL final", "IPL trophy", "") is True


def test_foreign_country_is_soft_only_does_not_block_tn_story():
    # Foreign mention alone is out of scope, but a TN-marked story stays in.
    assert _is_out_of_scope("China economy slows", "China GDP report", "") is True
    assert _looks_like_tn_content("Tamil Nadu CM meets China delegation", "TN MoU with China", "") is True


# --- people-issue precision ------------------------------------------------

PEOPLE_ISSUES = [
    ("Chennai drinking water shortage", "No water supply for a week in several wards; residents protest."),
    ("சேலம் மருத்துவமனையில் மருந்து தட்டுப்பாடு", "மருந்து இல்லாமல் நோயாளிகள் பாதிப்பு."),
    ("School building wall collapse injures students", "A wall collapsed at a government school injuring two students."),
    ("கிராமத்தில் மின்வெட்டு; மக்கள் போராட்டம்", "தொடர் மின்வெட்டால் பாதிக்கப்பட்ட மக்கள் போராட்டத்தில் ஈடுபட்டனர்."),
]

NOT_PEOPLE_ISSUES = [
    # bare common words with no civic problem
    ("Water park opens in Chennai", "A new water theme park opened for tourists in the city this weekend."),
    ("Road trip film review", "The road movie is a fun youth entertainer with good music and visuals."),
    # sport / foreign are never people issues even with 'death'/'accident'
    ("FIFA star injured in match", "The footballer suffered an accident during the World Cup fixture."),
    ("Pakistan blast kills many", "An explosion in Pakistan caused several deaths according to reports."),
    # plain positive government scheme — not a grievance
    ("CM launches new welfare scheme", "The Chief Minister announced a welcome new welfare scheme for families."),
]


@pytest.mark.parametrize("title,body", PEOPLE_ISSUES)
def test_genuine_people_issue_detected(title, body):
    assert _detect_people_issue(title, body) is True


@pytest.mark.parametrize("title,body", NOT_PEOPLE_ISSUES)
def test_non_people_issue_not_flagged(title, body):
    assert _detect_people_issue(title, body) is False


# --- analyzer-level (end to end through the cascade tiers) ------------------

def _item(title: str, body: str, url: str = "https://p.com/news/x") -> NormalizedItem:
    return NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Polimer",
        source_url=url,
        language="ta",
        title=title,
        raw_text_original=body,
        clean_text_original=body,
    )


@pytest.mark.parametrize("analyzer", [LocalTamilAnalyzer(), MockAIAnalyzer()])
def test_analyzer_marks_sports_not_relevant(analyzer):
    body = "FIFA World Cup 2026 underdogs preview ahead of the tournament kickoff this week in detail."
    analysis = analyzer.analyze(_item("The 4 Davids of FIFA WC 2026", body, "https://p.com/sportsnews/x"))
    assert analysis.government_relevance == GovernmentRelevance.NONE
    assert analysis.people_issue is False


@pytest.mark.parametrize("analyzer", [LocalTamilAnalyzer(), MockAIAnalyzer()])
def test_analyzer_does_not_overflag_people_issue(analyzer):
    # A positive TN government scheme is relevant, but NOT a people issue.
    body = (
        "தமிழக அரசு இன்று புதிய நலத்திட்டத்தை அறிவித்தது. முதலமைச்சர் இதனை வரவேற்றுள்ளார். "
        "இத்திட்டம் மக்களுக்கு பயன் தரும் என அரசு கூறுகிறது. அனைத்து மாவட்டங்களிலும் செயல்படும்."
    )
    analysis = analyzer.analyze(_item("தமிழக அரசு புதிய நலத்திட்டத்தை அறிவித்தது", body))
    assert analysis.government_relevance != GovernmentRelevance.NONE
    assert analysis.people_issue is False


@pytest.mark.parametrize("analyzer", [LocalTamilAnalyzer(), MockAIAnalyzer()])
def test_analyzer_flags_real_civic_grievance(analyzer):
    # Body kept comfortably above the RSS-shell length threshold so the gate
    # evaluates the real article, not a stub.
    body = (
        "சென்னையில் பல வார்டுகளில் கடந்த ஒரு வாரமாக குடிநீர் வழங்கப்படவில்லை என மக்கள் "
        "தெரிவிக்கின்றனர். தண்ணீர் தட்டுப்பாட்டால் பாதிக்கப்பட்ட பகுதி மக்கள் சாலை மறியலில் "
        "ஈடுபட்டு போராட்டம் நடத்தினர். மெட்ரோ நீர்வாரியம் இதுவரை எந்த பதிலையும் அளிக்கவில்லை "
        "என்பதால் மக்கள் கடும் சீற்றத்தில் உள்ளனர். உடனடியாக தண்ணீர் வழங்கக் கோரி மனு அளித்தனர்."
    )
    analysis = analyzer.analyze(_item("சென்னையில் குடிநீர் தட்டுப்பாடு; மக்கள் போராட்டம்", body))
    assert analysis.government_relevance != GovernmentRelevance.NONE
    assert analysis.people_issue is True
