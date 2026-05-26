from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Protocol

from openai import OpenAI

from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
    Sentiment,
    Severity,
    Stance,
)


PROMPT_VERSION = "chief-briefing-v6"


class AIAnalysisError(RuntimeError):
    """Raised when an AI provider cannot return a usable analysis."""


class AIAnalyzer(Protocol):
    model_name: str

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        ...


# ---------------------------------------------------------------------------
# Mock analyzer — used when an OpenAI key is unavailable. It uses real article
# text (first sentence, evidence quote) so the dashboard never displays a
# literal placeholder like "Mock analysis summary." in front of officials.
# ---------------------------------------------------------------------------

_TVK_KEYWORDS_TA = ("தவெக", "தமிழக வெற்றி", "விஜய்", "தலைவர்")
_TVK_KEYWORDS_EN = ("tvk", "tamilaga vettri", "vijay", "thalapathy")
_GOVERNMENT_KEYWORDS_TA = ("அரசு", "முதலமைச்சர்", "அமைச்சர்", "மாவட்ட ஆட்சியர்")
_GOVERNMENT_KEYWORDS_EN = ("government", "chief minister", "minister ", "collector", "tamil nadu government")
_POSITIVE_KEYWORDS = ("திட்ட", "வரவேற்", "நன்றி", "scheme", "welcome", "thank", "praise", "support")
_NEGATIVE_KEYWORDS = ("எதிர்ப்", "புகார்", "கண்டன", "protest", "blame", "criticis", "complain", "scam")

# Article shells — RSS landing/index/live pages that contain a generic title
# and a one-line abstract instead of a real article. We do not pretend to have
# a stance on these; relevance gets set to NONE so the briefing skips them.
_LISTING_PAGE_TITLE_MARKERS = (
    # Breaking / live / index variants
    "breaking news",
    "news live",
    "latest news",
    "live updates",
    "top headlines",
    "tamil news live",
    "latest tamil news",
    "newsletter",
    # Section landing pages (Dinamani / Hindu Tamil / Times of India patterns)
    "sports news",
    "cinema news",
    "movie news",
    "tamil cinema",
    "tamil movie",
    "election news",
    "business news",
    "world news",
    "news in tamil",
    "news & reviews",
    "photos, videos",
    # Tamil section markers
    "தமிழ் நியூஸ்",
    "இன்றைய செய்திகள்",
    "சமீபத்திய செய்திகள்",
    "லேட்டஸ்ட் செய்திகள்",
    "சினிமா செய்திகள்",
    "விளையாட்டுச் செய்திகள்",
    "தேர்தல் செய்திகள்",
    "உலக செய்திகள்",
    "வர்த்தக செய்திகள்",
)
_MIN_ARTICLE_CHARS = 160  # below this we treat it as an RSS shell, not an article
# Stacked-headline detection: section landing pages often paste 5+ distinct
# headlines together separated by "!" / "?" / "." within the first ~1.2 KB of
# body text. A real article rarely contains that many sentence-terminators in
# so little space.
_STACKED_HEADLINE_WINDOW = 1200
_STACKED_HEADLINE_MIN_BREAKS = 5


def _looks_like_listing_page(title: str, body: str) -> bool:
    """True if the article looks like a generic RSS landing/listing page rather
    than a real story. We err on the side of false (let it through) — only the
    obviously-thin pages get dropped."""
    body_stripped = body.strip()
    if len(body_stripped) < _MIN_ARTICLE_CHARS:
        return True
    title_lower = (title or "").lower()
    for marker in _LISTING_PAGE_TITLE_MARKERS:
        if marker in title_lower:
            return True
    # Stacked-headline detector: section landing pages glue 5+ headlines
    # together in the first ~1 KB. Real articles have a flowing paragraph.
    window = body_stripped[:_STACKED_HEADLINE_WINDOW]
    breaks = window.count("!") + window.count("?")
    if breaks >= _STACKED_HEADLINE_MIN_BREAKS:
        return True
    return False


# ---------------------------------------------------------------------------
# Tamil Nadu relevance gate
# ---------------------------------------------------------------------------
# The briefing covers Tamil Nadu public affairs only. National-only / Bollywood
# / cricket / Karnataka-only / Kerala-only / US news has no place on the
# chief's dashboard. We classify an article as "not about TN" when NONE of
# these tokens appears in the title or body — the dashboard then hides it the
# same way it hides RSS shells.
#
# Set is intentionally generous (includes all 38 districts, top politicians,
# party names, TN-specific institutions) so we never miss a real TN story.

_TN_KEYWORDS_EN = (
    # State name + obvious umbrella terms
    "tamil nadu", "tamilnadu", "tamil-nadu", " tn ", "tn govt", "tn government",
    "tamilian", "dravidian", "tamil ",
    # 38 districts (lower-cased; common alt spellings included)
    "chennai", "coimbatore", "kovai", "madurai", "tiruchirappalli", "trichy",
    "salem", "tirunelveli", "vellore", "erode", "thoothukudi", "tuticorin",
    "kanyakumari", "kanniyakumari", "thanjavur", "tanjore", "dindigul",
    "tiruvallur", "tiruvarur", "nagapattinam", "mayiladuthurai", "cuddalore",
    "villupuram", "kallakurichi", "kanchipuram", "krishnagiri", "dharmapuri",
    "namakkal", "karur", "pudukkottai", "ramanathapuram", "ariyalur", "perambalur",
    "sivaganga", "theni", "virudhunagar", "nilgiris", "ooty", "tiruppur",
    "tirupattur", "ranipet", "chengalpattu", "tenkasi",
    # TN political parties + leaders
    "dmk", "aiadmk", "tvk", "ntk", "pmk", "vck", "mdmk",
    "stalin", "udhayanidhi", "edappadi", "palaniswami",
    "anbumani", "ramadoss", "vijay ", "thirumavalavan", "vaiko",
    "kanimozhi", "tamilisai", "annamalai", "rajinikanth", "kamal haasan",
    # TN-specific institutions / topics
    "cauvery", "kaveri", "mullaperiyar", "tn assembly",
    "tamil nadu legislative", "marina beach", "chennai port", "kalaignar",
)

_TN_KEYWORDS_TA = (
    # State
    "தமிழ்நாடு", "தமிழக", "தமிழகம்", "தமிழன்", "தமிழ்",
    # Major districts
    "சென்னை", "கோவை", "கோயம்புத்தூர்", "மதுரை", "திருச்சி", "திருச்சிராப்பள்ளி",
    "சேலம்", "திருநெல்வேலி", "வேலூர்", "ஈரோடு", "தூத்துக்குடி", "கன்னியாகுமரி",
    "தஞ்சை", "தஞ்சாவூர்", "திண்டுக்கல்", "திருவள்ளூர்", "திருவாரூர்",
    "நாகப்பட்டினம்", "மயிலாடுதுறை", "கடலூர்", "விழுப்புரம்", "கல்லக்குறிச்சி",
    "காஞ்சிபுரம்", "கிருஷ்ணகிரி", "தர்மபுரி", "நாமக்கல்", "கரூர்",
    "புதுக்கோட்டை", "ராமநாதபுரம்", "அரியலூர்", "பெரம்பலூர்", "சிவகங்கை",
    "தேனி", "விருதுநகர்", "நீலகிரி", "ஊட்டி", "திருப்பூர்", "திருப்பத்தூர்",
    "ராணிப்பேட்டை", "செங்கல்பட்டு", "தென்காசி",
    # Parties + leaders
    "திமுக", "தி.மு.க", "அதிமுக", "அ.தி.மு.க", "தவெக", "தமிழக வெற்றி",
    "நாம் தமிழர்", "ந.த.க", "பாமக", "வி.சி.க", "ம.தி.மு.க",
    "ஸ்டாலின்", "முதலமைச்சர்", "உதயநிதி", "எடப்பாடியார்", "பழனிசாமி",
    "அன்புமணி", "ராமதாஸ்", "விஜய்", "தலைவர் விஜய்", "திருமாவளவன்",
    "வைகோ", "கனிமொழி", "தமிழிசை", "அண்ணாமலை", "ரஜினிகாந்த்", "கமல் ஹாசன்",
    # TN-specific affairs / institutions
    "காவிரி", "கூவம்", "முல்லைப்பெரியாறு", "மரீனா", "சென்னை துறைமுகம்",
    "தமிழ்நாடு சட்டப்பேரவை", "சட்டப்பேரவை", "தமிழ் மொழி", "கலைஞர்",
)


def _not_relevant_analysis(*, title: str, evidence_quote: str, issue_category: str) -> AIAnalysis:
    """Construct a clean 'not relevant to TVK briefing' AIAnalysis. Used by
    the listing-page gate and the Tamil-Nadu-relevance gate so both produce
    identically-shaped records that the dashboard hides via
    government_relevance == 'none'."""
    return AIAnalysis(
        government_relevance=GovernmentRelevance.NONE,
        stance_toward_government=Stance.NEUTRAL,
        sentiment=Sentiment.NEUTRAL,
        target="Not applicable",
        department="general",
        district="unspecified",
        scheme=None,
        topic=title or "out-of-scope",
        issue_category=issue_category,
        severity=Severity.LOW,
        summary_original=_truncate(evidence_quote or "", 200),
        summary_english="",
        party_action="",
        people_impact="",
        root_cause="",
        recommended_step="",
        positive_points=[],
        negative_points=[],
        evidence_quotes_original=[_truncate(evidence_quote, 200)] if evidence_quote else [],
        evidence_quotes_english=[],
        confidence=0.25,
        needs_human_review=False,
    )


def _looks_like_tn_content(title: str, body: str) -> bool:
    """True if the article references Tamil Nadu — state name, any of the 38
    districts, TN political parties or leaders, or TN-specific institutions.

    Heuristic only; OpenAI is the ground truth when available. We accept
    false positives (article mentions Chennai in passing) because over-
    including is cheaper than dropping a real TN story.
    """
    if not title and not body:
        return False
    haystack = f"{title}\n{body}".lower()
    if any(token in haystack for token in _TN_KEYWORDS_EN):
        return True
    if any(token in body for token in _TN_KEYWORDS_TA):
        return True
    if any(token in title for token in _TN_KEYWORDS_TA):
        return True
    return False


class MockAIAnalyzer:
    model_name = "mock"

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        title = (item.title or "").strip()
        body = (item.clean_text_original or "").strip()
        text = f"{title}\n{body}".lower()

        # Quality gate: RSS landing/listing pages have generic titles and a
        # one-line abstract. They are NOT articles. We classify them as
        # not-relevant so the briefing dashboard can skip them.
        if _looks_like_listing_page(title, body):
            evidence_quote = _first_sentence(body) or title
            return _not_relevant_analysis(
                title=title,
                evidence_quote=evidence_quote,
                issue_category="listing",
            )

        # Tamil Nadu relevance gate: if the article never references TN
        # (state, any district, a TN party/leader, or a TN-specific topic),
        # it's irrelevant to the chief's briefing — skip it.
        if not _looks_like_tn_content(title, body):
            evidence_quote = _first_sentence(body) or title
            return _not_relevant_analysis(
                title=title,
                evidence_quote=evidence_quote,
                issue_category="out-of-scope",
            )

        is_party = any(k in text for k in _TVK_KEYWORDS_EN) or any(k in body for k in _TVK_KEYWORDS_TA)
        is_government = (
            any(k in text for k in _GOVERNMENT_KEYWORDS_EN)
            or any(k in body for k in _GOVERNMENT_KEYWORDS_TA)
        )
        positive = any(k in text for k in _POSITIVE_KEYWORDS)
        negative = any(k in text for k in _NEGATIVE_KEYWORDS)

        if positive and not negative:
            stance = Stance.POSITIVE
        elif negative and not positive:
            stance = Stance.NEGATIVE
        elif positive and negative:
            stance = Stance.MIXED
        else:
            stance = Stance.NEUTRAL

        if is_government:
            relevance = GovernmentRelevance.HIGH
        elif is_party:
            relevance = GovernmentRelevance.MEDIUM
        else:
            relevance = GovernmentRelevance.LOW
        sentiment = (
            Sentiment.POSITIVE if stance == Stance.POSITIVE
            else Sentiment.NEGATIVE if stance == Stance.NEGATIVE
            else Sentiment.NEUTRAL
        )

        first_sentence_orig = _first_sentence(body) or title
        evidence_quote = _first_sentence(body) or title

        # Honest English summary: only synthesise an English line when the
        # title is itself in English (mostly ASCII). Otherwise leave English
        # fields empty so the dashboard's fallback uses the original-language
        # text — never present Tamil masquerading as the English analysis.
        title_is_english = bool(title) and _looks_english(title)
        english_summary = ""
        if title_is_english:
            english_summary = _truncate(title, 180)
        english_party = ""
        english_people = ""
        if english_summary:
            if is_party:
                english_party = english_summary
            elif positive or negative:
                english_people = english_summary

        recommended = ""
        root_cause = ""
        if negative:
            recommended = "Address the public concern raised in the report."
            root_cause = (
                "Reported public grievance or critical event highlighted in the article."
            )
        elif positive:
            recommended = "Sustain visibility on the development if relevant."
            root_cause = (
                "Reported government scheme or supportive action covered in the article."
            )
        elif is_government:
            root_cause = "Government activity reported in the article."

        return AIAnalysis(
            government_relevance=relevance,
            stance_toward_government=stance,
            sentiment=sentiment,
            target="TVK leadership" if is_party else "Public matter",
            department="general",
            district="unspecified",
            scheme=None,
            topic=title or "news item",
            issue_category=("welfare" if positive else "concern") if (positive or negative) else "general",
            severity=Severity.HIGH if negative else Severity.LOW,
            summary_original=_truncate(first_sentence_orig or "", 200),
            summary_english=english_summary,
            party_action=english_party,
            people_impact=english_people,
            root_cause=root_cause,
            recommended_step=recommended,
            positive_points=[_truncate(english_summary, 140)] if positive and english_summary else [],
            negative_points=[_truncate(english_summary, 140)] if negative and english_summary else [],
            evidence_quotes_original=[_truncate(evidence_quote, 240)] if evidence_quote else [],
            evidence_quotes_english=[],
            confidence=0.55,
            needs_human_review=negative,
        )


def _looks_english(text: str) -> bool:
    if not text:
        return False
    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    return ascii_chars / max(1, len(text)) >= 0.85


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    # Split on common sentence terminators across English + Tamil.
    candidates = re.split(r"(?<=[.!?।。])\s+|\n+", text.strip(), maxsplit=1)
    sentence = candidates[0].strip() if candidates else text.strip()
    return sentence


def _truncate(text: str, limit: int) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# OpenAI analyzer — produces real briefing-ready analysis under the chief's
# lens: party member activity, people impact, recommended step.
# ---------------------------------------------------------------------------


class OpenAIAnalyzer:
    def __init__(self, api_key: str, model_name: str = "gpt-5.4-mini") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        prompt = build_classification_prompt(item)
        response = self.client.responses.parse(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You write the daily intelligence briefing for the Tamilaga Vettri "
                        "Kazhagam (TVK) party leadership office. Be precise, neutral in tone, "
                        "evidence-driven, and write tight — every sentence must be short and "
                        "scannable for a senior reader."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            text_format=AIAnalysis,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise AIAnalysisError(_missing_output_message(response))
        if isinstance(parsed, dict):
            return AIAnalysis.model_validate(parsed)
        return parsed


def build_classification_prompt(item: NormalizedItem) -> str:
    return f"""
You are preparing a 30-second daily briefing for the TVK party leadership office
from a Tamil-Nadu newspaper article. Label every article and explain it in four lines.

The leader's decision loop is: WHAT happened → WHY → SO WHAT → WHAT NOW.

Required output:

  1. STANCE LABEL — exactly one of: positive / negative / mixed / neutral, toward
     the Tamil Nadu Government in office (not TVK). This is the headline label.
  2. PARTY ACTIVITY (party_action) — what TVK members or the party leadership
     did, well or badly, that this article reports. Empty if not about TVK.
  3. PEOPLE'S EXPERIENCE (people_impact) — what ordinary people are facing or
     benefiting from in the area the article covers. Empty if not applicable.
  4. ROOT CAUSE (root_cause) — the underlying WHY: the policy gap, decision,
     event, or structural condition driving what the article reports. Be
     concrete. Empty only if the article gives no causal signal.
  5. RECOMMENDED NEXT STEP (recommended_step) — one concrete action the
     leadership office could consider (visit, statement, relief, internal
     review of a member, public response, fact-finding, etc.). Empty if no
     action is clearly warranted by the evidence.

Tone rules:
- One sentence per briefing field. Each must be ≤ 22 words. No rhetoric.
- Plain English. Avoid "stakeholder", "ecosystem", "leverage", "synergy".
- Do not invent facts. If the article does not warrant a field, leave it "".
- Preserve the original Tamil quotation in evidence_quotes_original verbatim.
- If the article makes an allegation or sensitive claim, set
  needs_human_review = true.
- Populate scheme only when a named government scheme is explicitly mentioned.

Tamil Nadu relevance gate (IMPORTANT):
- This briefing is for the Tamil Nadu party leadership only.
- An article is in scope only when it materially relates to Tamil Nadu —
  the state, any of its 38 districts, a TN political party/leader, a TN
  public matter, or a national story where TN involvement is central.
- An article is OUT OF SCOPE if it is purely national, Bollywood, cricket
  (without TN angle), other states (Karnataka / Kerala / Maharashtra / Delhi /
  US / international) and does not concern TN.
- For OUT OF SCOPE articles set government_relevance = "none" and leave
  every briefing field ("party_action", "people_impact", "root_cause",
  "recommended_step") empty. The dashboard hides relevance=none rows.

Field guidance:
- summary_english: one factual sentence on WHAT happened. ≤ 22 words.
- summary_original: same sentence, original article language.
- party_action: ≤ 22 words. Empty if not applicable.
- people_impact: ≤ 22 words. Empty if not applicable.
- root_cause: ≤ 22 words. Cite a concrete driver, not a platitude.
- recommended_step: ≤ 22 words. Concrete and feasible. Empty if no action.
- topic: 3–6 word phrase, e.g. "Cauvery water release dispute".
- positive_points / negative_points: 1–2 short bullet phrases each at most.
- evidence_quotes_original: 1–2 short verbatim quotes from the article.

Output schema reminder (the SDK will validate; fields shown for reference):
{{
  "scheme": "string|null"
}}
Populate the scheme name only when explicitly mentioned; otherwise null.

Source: {item.source_name}
URL: {item.source_url}
Language: {item.language}
Title: {item.title or ""}
Article text:
{item.clean_text_original}
""".strip()


def _missing_output_message(response: Any) -> str:
    details: list[str] = []
    incomplete_reason = _safe_path(response, "incomplete_details", "reason")
    if incomplete_reason:
        details.append(f"incomplete_reason={incomplete_reason}")

    refusals = _collect_refusals(response)
    if refusals:
        details.append(f"refusal={'; '.join(refusals)}")

    message = "OpenAI structured analysis response did not include parsed output"
    if details:
        message = f"{message}: {', '.join(details)}"
    return message


def _collect_refusals(response: Any) -> list[str]:
    refusals: list[str] = []
    _append_text(refusals, _safe_getattr(response, "refusal"))

    for output_item in _safe_iterable(_safe_getattr(response, "output")):
        _append_text(refusals, _safe_getattr(output_item, "refusal"))
        for content_item in _safe_iterable(_safe_getattr(output_item, "content")):
            _append_text(refusals, _safe_getattr(content_item, "refusal"))

    return refusals


def _safe_path(obj: Any, *path: str) -> Any:
    current = obj
    for name in path:
        current = _safe_getattr(current, name)
        if current is None:
            return None
    return current


def _safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _safe_iterable(value: Any) -> Iterable[Any]:
    if value is None or isinstance(value, str):
        return ()
    try:
        return iter(value)
    except TypeError:
        return ()


def _append_text(values: list[str], value: Any) -> None:
    if isinstance(value, str) and value:
        values.append(value)
