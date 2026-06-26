from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
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
from tnmi.districts import detect_district


PROMPT_VERSION = "tvk-portrayal-v19"


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

_TVK_KEYWORDS_TA = ("தவெக", "தமிழக வெற்றி", "விஜய்", "தலைவர் விஜய்")
_TVK_KEYWORDS_EN = ("tvk", "tamilaga vettri", "vijay", "thalapathy")
_GOVERNMENT_KEYWORDS_TA = ("அரசு", "முதலமைச்சர்", "அமைச்சர்", "மாவட்ட ஆட்சியர்")
_GOVERNMENT_KEYWORDS_EN = ("government", "chief minister", "minister ", "collector", "tamil nadu government")
_POSITIVE_KEYWORDS = ("திட்ட", "வரவேற்", "நன்றி", "scheme", "welcome", "thank", "praise", "support")
_NEGATIVE_KEYWORDS = ("எதிர்ப்", "புகார்", "கண்டன", "protest", "blame", "criticis", "complain", "scam")
_PEOPLE_ISSUE_KEYWORDS = (
    "மக்கள்", "பொதுமக்கள்", "குடிநீர்", "மின்சாரம்", "சாலை", "மருத்துவ", "கல்வி",
    "வேலை", "விவசாய", "புகார்", "பாதிப்பு", "பள்ளி", "மாணவர்", "மாணவி",
    "தீ விபத்து", "தீவிபத்து", "விபத்து", "காயம்", "மரணம்", "உயிரிழப்பு", "பாதுகாப்பு",
    "people", "public", "water", "power", "road", "hospital", "school",
    "student", "students", "fire", "accident", "injury", "injured", "death",
    "safety", "jobs", "farmers", "grievance",
)
_HIGH_RISK_PEOPLE_KEYWORDS = (
    "தீ விபத்து", "தீவிபத்து", "விபத்து", "மரணம்", "உயிரிழப்பு",
    "வன்முறை", "பள்ளி", "மாணவர்", "மாணவி", "மருத்துவமனை", "fire", "accident",
    "killed", "death", "fatal", "school", "student", "hospital", "violence",
)


@dataclass(frozen=True)
class PublicIssueProfile:
    issue_category: str
    public_issue: str
    severity: Severity
    root_cause: str
    recommended_step: str
    action_owner: str
    action_type: str

# TVK is the ruling party in this deployment, so the Chief Minister, ministers
# and many MLAs are TVK office-holders — the office-role keywords below denote
# TVK people. Rival-party leaders are grouped under their (opposition) parties
# so their conduct never sets tvk_portrayal.
_ACTOR_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("TVK", ("tvk", "தவெக", "தமிழக வெற்றி")),
    ("Vijay (CM)", ("vijay", "விஜய்", "தலைவர் விஜய்", "thalapathy", "chief minister", "முதலமைச்சர்", "முதல்வர்")),
    ("Minister", ("minister", "அமைச்சர்")),
    ("MLA", ("mla", "எம்எல்ஏ", "சட்டமன்ற உறுப்பினர்")),
    ("DMK (opposition)", ("dmk", "திமுக", "தி.மு.க", "stalin", "ஸ்டாலின்", "udhayanidhi", "உதயநிதி")),
    ("AIADMK", ("aiadmk", "அதிமுக", "அ.தி.மு.க")),
    ("NTK", ("ntk", "நாம் தமிழர்", "ந.த.க")),
)

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

# NOTE: matched with WORD BOUNDARIES (see _looks_like_tn_content), never raw
# substring — substring search once matched 'theni' inside "strengthening"
# and let international wire stories into the briefing as TN people-issues.
_TN_KEYWORDS_EN = (
    # State name + obvious umbrella terms
    "tamil nadu", "tamilnadu", "tamil-nadu", "tn govt", "tn government",
    "tamilian", "dravidian",
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
    "anbumani", "ramadoss", "vijay", "thirumavalavan", "vaiko",
    "kanimozhi", "tamilisai", "annamalai", "rajinikanth", "kamal haasan",
    # TN-specific institutions / topics
    "cauvery", "kaveri", "mullaperiyar", "tn assembly",
    "tamil nadu legislative", "marina beach", "chennai port", "kalaignar",
)

_TN_KEYWORDS_TA = (
    # State (தமிழக covers தமிழகம்/தமிழகத்தில்; bare தமிழ் is intentionally
    # absent — it matches any mention of the Tamil language anywhere on earth)
    "தமிழ்நாடு", "தமிழ்நாட்", "தமிழக",
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
        tvk_relevance=GovernmentRelevance.NONE,
        tvk_portrayal=Stance.NEUTRAL,
        target="Not applicable",
        political_actors=[],
        department="general",
        district="unspecified",
        scheme=None,
        topic=title or "out-of-scope",
        issue_category=issue_category,
        people_issue=False,
        public_issue="",
        severity=Severity.LOW,
        summary_original=_truncate(evidence_quote or "", 200),
        summary_english="",
        party_action="",
        people_impact="",
        root_cause="",
        recommended_step="",
        action_owner="",
        action_type="monitor",
        action_priority=Severity.LOW,
        positive_points=[],
        negative_points=[],
        evidence_quotes_original=[_truncate(evidence_quote, 200)] if evidence_quote else [],
        evidence_quotes_english=[],
        confidence=0.25,
        needs_human_review=False,
    )


_TN_KEYWORDS_EN_RE = re.compile(
    r"\b(" + "|".join(re.escape(token) for token in _TN_KEYWORDS_EN) + r")\b",
    re.IGNORECASE,
)
# 'erode' is also an English verb — accept only the capitalised place name.
_ERODE_VERB_RE = re.compile(r"\berode[sd]?\b|\beroding\b")


def _tamil_script_ratio(text: str) -> float:
    if not text:
        return 0.0
    tamil = sum(1 for ch in text if "஀" <= ch <= "௿")
    return tamil / len(text)


# ---------------------------------------------------------------------------
# Out-of-scope gate
# ---------------------------------------------------------------------------
# A Tamil-language story with no explicit TN marker used to pass the relevance
# gate purely because it was ≥30% Tamil script. That let Tamil-script FIFA /
# World Cup / foreign-affairs / film-gossip wire copy into a *government*
# briefing. These topics are out of scope no matter what language they are
# written in. URL sections (where the paper exposes them) are a strong booster.

# English markers are matched on WORD BOUNDARIES so short tokens cannot match
# inside other words ('odi' must not fire on "Modi", 'ipl' not on "multiple").
#
# HARD out of scope — international / professional sport and film-trade gossip.
# These never belong in a government briefing even when a TN place is named
# (e.g. "Chennai Super Kings win IPL"), so they override the TN-keyword match.
_SPORT_FILM_EN_RE = re.compile(
    r"\b("
    r"fifa|world cup|t20|t-20|odi|ipl|isl|olympics?|paralympics?|uefa|la liga|"
    r"premier league|bundesliga|serie a|ballon d'?or|wimbledon|grand slam|"
    r"champions league|champions trophy|asia cup|world test|"
    r"box office|teaser|trailer|first look|movie review|film review"
    r")\b",
    re.IGNORECASE,
)
_SPORT_FILM_TA = (
    "கால்பந்து", "கிரிக்கெட்", "உலகக் கோப்பை", "உலக கோப்பை", "டெஸ்ட் போட்டி",
    "ஐபிஎல்", "ஒலிம்பிக்", "டி20", "டி 20", "டி-20",
    "பாக்ஸ் ஆபீஸ்", "டிரெய்லர்", "ட்ரெய்லர்", "டீசர்", "டீஸர்", "ஃபர்ஸ்ட் லுக்",
)
_SPORT_FILM_URL_SECTIONS = (
    "/sportsnews/", "/sports-news/", "/sports/", "/cinema/", "/movies/",
    "/entertainment/", "/video/",
)
# SOFT out of scope — foreign affairs with no Tamil-Nadu nexus. Only applied
# when the article carries NO explicit TN marker, because a genuine TN-government
# story may legitimately mention a foreign country (trade, diaspora, delegation).
_FOREIGN_EN_RE = re.compile(
    r"\b("
    r"pakistan|afghanistan|ukraine|russia|china|beijing|moscow|"
    r"israel|palestine|gaza|iran|iraq|syria|"
    r"thailand|malaysia|singapore|nepal|bangladesh|myanmar|"
    r"usa|america|washington|"
    r"britain|england|france|germany|japan|korea|canada|australia|"
    r"trump|putin|zelensky|netanyahu|biden"
    r")\b",
    re.IGNORECASE,
)
_FOREIGN_TA = (
    "பாகிஸ்தான்", "ஆப்கானிஸ்தான்", "உக்ரைன்", "ரஷ்யா", "சீனா",
    "இஸ்ரேல்", "காசா", "ஈரான்", "ஈராக்", "சிரியா",
    "தாய்லாந்து", "மலேசியா", "சிங்கப்பூர்", "நேபாள", "வங்காளதேச", "வங்கதேச",
    "அமெரிக்க", "வாஷிங்டன்", "இங்கிலாந்து", "பிரான்ஸ்", "ஜெர்மனி", "ஜப்பான்",
    "டிரம்ப்", "புதின்",
)
_FOREIGN_URL_SECTIONS = ("/worldnews/", "/world-news/")

# Other Indian states + their major cities. A Tamil-language story about Kerala
# or Karnataka is still out of scope for a *Tamil Nadu* briefing. Like the
# foreign markers this is SOFT — only applied when the article carries no
# explicit TN marker, so genuine cross-border stories (Cauvery, Mullaperiyar, a
# TN CM visiting another state) that name a TN place are kept.
_OTHER_STATE_EN_RE = re.compile(
    r"\b("
    r"kerala|karnataka|andhra pradesh|andhra|telangana|maharashtra|gujarat|"
    r"rajasthan|punjab|haryana|bihar|odisha|west bengal|assam|jharkhand|"
    r"chhattisgarh|madhya pradesh|uttar pradesh|uttarakhand|himachal|goa|"
    r"manipur|meghalaya|tripura|nagaland|mizoram|sikkim|"
    r"bengaluru|bangalore|mysuru|mysore|mangaluru|mangalore|hubli|"
    r"kochi|cochin|thiruvananthapuram|trivandrum|kozhikode|kollam|thrissur|"
    r"hyderabad|secunderabad|vijayawada|visakhapatnam|vizag|guntur|tirupati|"
    r"mumbai|pune|nagpur|kolkata|ahmedabad|surat|jaipur|lucknow|patna|bhopal|"
    r"new delhi"
    r")\b",
    re.IGNORECASE,
)
_OTHER_STATE_TA = (
    "கேரள", "கர்நாடக", "ஆந்திர", "தெலங்கானா", "தெலுங்கானா", "மகாராஷ்டிர",
    "குஜராத்", "ராஜஸ்தான்", "பஞ்சாப்", "பீகார்", "ஒடிசா", "மேற்கு வங்க", "அஸ்ஸாம்",
    "பெங்களூரு", "பெங்களூர்", "மைசூர்", "மங்களூரு", "கொச்சி", "திருவனந்தபுரம்",
    "ஐதராபாத்", "ஹைதராபாத்", "விஜயவாடா", "விசாகப்பட்டினம்",
    "மும்பை", "புனே", "கொல்கத்தா", "புது தில்லி", "புதுடெல்லி", "டெல்லி",
)


# A story's URL SECTION is authoritative about which desk filed it. A
# `/kerala/` or `/sportsnews/` article is out of scope even when its scraped
# body mentions "Tamil Nadu" — that match is usually site navigation/footer
# boilerplate (e.g. The News Minute lists a "Tamil Nadu" section on every page),
# not the story's subject. So the URL section overrides the TN-keyword check.
_OTHER_STATE_URL_SECTIONS = (
    "/kerala/", "/karnataka/", "/andhra-pradesh/", "/andhrapradesh/", "/andhra/",
    "/telangana/", "/maharashtra/", "/gujarat/", "/rajasthan/", "/punjab/",
    "/bihar/", "/odisha/", "/west-bengal/", "/assam/", "/jharkhand/",
    "/madhya-pradesh/", "/uttar-pradesh/", "/uttarakhand/", "/goa/",
    "/delhi/", "/mumbai/", "/bengaluru/", "/bangalore/", "/hyderabad/",
    "/kochi/", "/kolkata/",
)


def _is_hard_out_of_scope(title: str, body: str, source_url: str = "") -> bool:
    """Topics that never belong in a government briefing even when a TN place is
    named — international / professional sport, film-trade gossip, and stories
    a paper filed under another state's / foreign desk (URL section is the
    authoritative signal, ahead of any TN keyword in scraped page chrome)."""
    url = (source_url or "").lower()
    if any(section in url for section in _SPORT_FILM_URL_SECTIONS):
        return True
    if any(section in url for section in _OTHER_STATE_URL_SECTIONS):
        return True
    if any(section in url for section in _FOREIGN_URL_SECTIONS):
        return True
    haystack_lower = f"{title}\n{body}".lower()
    haystack_original = f"{title}\n{body}"
    if _SPORT_FILM_EN_RE.search(haystack_lower):
        return True
    return any(token in haystack_original for token in _SPORT_FILM_TA)


def _is_out_of_scope(title: str, body: str, source_url: str = "") -> bool:
    """Hard out of scope (sport / film) OR foreign affairs with no TN nexus.
    Used to keep both the relevance gate's Tamil-script fallback and people-issue
    detection free of off-topic copy that merely happens to be in Tamil."""
    if _is_hard_out_of_scope(title, body, source_url):
        return True
    url = (source_url or "").lower()
    if any(section in url for section in _FOREIGN_URL_SECTIONS):
        return True
    haystack_lower = f"{title}\n{body}".lower()
    haystack_original = f"{title}\n{body}"
    if _FOREIGN_EN_RE.search(haystack_lower):
        return True
    if any(token in haystack_original for token in _FOREIGN_TA):
        return True
    # Other Indian states / non-TN cities (e.g. a Kerala or Karnataka story).
    if _OTHER_STATE_EN_RE.search(haystack_lower):
        return True
    return any(token in haystack_original for token in _OTHER_STATE_TA)


def _looks_like_tn_content(title: str, body: str, source_url: str = "") -> bool:
    """True if the article belongs in a Tamil Nadu government briefing.

    An article qualifies when it names Tamil Nadu — the state, any of the 38
    districts, TN parties/leaders, or TN-specific institutions (English tokens
    matched on WORD BOUNDARIES so 'theni' can't match "strengthening").

    A mostly-Tamil article with no explicit marker is a hyperlocal TN story
    (ward roads, school events) and is kept — UNLESS it is out of scope.
    Sport / film is rejected outright; foreign affairs is rejected only when no
    TN marker is present. Previously the Tamil-script ratio alone was enough to
    pass, which let Tamil-language FIFA and world-news copy into the briefing.
    """
    if not title and not body:
        return False
    # Sport / film never qualifies, even with a TN place name in the text.
    if _is_hard_out_of_scope(title, body, source_url):
        return False
    haystack = f"{title}\n{body}"
    for match in _TN_KEYWORDS_EN_RE.finditer(haystack):
        token = match.group(1).lower()
        if token == "erode" and match.group(1)[0].islower():
            continue  # the verb, not the district
        return True
    if any(token in body for token in _TN_KEYWORDS_TA):
        return True
    if any(token in title for token in _TN_KEYWORDS_TA):
        return True
    # No explicit TN marker: keep a Tamil-language local story, but only if it
    # is not an out-of-scope topic that merely happens to be written in Tamil.
    if _tamil_script_ratio(haystack) >= 0.3 and not _is_out_of_scope(title, body, source_url):
        return True
    return False


# Named individuals worth tracking as their own dossier, in addition to the
# party/office bucket. The emitted surface MUST match an alias in
# configs/entities.seed.yaml so the resolver maps it to that person entity —
# this is what lights up per-figure scorecards (Stalin distinct from "DMK").
_NAMED_PERSON_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("M.K. Stalin", ("stalin", "ஸ்டாலின்")),
    ("Udhayanidhi Stalin", ("udhayanidhi", "உதயநிதி")),
    ("Edappadi K. Palaniswami", ("edappadi", "palaniswami", "eps", "எடப்பாடி", "பழனிசாமி")),
    ("K. Annamalai", ("annamalai", "அண்ணாமலை")),
    ("Seeman", ("seeman", "சீமான்")),
    ("Anbumani Ramadoss", ("anbumani", "அன்புமணி")),
    ("Thol. Thirumavalavan", ("thirumavalavan", "திருமாவளவன்")),
)


def _extract_political_actors(title: str, body: str) -> list[str]:
    haystack_lower = f"{title}\n{body}".lower()
    haystack_original = f"{title}\n{body}"
    actors: list[str] = []
    for label, tokens in _ACTOR_KEYWORDS:
        if any(_actor_token_matches(token, haystack_lower, haystack_original) for token in tokens):
            actors.append(label)
    # Named individuals are additive: a Stalin story still carries the
    # "DMK (opposition)" bucket AND now a distinct "M.K. Stalin" actor.
    for label, tokens in _NAMED_PERSON_KEYWORDS:
        if any(_actor_token_matches(token, haystack_lower, haystack_original) for token in tokens):
            actors.append(label)
    # Preserve first-seen order while de-duplicating.
    seen: set[str] = set()
    unique = [a for a in actors if not (a in seen or seen.add(a))]
    return unique[:10]


def _actor_token_matches(token: str, haystack_lower: str, haystack_original: str) -> bool:
    normalized = token.strip().lower()
    if not normalized:
        return False
    if any("a" <= char <= "z" for char in normalized):
        return re.search(rf"\b{re.escape(normalized)}\b", haystack_lower) is not None
    return token in haystack_original


def _has_people_issue(text: str, body: str) -> bool:
    return any(token in text for token in _PEOPLE_ISSUE_KEYWORDS) or any(token in body for token in _PEOPLE_ISSUE_KEYWORDS)


def _people_issue_severity(text: str, body: str) -> Severity:
    if any(token in text for token in _HIGH_RISK_PEOPLE_KEYWORDS) or any(token in body for token in _HIGH_RISK_PEOPLE_KEYWORDS):
        return Severity.HIGH
    return Severity.MEDIUM


def _contains_any(text_lower: str, text_original: str, tokens: tuple[str, ...]) -> bool:
    return any(token in text_lower or token in text_original for token in tokens)


def _has_tvk_reference(text_lower: str, text_original: str) -> bool:
    english_match = any(re.search(rf"\b{re.escape(token)}\b", text_lower) for token in _TVK_KEYWORDS_EN)
    tamil_match = any(token in text_original for token in _TVK_KEYWORDS_TA)
    return english_match or tamil_match


def _brief_article_focus(title: str, evidence_quote: str, *, limit: int = 96) -> str:
    focus = (title or evidence_quote or "").strip()
    focus = re.sub(r"\s+", " ", focus)
    return _truncate(focus, limit)


def _has_school_context(text_lower: str, text_original: str) -> bool:
    # Pallikaranai is a Chennai place name; substring matching would otherwise
    # read the Tamil prefix "பள்ளி" as "school".
    for false_school_place in ("pallikaranai", "பள்ளிக்கரணை"):
        text_lower = text_lower.replace(false_school_place, "")
        text_original = text_original.replace(false_school_place, "")
    return _contains_any(
        text_lower,
        text_original,
        ("school", "student", "students", "பள்ளி", "மாணவர்", "மாணவி"),
    )


def _verification_targets(profile: PublicIssueProfile) -> str:
    issue = profile.public_issue
    if issue == "school safety incident":
        return "student safety, injuries, evacuation and school/fire-service response"
    if profile.issue_category == "public_safety":
        return "casualties, public safety risk, cause and fire/police/civic response"
    if issue == "drinking water shortage":
        return "affected streets, outage duration and Metro Water or local-body response"
    if issue == "electricity service issue":
        return "outage or safety details, affected homes and electricity-board response"
    if issue == "road or traffic issue":
        return "exact location, photos, commuter impact and local authority response"
    if profile.issue_category == "health":
        return "patient impact, facility response and health-department follow-up"
    if profile.issue_category == "education":
        return "student or parent impact, school facts and education-department response"
    if profile.issue_category == "livelihood":
        return "affected workers or youth, numbers involved and available relief route"
    if profile.issue_category == "agriculture":
        return "affected farmers, crop or compensation impact and district authority response"
    return "affected people, location, evidence and official response"


def _contextual_root_cause(profile: PublicIssueProfile, *, evidence_quote: str) -> str:
    evidence = _truncate(re.sub(r"\s+", " ", evidence_quote.strip()), 160)
    if not evidence:
        return profile.root_cause
    if profile.issue_category == "public_safety":
        return f"Evidence reports a safety incident: {evidence}"
    if profile.issue_category == "civic_services":
        return f"Evidence points to a civic-service gap: {evidence}"
    if profile.issue_category == "health":
        return f"Evidence points to a health-service concern: {evidence}"
    if profile.issue_category == "education":
        return f"Evidence points to an education concern: {evidence}"
    if profile.issue_category in {"livelihood", "agriculture"}:
        return f"Evidence points to a livelihood impact: {evidence}"
    return f"Evidence reports a public grievance: {evidence}"


def _contextual_recommended_step(
    profile: PublicIssueProfile,
    *,
    title: str,
    evidence_quote: str,
) -> str:
    focus = _brief_article_focus(title, evidence_quote)
    targets = _verification_targets(profile)
    owner = profile.action_owner or "District field team"
    if focus:
        return f"{owner}: verify {targets} for '{focus}' before statement or support."
    return f"{owner}: verify {targets} before statement or support."


# ---------------------------------------------------------------------------
# People-issue detection (precise)
# ---------------------------------------------------------------------------
# The old rule — people_issue = (any people keyword) OR (any negative keyword) —
# fired on a single common word ("water", "road", "death") and flagged ~79% of
# relevant items, including sports copy. A real public/people issue needs a
# public-service DOMAIN together with a PROBLEM indicator, or a genuine
# public-safety incident. English tokens use word boundaries.
_CIVIC_DOMAIN_EN_RE = re.compile(
    r"\b("
    r"water|drinking water|electricity|power supply|power cut|powercut|"
    r"road|roads|street|streets|bus|transport|train|railway|"
    r"hospital|clinic|phc|medical|healthcare|"
    r"school|schools|college|education|student|students|"
    r"housing|ration|pension|sewage|sewerage|drainage|garbage|sanitation|toilet|"
    r"job|jobs|employment|unemploy\w*|wage|wages|"
    r"farmer|farmers|agriculture|crop|crops|irrigation"
    r")\b",
    re.IGNORECASE,
)
_PROBLEM_EN_RE = re.compile(
    r"\b("
    r"shortage|scarcity|cut|cuts|without|lack|denied|deny|delay|delayed|pending|"
    r"damage|damaged|broken|pothole|potholes|contaminat\w*|polluted|pollution|"
    r"dirty|leak|leakage|overflow|protest|protests|strike|demand|demands|"
    r"grievance|grievances|complaint|complaints|stranded|unsafe|danger|hazard|"
    r"collapse|collapsed|closed|shut|bribe|corrupt\w*|eviction|encroach\w*|"
    r"stagnant|blocked|disrupt\w*|suffer\w*|affected|deprived|negligence"
    r")\b",
    re.IGNORECASE,
)
_SAFETY_INCIDENT_EN_RE = re.compile(
    r"\b("
    r"fire|blaze|accident|crash|drown|drowned|electrocut\w*|stampede|"
    r"building collapse|wall collapse|gas leak"
    r")\b",
    re.IGNORECASE,
)
_CIVIC_DOMAIN_TA = (
    "குடிநீர்", "தண்ணீர்", "மின்சாரம்", "மின்வெட்டு", "சாலை", "தெரு",
    "பேருந்து", "ரயில்", "மருத்துவமனை", "மருத்துவம்", "பள்ளி", "கல்லூரி",
    "மாணவ", "கல்வி", "வீட்டுவசதி", "குடியிருப்பு", "ரேஷன்", "ஓய்வூதியம்",
    "கழிவுநீர்", "வடிகால்", "குப்பை", "கழிப்பறை", "வேலைவாய்ப்பு", "ஊதியம்",
    "விவசாய", "பயிர்", "நீர்ப்பாசன",
)
_PROBLEM_TA = (
    "பற்றாக்குறை", "தட்டுப்பாடு", "வெட்டு", "இல்லாமல்", "இல்லை", "பாதிப்பு",
    "பாதிக்கப்பட்ட", "சேதம்", "புகார்", "போராட்டம்", "மறியல்", "கோரிக்கை",
    "தாமதம்", "மாசு", "கசிவு", "ஆபத்து", "மூடல்", "அவலம்", "துயரம்",
)
_SAFETY_INCIDENT_TA = (
    "தீ விபத்து", "தீவிபத்து", "விபத்து", "மூழ்கி", "மின்தாக்க",
    "சரிந்து", "இடிந்து", "நெரிசல்",
)


def _detect_people_issue(title: str, body: str, source_url: str = "") -> bool:
    """A genuine Tamil-Nadu public/people issue — not merely a negative tone or
    a passing common word. Requires a public-service DOMAIN together with a
    PROBLEM indicator, or a real public-safety incident. Out-of-scope topics
    (sport / foreign affairs / film) are never people issues."""
    if _is_out_of_scope(title, body, source_url):
        return False
    lower = f"{title}\n{body}".lower()
    original = f"{title}\n{body}"
    if _SAFETY_INCIDENT_EN_RE.search(lower) or any(t in original for t in _SAFETY_INCIDENT_TA):
        return True
    domain = bool(_CIVIC_DOMAIN_EN_RE.search(lower)) or any(t in original for t in _CIVIC_DOMAIN_TA)
    problem = bool(_PROBLEM_EN_RE.search(lower)) or any(t in original for t in _PROBLEM_TA)
    return domain and problem


def _public_issue_profile(title: str, body: str) -> PublicIssueProfile:
    text_lower = f"{title}\n{body}".lower()
    text_original = f"{title}\n{body}"
    fire_terms = ("fire", "தீ விபத்து", "தீவிபத்து")
    accident_terms = (
        "accident", "killed", "runs over", "run over", "hit by", "death", "fatal",
        "விபத்து", "மரணம்", "உயிரிழப்பு",
    )
    water_terms = ("water", "drinking water", "குடிநீர்", "தண்ணீர்")
    power_terms = ("power", "electricity", "மின்சாரம்")
    road_terms = ("road", "pothole", "traffic", "சாலையில்", "சாலைப்", "சாலை ", " சாலை")
    health_terms = ("hospital", "health", "medical", "மருத்துவ", "மருத்துவமனை")
    education_terms = ("education", "கல்வி")
    jobs_terms = ("job", "jobs", "employment", "youth", "வேலை", "தொழில்", "இளைஞர்")
    farmer_terms = ("farmer", "farmers", "agriculture", "விவசாய", "விவசாயி")
    has_school_context = _has_school_context(text_lower, text_original)

    if has_school_context and (
        _contains_any(text_lower, text_original, fire_terms)
        or _contains_any(text_lower, text_original, accident_terms)
    ):
        return PublicIssueProfile(
            issue_category="public_safety",
            public_issue="school safety incident",
            severity=Severity.HIGH,
            root_cause="The article reports a school-linked safety incident; facts, injuries and official response need local verification.",
            recommended_step="Send the district team to verify injuries, permissions and response before deciding TVK's public action.",
            action_owner="District field team",
            action_type="field_verification",
        )
    if _contains_any(text_lower, text_original, fire_terms) or _contains_any(text_lower, text_original, accident_terms):
        return PublicIssueProfile(
            issue_category="public_safety",
            public_issue="public safety incident",
            severity=Severity.HIGH,
            root_cause="The article reports a safety incident; cause, casualties and civic response need verification.",
            recommended_step="Verify casualties, official response and local needs before issuing a statement or arranging support.",
            action_owner="District field team",
            action_type="field_verification",
        )
    if _contains_any(text_lower, text_original, water_terms):
        return PublicIssueProfile(
            issue_category="civic_services",
            public_issue="drinking water shortage",
            severity=Severity.MEDIUM,
            root_cause="The report points to a local drinking-water service gap affecting residents.",
            recommended_step="Ask the local team to confirm affected streets and escalate the water issue with evidence.",
            action_owner="District field team",
            action_type="field_verification",
        )
    if _contains_any(text_lower, text_original, power_terms):
        return PublicIssueProfile(
            issue_category="civic_services",
            public_issue="electricity service issue",
            severity=Severity.MEDIUM,
            root_cause="The report points to a local electricity service or safety problem affecting residents.",
            recommended_step="Verify outage or safety details locally and route the evidence to the electricity grievance channel.",
            action_owner="District field team",
            action_type="field_verification",
        )
    if _contains_any(text_lower, text_original, road_terms):
        return PublicIssueProfile(
            issue_category="civic_services",
            public_issue="road or traffic issue",
            severity=Severity.MEDIUM,
            root_cause="The report points to a local road, access or traffic problem affecting daily movement.",
            recommended_step="Collect location photos and resident accounts, then escalate the road issue to local authorities.",
            action_owner="District field team",
            action_type="field_verification",
        )
    if _contains_any(text_lower, text_original, health_terms):
        return PublicIssueProfile(
            issue_category="health",
            public_issue="health service issue",
            severity=Severity.MEDIUM,
            root_cause="The report points to a health-service concern that may affect patient access or safety.",
            recommended_step="Verify patient impact and facility response before preparing a health-department follow-up.",
            action_owner="Policy research team",
            action_type="policy_research",
        )
    if has_school_context or _contains_any(text_lower, text_original, education_terms):
        return PublicIssueProfile(
            issue_category="education",
            public_issue="education issue",
            severity=Severity.MEDIUM,
            root_cause="The report points to an education concern affecting students or school operations.",
            recommended_step="Verify school-level facts and identify whether parents or students need immediate support.",
            action_owner="District field team",
            action_type="field_verification",
        )
    if _contains_any(text_lower, text_original, jobs_terms):
        return PublicIssueProfile(
            issue_category="livelihood",
            public_issue="jobs or youth issue",
            severity=Severity.MEDIUM,
            root_cause="The report points to a livelihood or employment concern affecting workers or youth.",
            recommended_step="Collect affected-group details and prepare an evidence-backed livelihood response.",
            action_owner="Policy research team",
            action_type="policy_research",
        )
    if _contains_any(text_lower, text_original, farmer_terms):
        return PublicIssueProfile(
            issue_category="agriculture",
            public_issue="farmer issue",
            severity=Severity.MEDIUM,
            root_cause="The report points to an agriculture concern affecting farmers or rural livelihoods.",
            recommended_step="Verify farmer impact locally and prepare a district-specific relief or policy demand.",
            action_owner="District field team",
            action_type="field_verification",
        )
    return PublicIssueProfile(
        issue_category="concern",
        public_issue="public grievance",
        severity=_people_issue_severity(text_lower, text_original),
        root_cause="The article reports a public grievance; local facts and affected groups need verification.",
        recommended_step="Verify the affected people, location and official response before choosing a public or internal follow-up.",
        action_owner="District field team",
        action_type="field_verification",
    )


def _action_owner(*, is_party: bool, people_issue: bool, is_government: bool) -> str:
    if is_party:
        return "TVK leadership office"
    if people_issue:
        return "District field team"
    if is_government:
        return "Policy research team"
    return "Media monitoring desk"


def _action_type(*, stance: Stance, is_party: bool, people_issue: bool) -> str:
    if is_party and stance == Stance.NEGATIVE:
        return "internal_review"
    if people_issue:
        return "field_verification"
    if stance == Stance.POSITIVE:
        return "amplify"
    if stance == Stance.MIXED:
        return "monitor"
    return "monitor"


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
        if not _looks_like_tn_content(title, body, item.source_url):
            evidence_quote = _first_sentence(body) or title
            return _not_relevant_analysis(
                title=title,
                evidence_quote=evidence_quote,
                issue_category="out-of-scope",
            )

        is_party = _has_tvk_reference(text, f"{title}\n{body}")
        is_government = (
            any(k in text for k in _GOVERNMENT_KEYWORDS_EN)
            or any(k in body for k in _GOVERNMENT_KEYWORDS_TA)
        )
        positive = any(k in text for k in _POSITIVE_KEYWORDS)
        negative = any(k in text for k in _NEGATIVE_KEYWORDS)
        people_issue = _detect_people_issue(title, body, item.source_url)
        actors = _extract_political_actors(title, body)
        issue_profile = _public_issue_profile(title, body) if people_issue else None
        issue_severity = issue_profile.severity if issue_profile else Severity.LOW

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
        # TVK runs the administration here, so a TN-government office-holder is a
        # TVK office-holder: government news reflects on TVK, not only the party's
        # own events. A pure public issue with no party/government actor stays
        # neutral on the TVK axis — the people_issue bucket carries it instead.
        tvk_governing_actor = is_party or is_government
        tvk_relevance = (
            GovernmentRelevance.HIGH if tvk_governing_actor
            else GovernmentRelevance.MEDIUM if people_issue
            else GovernmentRelevance.LOW
        )
        tvk_portrayal = stance if tvk_governing_actor else Stance.NEUTRAL
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
            elif people_issue:
                english_people = english_summary

        recommended = ""
        root_cause = ""
        action_type = _action_type(stance=stance, is_party=is_party, people_issue=people_issue)
        if negative:
            if issue_profile:
                recommended = _contextual_recommended_step(
                    issue_profile,
                    title=title,
                    evidence_quote=evidence_quote,
                )
                root_cause = _contextual_root_cause(
                    issue_profile,
                    evidence_quote=evidence_quote,
                )
            else:
                focus = _brief_article_focus(title, evidence_quote)
                recommended = f"Media team: verify the allegation and named actor in '{focus}' before any response."
                root_cause = f"Evidence reports criticism or allegation: {_truncate(evidence_quote, 160)}"
        elif people_issue:
            if issue_profile:
                recommended = _contextual_recommended_step(
                    issue_profile,
                    title=title,
                    evidence_quote=evidence_quote,
                )
                root_cause = _contextual_root_cause(
                    issue_profile,
                    evidence_quote=evidence_quote,
                )
        elif positive:
            focus = _brief_article_focus(title, evidence_quote)
            root_cause = f"Evidence reports a positive development: {_truncate(evidence_quote, 160)}"
            if is_party:
                recommended = f"TVK media team: amplify the reported party action in '{focus}' after source check."
        elif is_government:
            root_cause = f"Evidence reports government activity: {_truncate(evidence_quote, 160)}"

        # Action playbook — only for negative or people-issue rows, where the
        # leadership office needs a ready-to-act brief, not just a one-liner.
        risk_if_ignored = ""
        talking_points: list[str] = []
        verification_checklist: list[str] = []
        draft_statement_original = ""
        draft_statement_english = ""
        if negative or (people_issue and not positive):
            focus = _brief_article_focus(title, evidence_quote)
            if negative and is_party:
                risk_if_ignored = (
                    "Unanswered criticism of a TVK office-holder hardens into the "
                    "dominant narrative and is amplified by rivals."
                )
            elif negative:
                risk_if_ignored = (
                    "An unaddressed allegation spreads unchecked and is used to "
                    "question the government's competence."
                )
            else:
                risk_if_ignored = (
                    "An unmet public grievance escalates locally and erodes trust "
                    "in the administration's responsiveness."
                )
            verification_checklist = [
                "Confirm the named people, location and date against a second source.",
                "Check whether any official or department response already exists.",
                "Assess whether the reported impact is ongoing or already resolved.",
            ]
            if english_summary:
                if is_party:
                    talking_points = [
                        f"Acknowledge the concern raised in '{focus}' without conceding unverified claims.",
                        "Point to the verification under way before any public commitment.",
                    ]
                else:
                    talking_points = [
                        f"Note the issue in '{focus}' and the step being taken to verify it.",
                        "Frame the response around resolving the public's grievance.",
                    ]
                draft_statement_english = (
                    f"We are aware of the report regarding '{focus}'. The facts are being "
                    "verified with the local team, and an appropriate response will follow."
                )
            if first_sentence_orig:
                draft_statement_original = _truncate(first_sentence_orig, 200)

        return AIAnalysis(
            government_relevance=relevance,
            stance_toward_government=stance,
            tvk_relevance=tvk_relevance,
            tvk_portrayal=tvk_portrayal,
            sentiment=sentiment,
            target="TVK leadership" if is_party else "Public matter",
            political_actors=actors,
            department="general",
            district=detect_district(title, body) or "unspecified",
            scheme=None,
            topic=title or "news item",
            issue_category=(
                issue_profile.issue_category if issue_profile
                else ("welfare" if positive else "concern") if (positive or negative)
                else "general"
            ),
            people_issue=people_issue,
            public_issue=issue_profile.public_issue if issue_profile else "",
            severity=Severity.HIGH if negative else issue_severity,
            summary_original=_truncate(first_sentence_orig or "", 200),
            summary_english=english_summary,
            party_action=english_party,
            people_impact=english_people,
            root_cause=root_cause,
            recommended_step=recommended,
            action_owner=issue_profile.action_owner if issue_profile else _action_owner(is_party=is_party, people_issue=people_issue, is_government=is_government),
            action_type=issue_profile.action_type if issue_profile else action_type,
            action_priority=Severity.HIGH if negative else issue_severity,
            risk_if_ignored=risk_if_ignored,
            talking_points=talking_points,
            verification_checklist=verification_checklist,
            draft_statement_original=draft_statement_original,
            draft_statement_english=draft_statement_english,
            positive_points=[_truncate(english_summary, 140)] if positive and english_summary else [],
            negative_points=[_truncate(english_summary, 140)] if negative and english_summary else [],
            evidence_quotes_original=[_truncate(evidence_quote, 240)] if evidence_quote else [],
            evidence_quotes_english=[],
            confidence=0.55,
            needs_human_review=negative
            or (issue_profile is not None and issue_profile.issue_category == "public_safety"),
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
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-5.4-mini",
        *,
        subject: str = "TVK",
        leader: str = "Vijay",
        governing: bool = True,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self._lens = {"subject": subject, "leader": leader, "governing": governing}

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        prompt = build_classification_prompt(item, **getattr(self, "_lens", {}))
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


def build_classification_prompt(
    item: NormalizedItem,
    *,
    subject: str = "TVK",
    leader: str = "Vijay",
    governing: bool = True,
) -> str:
    # The political lens is the only per-tenant variable: which party is "us"
    # and whether it holds government. Defaults reproduce the TVK-governing
    # prompt verbatim. ponytail: same-vertical SaaS — geography/sources/roster
    # are shared TN, only the subject changes.
    if governing:
        roster_clause = (
            f"who hold public office — MLAs, ministers, or the Chief Minister — when those\n"
            f"office-holders belong to {subject}. The official conduct of a {subject} office-holder\n"
            f"counts as {subject} activity, not merely government activity."
        )
        portrayal_def = (
            f"positive = good news for {subject}, {leader}, or a {subject} office-holder\n"
            f"     (scheme delivered, praise, achievement, decisive response); negative =\n"
            f"     failure, broken promise, scandal, criticism, or a governance lapse on a {subject}\n"
            f"     office-holder's watch"
        )
        gov_axis = (
            f"  an institution. When {subject} runs the government the two usually agree, but a\n"
            f"  bureaucratic lapse with no {subject} person named is government-negative while\n"
            f"  {subject.lower()}_portrayal stays neutral; {subject} opposing a harmful decision can be"
        )
    else:
        roster_clause = (
            f"and its spokespeople. {subject} is in OPPOSITION — it does not hold government.\n"
            f"A government failure is not {subject}'s failure; it is often an opening {subject} can use."
        )
        portrayal_def = (
            f"positive = good for {subject} or {leader} (effective opposition, exposing a\n"
            f"     government failure, winning public support, a lapse {subject} can capitalise on); negative =\n"
            f"     a {subject} scandal, infighting, a broken {subject} promise, or {subject} being outmanoeuvred"
        )
        gov_axis = (
            f"  an institution. {subject} does NOT run the government, so a government failure is\n"
            f"  government-negative yet often {subject.lower()}_portrayal POSITIVE; a government win can be"
        )
    return f"""
You are preparing a 30-second daily briefing for the {subject} party leadership office
from a Tamil-Nadu newspaper article. Label every article as a political
intelligence record, not just a sentiment row.

Who counts as {subject}: the party led by {leader}, its organisation and cadre, AND any {subject} members
{roster_clause} People from other
parties are NOT {subject}, even when they hold
office; their conduct never sets tvk_portrayal to positive or negative.

The leader's decision loop is: WHAT happened → WHY → SO WHAT → WHAT NOW.

Two independent judgement axes — score each on its own evidence, never assume
they agree:
- tvk_portrayal is the HEADLINE label that drives the leadership dashboard:
  how this news reflects on {subject}, {leader}, the party, or a {subject} office-holder.
- stance_toward_government tracks the sitting administration's performance as
{gov_axis}
  tvk_portrayal positive while government stance is negative.

Required output:

  1. {subject} PORTRAYAL (tvk_portrayal) — positive / negative / mixed / neutral, the
     headline label. {portrayal_def}; mixed = both signals present; neutral = no {subject} person
     portrayed either way.
  2. GOVERNMENT STANCE (stance_toward_government) — positive / negative / mixed /
     neutral toward the Tamil Nadu administration in office, judged as an
     institution. Keep this field for government-performance tracking.
  3. {subject} RELEVANCE (tvk_relevance) — high when {subject}, {leader}, or a {subject} office-holder
     is directly involved; medium when the story is a public issue or political
     opening {subject} may need to act on; low for background TN context; none for
     out-of-scope.
  4. PEOPLE ISSUE (people_issue) — true when ordinary people face or benefit
     from a public matter: welfare, civic services, jobs, health, education,
     safety, law and order, corruption, price rise, farmers, youth, women, etc.
  5. POLITICAL ACTORS (political_actors) — list explicitly mentioned actors:
     {subject}, {leader}, party members, MLAs, ministers, Chief Minister, departments,
     parties, or named officials. Do not invent names.
  6. PARTY ACTIVITY (party_action) — what {subject} members or the party leadership
     did, well or badly, that this article reports. Empty if not about {subject}.
  7. PEOPLE'S EXPERIENCE (people_impact) — what ordinary people are facing or
     benefiting from in the area the article covers. Empty if not applicable.
  8. PUBLIC ISSUE (public_issue) — one short issue label, e.g. "drinking water
     shortage", "school safety", "farmer compensation delay". Empty if none.
  9. ROOT CAUSE (root_cause) — the underlying WHY: the policy gap, decision,
     event, or structural condition driving what the article reports. Be
     concrete. Empty only if the article gives no causal signal.
  10. RECOMMENDED NEXT STEP (recommended_step) — one concrete action the
     leadership office could consider (visit, statement, relief, internal
     review of a member, public response, fact-finding, etc.). Empty if no
     action is clearly warranted by the evidence.
  11. ACTION ROUTING — action_owner, action_type, action_priority. Route only
      evidence-backed actions. action_type examples: monitor, amplify,
      field_verification, public_statement, internal_review, legal_review,
      escalation, policy_research.
  12. ACTION PLAYBOOK — only for negative or people-issue articles, build a
      ready-to-act brief:
      - risk_if_ignored: what worsens if the leadership does nothing.
      - talking_points: 2–3 short defensible lines the office can say publicly.
      - verification_checklist: 2–3 concrete facts to confirm before acting.
      - draft_statement_original / draft_statement_english: a short, careful
        public statement (2–3 sentences) that concedes nothing unverified.
      Leave ALL playbook fields empty for positive, neutral, or out-of-scope rows.

Tone rules:
- One sentence per briefing field. Each must be ≤ 22 words. No rhetoric.
- Plain English. Avoid "stakeholder", "ecosystem", "leverage", "synergy".
- Do not invent facts. If the article does not warrant a field, leave it "".
- Do not write generic next steps such as "address the concern" or
  "take appropriate action". Name a concrete owner, verification target,
  and action route, or set action_type = "monitor".
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
- public_issue: 2–6 words. Empty if not a public issue.
- root_cause: ≤ 22 words. Cite a concrete driver, not a platitude.
- recommended_step: ≤ 22 words. Concrete and feasible. Empty if no action.
- For public harm or civic issues, recommended_step must name what to verify
  locally before any public statement.
- action_owner: "{subject} leadership office", "District field team",
  "Policy research team", "Media monitoring desk", or a similarly concrete owner.
- action_type: one compact snake_case action category.
- action_priority: low / medium / high / critical.
- risk_if_ignored: ≤ 22 words. Empty unless negative or people-issue.
- talking_points: 2–3 short lines, each ≤ 18 words. Empty unless negative/people-issue.
- verification_checklist: 2–3 short facts to confirm. Empty unless negative/people-issue.
- draft_statement_english / draft_statement_original: 2–3 careful sentences,
  concede nothing unverified. Empty unless negative or people-issue.
- topic: 3–6 word phrase, e.g. "Cauvery water release dispute".
- positive_points / negative_points: 1–2 short bullet phrases each at most.
- evidence_quotes_original: 1–2 short verbatim quotes from the article.

Output schema reminder (the SDK will validate; fields shown for reference):
{{
  "tvk_relevance": "high|medium|low|none",
  "tvk_portrayal": "positive|negative|mixed|neutral",
  "people_issue": "boolean",
  "political_actors": ["string"],
  "public_issue": "string",
  "action_owner": "string",
  "action_type": "string",
  "action_priority": "low|medium|high|critical",
  "risk_if_ignored": "string",
  "talking_points": ["string"],
  "verification_checklist": ["string"],
  "draft_statement_original": "string",
  "draft_statement_english": "string",
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
