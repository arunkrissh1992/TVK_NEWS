"""Curate the latest local analyses with demo-ready operational details.

This is a pragmatic bridge for demos when the semantic LLM is not yet available
or only a subset of articles has been re-analysed. It does not change stance
labels. It enriches routing fields with evidence-specific district, department,
priority and next-step text so the dashboard reads like an operational monitor.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sqlalchemy import select

from tnmi.ai import PROMPT_VERSION, _first_sentence, _truncate
from tnmi.config import Settings
from tnmi.storage import AIAnalysisRecord, RawItemRecord, create_session_factory, init_db


DISTRICT_TERMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Chennai", ("chennai", "சென்னை", "பள்ளிக்கரணை", "tambaram", "தாம்பரம்")),
    ("Madurai", ("madurai", "மதுரை")),
    ("Coimbatore", ("coimbatore", "கோவை", "கோயம்புத்தூர்")),
    ("Thoothukudi", ("thoothukudi", "tuticorin", "தூத்துக்குடி")),
    ("Tirunelveli", ("tirunelveli", "நெல்லை", "திருநெல்வேலி")),
    ("Salem", ("salem", "சேலம்")),
    ("Trichy", ("trichy", "tiruchirappalli", "திருச்சி")),
    ("Erode", ("erode", "ஈரோடு")),
    ("Vellore", ("vellore", "வேலூர்")),
    ("Kancheepuram", ("kancheepuram", "kanchipuram", "காஞ்சிபுரம்")),
    ("Tiruvallur", ("tiruvallur", "திருவள்ளூர்")),
    ("Cuddalore", ("cuddalore", "கடலூர்")),
    ("Villupuram", ("villupuram", "விழுப்புரம்")),
    ("Tiruppur", ("tiruppur", "திருப்பூர்")),
    ("Thanjavur", ("thanjavur", "தஞ்சாவூர்")),
    ("Dindigul", ("dindigul", "திண்டுக்கல்")),
    ("Kanniyakumari", ("kanyakumari", "kanniyakumari", "கன்னியாகுமரி")),
)

DEPARTMENT_BY_CATEGORY = {
    "public_safety": "Public safety / fire and rescue",
    "civic_services": "Local administration",
    "health": "Health",
    "education": "School education",
    "livelihood": "Labour and youth welfare",
    "agriculture": "Agriculture",
    "welfare": "Welfare",
}


def _contains(text_lower: str, text_original: str, terms: tuple[str, ...]) -> bool:
    return any(term in text_lower or term in text_original for term in terms)


def _explicit_tvk(title: str, body: str) -> bool:
    evidence = _first_sentence(body) or ""
    text_lower = f"{title}\n{evidence}".lower()
    text_original = f"{title}\n{evidence}"
    english_match = re.search(r"\b(tvk|tamilaga vettri|vijay)\b", text_lower) is not None
    tamil_match = _contains(text_lower, text_original, ("தவெக", "தமிழக வெற்றி", "விஜய்"))
    return english_match or tamil_match


def _school_context(title: str, body: str) -> bool:
    text_lower = f"{title}\n{body}".lower()
    text_original = f"{title}\n{body}"
    for false_school_place in ("pallikaranai", "பள்ளிக்கரணை"):
        text_lower = text_lower.replace(false_school_place, "")
        text_original = text_original.replace(false_school_place, "")
    return _contains(
        text_lower,
        text_original,
        ("school", "student", "students", "பள்ளி", "மாணவர்", "மாணவி"),
    )


def _district_for(title: str, body: str) -> str:
    text_lower = f"{title}\n{body}".lower()
    text_original = f"{title}\n{body}"
    for district, terms in DISTRICT_TERMS:
        if _contains(text_lower, text_original, terms):
            return district
    return "Tamil Nadu"


def _department_for(analysis: AIAnalysisRecord, title: str, body: str) -> str:
    category = (analysis.issue_category or "").lower()
    issue = (analysis.public_issue or "").lower()
    text = f"{title}\n{body}".lower()
    if _explicit_tvk(title, body):
        return "TVK party affairs"
    if "school" in issue or _school_context(title, body):
        return "School education"
    if "fire" in text or "தீ விபத்து" in f"{title}\n{body}" or "தீவிபத்து" in f"{title}\n{body}":
        return "Public safety / fire and rescue"
    if "train" in text or "lorry" in text or "road" in issue:
        return "Transport / road safety"
    return DEPARTMENT_BY_CATEGORY.get(category, analysis.department or "Media monitoring")


def _owner_for(analysis: AIAnalysisRecord, district: str, *, explicit_tvk: bool = False) -> str:
    if explicit_tvk:
        return "TVK leadership office"
    if analysis.people_issue:
        return f"{district} district field team"
    if (analysis.government_relevance or "").lower() in {"high", "medium"}:
        return "Policy research team"
    return "Media monitoring desk"


def _priority_for(analysis: AIAnalysisRecord) -> str:
    category = (analysis.issue_category or "").lower()
    severity = (analysis.severity or "").lower()
    if category == "public_safety" or severity in {"critical", "high"}:
        return "high"
    if analysis.people_issue or severity == "medium":
        return "medium"
    return "low"


def _action_type_for(analysis: AIAnalysisRecord) -> str:
    if (analysis.tvk_portrayal or "").lower() == "negative":
        return "internal_review"
    if (analysis.tvk_portrayal or "").lower() == "positive":
        return "amplify"
    if analysis.people_issue:
        return "field_verification"
    if (analysis.government_relevance or "").lower() in {"high", "medium"}:
        return "policy_research"
    return "monitor"


def _verification_targets(analysis: AIAnalysisRecord) -> str:
    category = (analysis.issue_category or "").lower()
    issue = (analysis.public_issue or "").lower()
    if "school" in issue:
        return "student safety, parent concerns, school response and department action"
    if category == "public_safety":
        return "casualties, safety risk, root cause and fire/police/civic response"
    if "water" in issue:
        return "affected streets, water supply timeline and local-body response"
    if "electricity" in issue:
        return "outage details, affected homes and electricity-board response"
    if "road" in issue or category == "transport":
        return "exact location, photos, commuter impact and authority response"
    if category == "health":
        return "patient impact, facility response and health-department action"
    if category == "education":
        return "student or parent impact, school facts and education-department action"
    if category == "agriculture":
        return "affected farmers, crop impact and district authority response"
    if category == "livelihood":
        return "affected workers or youth, scale of impact and available relief route"
    return "affected people, location, evidence and official response"


def _curated_lines(analysis: AIAnalysisRecord, item: RawItemRecord, district: str) -> tuple[str, str]:
    title = re.sub(r"\s+", " ", (item.title or "").strip())
    evidence = _first_sentence(item.clean_text_original or item.raw_text_original or "") or title
    evidence = re.sub(r"\s+", " ", evidence.strip())
    evidence_short = _truncate(evidence, 170)
    title_short = _truncate(title or evidence, 96)
    owner = _owner_for(analysis, district, explicit_tvk=_explicit_tvk(title, item.clean_text_original or item.raw_text_original or ""))

    if analysis.people_issue:
        root = f"Evidence shows a {analysis.public_issue or 'people issue'} in {district}: {evidence_short}"
        step = (
            f"{owner}: verify {_verification_targets(analysis)} for '{title_short}' "
            "before statement, visit or relief support."
        )
        return root, step

    portrayal = (analysis.tvk_portrayal or "").lower()
    if portrayal == "positive":
        root = f"Evidence shows favourable TVK coverage: {evidence_short}"
        step = f"{owner}: source-check '{title_short}' and amplify only the verified TVK action."
        return root, step
    if portrayal == "negative":
        root = f"Evidence shows a negative TVK allegation or criticism: {evidence_short}"
        step = f"{owner}: verify the allegation, named actor and evidence in '{title_short}' before response."
        return root, step
    if portrayal == "mixed":
        root = f"Evidence carries mixed TVK signals: {evidence_short}"
        step = f"{owner}: compare claims in '{title_short}' and prepare a measured response only after verification."
        return root, step

    if (analysis.government_relevance or "").lower() in {"high", "medium"}:
        root = f"Evidence reports government activity in {district}: {evidence_short}"
        return root, "Policy research team: monitor for TVK relevance and prepare notes only if public impact is confirmed."

    return "", ""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    session_factory = create_session_factory(Settings().database_url)
    init_db(session_factory)
    updated = 0
    with session_factory() as session:
        query = (
            select(AIAnalysisRecord, RawItemRecord)
            .join(RawItemRecord, RawItemRecord.id == AIAnalysisRecord.raw_item_id)
            .where(AIAnalysisRecord.prompt_version == args.prompt_version)
            .order_by(RawItemRecord.ingested_at.desc(), RawItemRecord.id.desc())
        )
        if args.limit:
            query = query.limit(args.limit)
        for analysis, item in session.execute(query):
            body = item.clean_text_original or item.raw_text_original or ""
            title = item.title or ""
            district = _district_for(title, body)
            department = _department_for(analysis, title, body)
            analysis.district = district
            analysis.department = department
            explicit_tvk = _explicit_tvk(title, body)
            analysis.action_owner = _owner_for(analysis, district, explicit_tvk=explicit_tvk)
            analysis.action_type = _action_type_for(analysis)
            analysis.action_priority = _priority_for(analysis)
            if analysis.action_priority == "high":
                analysis.needs_human_review = True
            root, step = _curated_lines(analysis, item, district)
            if root:
                analysis.root_cause = root
            if step:
                analysis.recommended_step = step
            updated += 1
        session.commit()
    print(f"updated={updated} prompt_version={args.prompt_version}")


if __name__ == "__main__":
    main()
