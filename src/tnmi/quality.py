from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tnmi.ai import (
    _has_people_issue,
    _has_tvk_reference,
    _public_issue_profile,
)
from tnmi.storage import AIAnalysisRecord, RawItemRecord


_GENERIC_NEXT_STEP_MARKERS = (
    "address the concern",
    "appropriate department",
    "take appropriate action",
    "necessary action",
    "look into the matter",
    "monitor the situation",
)
_SERIOUS_ISSUE_CATEGORIES = {"public_safety"}
_HIGH_SEVERITY = {"high", "critical"}
_PUBLIC_IMPACT_MARKERS = (
    "பொதுமக்கள்", "மக்கள் கூற", "மக்கள் புகார்", "பெற்றோர்", "புகார்", "பாதிப்பு", "பாதுகாப்பு",
    "வெளியேற்றப்பட்ட", "காயம்", "மரணம்", "உயிரிழப்பு", "தீ விபத்து", "தீவிபத்து",
    "விபத்து", "குடிநீர்", "மின்சாரம்", "சாலை", "கழிவுநீர்", "மருத்துவ",
    "வேலை", "விவசாய", "public", "residents", "parents", "complaint",
    "affected", "impact", "safety", "evacuated", "injured", "injury", "death",
    "fire", "accident", "water", "power", "road", "hospital", "jobs", "farmers",
)
_EDUCATION_CEREMONY_MARKERS = (
    "திறப்பு விழா", "திறந்து வைத்த", "ரிப்பன்", "குத்து விளக்கு", "protocol",
    "ceremony", "inauguration", "ribbon",
)


@dataclass(frozen=True)
class AnalysisQualityIssue:
    raw_item_id: int
    analysis_id: int
    title: str
    source_name: str
    issue_code: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "raw_item_id": self.raw_item_id,
            "analysis_id": self.analysis_id,
            "title": self.title,
            "source_name": self.source_name,
            "issue_code": self.issue_code,
            "detail": self.detail,
        }


def audit_analysis_quality(session: Session, *, limit: int = 100) -> list[dict[str, Any]]:
    """Scan latest analyses for contradictions that should be reviewed.

    This is a deterministic guardrail around the AI. It does not replace the
    model; it catches obvious product failures such as a school fire being
    treated as ordinary neutral coverage, or a next step that is too generic
    to act on.
    """
    rows = session.execute(
        select(RawItemRecord, AIAnalysisRecord)
        .join(AIAnalysisRecord, AIAnalysisRecord.raw_item_id == RawItemRecord.id)
        .order_by(RawItemRecord.id, AIAnalysisRecord.created_at.desc(), AIAnalysisRecord.id.desc())
    ).all()
    latest: dict[int, tuple[RawItemRecord, AIAnalysisRecord]] = {}
    for raw, analysis in rows:
        existing = latest.get(raw.id)
        if existing is None:
            latest[raw.id] = (raw, analysis)
            continue
        _existing_raw, existing_analysis = existing
        if existing_analysis.model_name == "mock" and analysis.model_name != "mock":
            latest[raw.id] = (raw, analysis)

    issues: list[AnalysisQualityIssue] = []
    for raw, analysis in latest.values():
        issues.extend(_audit_one(raw, analysis))
        if len(issues) >= limit:
            break
    return [issue.as_dict() for issue in issues[: max(0, limit)]]


def _audit_one(raw: RawItemRecord, analysis: AIAnalysisRecord) -> list[AnalysisQualityIssue]:
    if (analysis.government_relevance or "").lower() == "none":
        return []
    if (analysis.issue_category or "").lower() in {"out-of-scope", "listing"}:
        return []

    title = raw.title or raw.source_url
    body = raw.clean_text_original or raw.raw_text_original or ""
    text_original = f"{title}\n{body}"
    text_lower = text_original.lower()
    has_people_signal = _has_people_issue(text_lower, body)
    profile = _public_issue_profile(title, body) if has_people_signal else None
    expected_people_issue = _expects_people_issue(
        profile=profile,
        text_lower=text_lower,
        text_original=text_original,
    )

    issues: list[AnalysisQualityIssue] = []
    if expected_people_issue and not analysis.people_issue:
        issues.append(
            _issue(
                raw,
                analysis,
                "people_issue_missing",
                "Article text contains people/public-safety signals but analysis.people_issue is false.",
            )
        )
    if profile and profile.issue_category in _SERIOUS_ISSUE_CATEGORIES:
        if (analysis.severity or "").lower() not in _HIGH_SEVERITY:
            issues.append(
                _issue(
                    raw,
                    analysis,
                    "serious_issue_low_severity",
                    f"{profile.public_issue} should be high or critical severity.",
                )
            )
        if not analysis.needs_human_review:
            issues.append(
                _issue(
                    raw,
                    analysis,
                    "serious_issue_not_reviewed",
                    f"{profile.public_issue} should require human review before action.",
                )
            )
    if expected_people_issue and _is_generic_next_step(analysis.recommended_step):
        issues.append(
            _issue(
                raw,
                analysis,
                "generic_next_step",
                "People-issue next step is too generic to execute.",
            )
        )
    if expected_people_issue and not (analysis.action_owner or "").strip():
        issues.append(
            _issue(
                raw,
                analysis,
                "missing_action_owner",
                "People-issue analysis needs a concrete owner such as District field team.",
            )
        )
    if _has_tvk_reference(text_lower, text_original) and (analysis.tvk_relevance or "").lower() != "high":
        issues.append(
            _issue(
                raw,
                analysis,
                "tvk_relevance_missing",
                "Article mentions TVK/Vijay but analysis.tvk_relevance is not high.",
            )
        )
    return issues


def _expects_people_issue(
    *,
    profile: Any,
    text_lower: str,
    text_original: str,
) -> bool:
    if not profile or profile.issue_category == "concern":
        return False
    if profile.issue_category in _SERIOUS_ISSUE_CATEGORIES:
        return True
    has_public_impact = _contains_marker(text_lower, text_original, _PUBLIC_IMPACT_MARKERS)
    ceremony_only = (
        profile.issue_category == "education"
        and _contains_marker(text_lower, text_original, _EDUCATION_CEREMONY_MARKERS)
        and not has_public_impact
    )
    return has_public_impact and not ceremony_only


def _contains_marker(text_lower: str, text_original: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text_lower or marker in text_original for marker in markers)


def _is_generic_next_step(value: str | None) -> bool:
    normalized = (value or "").strip().lower()
    if not normalized:
        return True
    return any(marker in normalized for marker in _GENERIC_NEXT_STEP_MARKERS)


def _issue(
    raw: RawItemRecord,
    analysis: AIAnalysisRecord,
    issue_code: str,
    detail: str,
) -> AnalysisQualityIssue:
    return AnalysisQualityIssue(
        raw_item_id=raw.id,
        analysis_id=analysis.id,
        title=raw.title or raw.source_url,
        source_name=raw.source_name,
        issue_code=issue_code,
        detail=detail,
    )
