"""Map a news item to who in government should act on it.

The briefing's job is not just to flag a problem but to point at the office that
owns it: the responsible **government department**, the **district** it sits in
(Collector + the area's MLAs), and whether it is serious enough to escalate to
the **Chief Minister**. Article tagging is at district granularity, not
constituency, so we name the department and the district's representation rather
than inventing a single accountable MLA.
"""

from __future__ import annotations

from typing import Any

from tnmi.mla import mlas_by_district

# Keyword → readable Tamil Nadu government department. First match wins, so the
# more specific rows come first. Matched against the article's issue text.
_DEPARTMENT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("drinking water", "metro water", "water shortage", "water supply", "tap", "tank", "borewell"),
     "Municipal Admin & Water Supply"),
    (("sewage", "drainage", "garbage", "sanitation", "toilet", "street light", "encroach"),
     "Municipal Administration"),
    (("electricity", "power cut", "powercut", "tangedco", "transformer", "current ", "eb office"),
     "Energy / TANGEDCO"),
    (("road", "pothole", "traffic", "bus ", "transport", "highway", "bridge", "culvert"),
     "Highways & Transport"),
    (("hospital", "health", "medical", "clinic", "phc", "ambulance", "doctor", "dengue", "fever"),
     "Health & Family Welfare"),
    (("school", "education", "student", "college", "teacher", "exam", "scholarship"),
     "School Education"),
    (("police", "crime", "murder", "assault", "theft", "harass", "law and order", "kidnap"),
     "Home (Police)"),
    (("fire", "accident", "rescue", "drown", "collapse", "flood", "disaster", "relief"),
     "Revenue & Disaster Management"),
    (("job", "jobs", "employment", "wage", "labour", "worker", "factory", "strike"),
     "Labour & Employment"),
    (("farmer", "agriculture", "crop", "irrigation", "cauvery", "mettur", "dam", "paddy"),
     "Agriculture & Farmers' Welfare"),
    (("ration", "pds", "civil supplies", "fair price"),
     "Co-operation, Food & Consumer Protection"),
    (("pension", "welfare", "widow", "disabled", "old age", "differently abled"),
     "Social Welfare"),
)

_GENERIC_DEPARTMENTS = {"", "general", "unspecified", "unknown", "n/a", "none"}
_NON_DISTRICTS = {"", "unspecified", "unknown", "statewide", "general", "none"}
_PRIORITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def _match_department(text: str) -> str | None:
    lowered = text.lower()
    for keywords, department in _DEPARTMENT_RULES:
        if any(keyword in lowered for keyword in keywords):
            return department
    return None


def resolve_responsibility(
    item: dict[str, Any],
    district_mlas: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Return who should act on ``item``: responsible department, the district
    administration + its MLAs, and whether to escalate to the CM.

    ``item`` is a briefing card payload (see ``list_latest_items``). ``actionable``
    is True only when there is a real owner to show, so positive/quiet coverage
    does not get a needless "who acts" row.
    """
    if district_mlas is None:
        district_mlas = mlas_by_district()

    # 1) Responsible government department — infer from the issue text, falling
    #    back to an explicit, non-generic department on the analysis.
    issue_text = " ".join(
        str(item.get(key) or "")
        for key in ("public_issue", "department", "title", "summary", "root_cause", "people_impact")
    )
    department = _match_department(issue_text)
    if not department:
        explicit = (item.get("department") or "").strip()
        if explicit.lower() not in _GENERIC_DEPARTMENTS:
            department = explicit.title()

    # 2) District / local tier.
    canonical = (item.get("district_canonical") or "").strip()
    raw_district = (item.get("district") or "").strip()
    has_district = bool(canonical) and raw_district.lower() not in _NON_DISTRICTS
    mlas = list(district_mlas.get(canonical, [])) if has_district else []

    # 3) Escalation — high/critical problems are CM-level.
    priority = (item.get("action_priority") or item.get("severity") or "").lower()
    is_problem = bool(item.get("people_issue")) or item.get("portrayal_kind") in {"negative", "people"}
    escalate_cm = is_problem and _PRIORITY_RANK.get(priority, 0) >= 3

    actionable = bool(department or (has_district and is_problem) or escalate_cm)

    return {
        "actionable": actionable,
        "department": department or "",
        "district": raw_district if has_district else "",
        "district_canonical": canonical if has_district else "",
        "collector": f"{raw_district} Collector" if has_district else "",
        "mlas": mlas[:6],
        "mla_count": len(mlas),
        "escalate_cm": escalate_cm,
        "priority": priority,
    }
