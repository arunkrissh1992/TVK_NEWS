"""Tamil Nadu district registry, name normalization, and geo aggregation.

The dashboard's district map is a tile cartogram: each of the 38 districts is a
clickable tile positioned on a coarse grid that approximates the state's real
geography (Chennai north-east, Kanyakumari at the southern tip, Nilgiris on the
western ghats). Tiles need no licensed map asset, render crisply at any size,
and every district is equally tappable.

AI analyses store ``district`` as free text — English, Tamil, or a common
variant spelling — so everything funnels through ``canonical_district`` before
aggregation. Unmatched values (including "unspecified") count toward the
statewide bucket rather than silently vanishing.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any


# canonical name -> (grid column, grid row, short display label)
DISTRICT_TILES: dict[str, tuple[int, int, str]] = {
    "Ariyalur": (5, 5, "Ariyalur"),
    "Chengalpattu": (6, 2, "Chengalpattu"),
    "Chennai": (6, 1, "Chennai"),
    "Coimbatore": (1, 7, "Coimbatore"),
    "Cuddalore": (6, 4, "Cuddalore"),
    "Dharmapuri": (3, 3, "Dharmapuri"),
    "Dindigul": (2, 7, "Dindigul"),
    "Erode": (2, 5, "Erode"),
    "Kallakurichi": (4, 4, "Kallakurichi"),
    "Kancheepuram": (6, 3, "Kanchipuram"),
    "Kanyakumari": (2, 12, "Kanyakumari"),
    "Karur": (3, 6, "Karur"),
    "Krishnagiri": (3, 2, "Krishnagiri"),
    "Madurai": (3, 8, "Madurai"),
    "Mayiladuthurai": (6, 5, "Mayiladuthurai"),
    "Nagapattinam": (6, 6, "Nagapattinam"),
    "Namakkal": (3, 5, "Namakkal"),
    "Nilgiris": (1, 6, "Nilgiris"),
    "Perambalur": (4, 5, "Perambalur"),
    "Pudukkottai": (4, 7, "Pudukkottai"),
    "Ramanathapuram": (4, 9, "Ramnad"),
    "Ranipet": (5, 2, "Ranipet"),
    "Salem": (3, 4, "Salem"),
    "Sivaganga": (4, 8, "Sivaganga"),
    "Tenkasi": (2, 10, "Tenkasi"),
    "Thanjavur": (5, 6, "Thanjavur"),
    "Theni": (2, 8, "Theni"),
    "Thoothukudi": (3, 10, "Thoothukudi"),
    "Tiruchirappalli": (4, 6, "Trichy"),
    "Tirunelveli": (2, 11, "Tirunelveli"),
    "Tirupathur": (4, 3, "Tirupathur"),
    "Tiruppur": (2, 6, "Tiruppur"),
    "Tiruvallur": (5, 1, "Tiruvallur"),
    "Tiruvannamalai": (5, 3, "Tiruvannamalai"),
    "Tiruvarur": (5, 7, "Tiruvarur"),
    "Vellore": (4, 2, "Vellore"),
    "Viluppuram": (5, 4, "Viluppuram"),
    "Virudhunagar": (3, 9, "Virudhunagar"),
}

assert len(DISTRICT_TILES) == 38, "Tamil Nadu has 38 districts"


_EXTRA_ALIASES: dict[str, str] = {
    # Spelling variants the AI commonly emits.
    "kanchipuram": "Kancheepuram",
    "kanniyakumari": "Kanyakumari",
    "trichy": "Tiruchirappalli",
    "tiruchirapalli": "Tiruchirappalli",
    "tiruchchirappalli": "Tiruchirappalli",
    "trichirappalli": "Tiruchirappalli",
    "tuticorin": "Thoothukudi",
    "thoothukkudi": "Thoothukudi",
    "villupuram": "Viluppuram",
    "thiruvallur": "Tiruvallur",
    "thiruvannamalai": "Tiruvannamalai",
    "thiruvarur": "Tiruvarur",
    "sivagangai": "Sivaganga",
    "the nilgiris": "Nilgiris",
    "nilgiri": "Nilgiris",
    "ooty": "Nilgiris",
    "udhagamandalam": "Nilgiris",
    "tirupattur": "Tirupathur",
    "tirupatur": "Tirupathur",
    "virudunagar": "Virudhunagar",
    "ramnad": "Ramanathapuram",
    "mayuram": "Mayiladuthurai",
    "tanjore": "Thanjavur",
    "madras": "Chennai",
    # Tamil names.
    "சென்னை": "Chennai",
    "மதுரை": "Madurai",
    "கோயம்புத்தூர்": "Coimbatore",
    "கோவை": "Coimbatore",
    "திருச்சி": "Tiruchirappalli",
    "திருச்சிராப்பள்ளி": "Tiruchirappalli",
    "சேலம்": "Salem",
    "திருநெல்வேலி": "Tirunelveli",
    "வேலூர்": "Vellore",
    "தஞ்சாவூர்": "Thanjavur",
    "ஈரோடு": "Erode",
    "திருப்பூர்": "Tiruppur",
    "கன்னியாகுமரி": "Kanyakumari",
    "தூத்துக்குடி": "Thoothukudi",
    "காஞ்சிபுரம்": "Kancheepuram",
    "கடலூர்": "Cuddalore",
    "திண்டுக்கல்": "Dindigul",
    "தேனி": "Theni",
    "நாகப்பட்டினம்": "Nagapattinam",
    "கரூர்": "Karur",
    "நாமக்கல்": "Namakkal",
    "நீலகிரி": "Nilgiris",
    "புதுக்கோட்டை": "Pudukkottai",
    "இராமநாதபுரம்": "Ramanathapuram",
    "ராமநாதபுரம்": "Ramanathapuram",
    "சிவகங்கை": "Sivaganga",
    "விழுப்புரம்": "Viluppuram",
    "விருதுநகர்": "Virudhunagar",
    "அரியலூர்": "Ariyalur",
    "பெரம்பலூர்": "Perambalur",
    "தருமபுரி": "Dharmapuri",
    "கிருஷ்ணகிரி": "Krishnagiri",
    "திருவள்ளூர்": "Tiruvallur",
    "திருவண்ணாமலை": "Tiruvannamalai",
    "திருவாரூர்": "Tiruvarur",
    "செங்கல்பட்டு": "Chengalpattu",
    "இராணிப்பேட்டை": "Ranipet",
    "ராணிப்பேட்டை": "Ranipet",
    "கள்ளக்குறிச்சி": "Kallakurichi",
    "தென்காசி": "Tenkasi",
    "மயிலாடுதுறை": "Mayiladuthurai",
    "திருப்பத்தூர்": "Tirupathur",
    # Major towns → their district, so a story that names the town (not the
    # district) still lands on the right tile. Only distinctive place names that
    # don't collide with common English words.
    "hosur": "Krishnagiri",
    "pollachi": "Coimbatore",
    "sulur": "Coimbatore",
    "mettur": "Salem",
    "avadi": "Tiruvallur",
    "ambattur": "Chennai",
    "tambaram": "Chengalpattu",
    "pallavaram": "Chengalpattu",
    "sriperumbudur": "Kancheepuram",
    "nagercoil": "Kanyakumari",
    "karaikudi": "Sivaganga",
    "sivakasi": "Virudhunagar",
    "rajapalayam": "Virudhunagar",
    "kumbakonam": "Thanjavur",
    "neyveli": "Cuddalore",
    "chidambaram": "Cuddalore",
    "velankanni": "Nagapattinam",
    "gudiyatham": "Vellore",
    "arakkonam": "Ranipet",
    "dharapuram": "Tiruppur",
    "udumalpet": "Tiruppur",
    "palani": "Dindigul",
    "kovilpatti": "Thoothukudi",
    "gobichettipalayam": "Erode",
    "karur town": "Karur",
}

_ALIASES: dict[str, str] = {name.lower(): name for name in DISTRICT_TILES}
_ALIASES.update(_EXTRA_ALIASES)

_NOT_A_DISTRICT = {"", "unspecified", "unknown", "none", "n/a", "statewide", "tamil nadu", "tamilnadu"}


def canonical_district(value: str | None) -> str | None:
    """Map a free-text district mention to its canonical name, or None."""
    if not value:
        return None
    cleaned = value.strip()
    # Drop common suffixes in either language.
    for suffix in (" district", " District", " மாவட்டம்"):
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    key = cleaned.lower()
    if key in _NOT_A_DISTRICT:
        return None
    return _ALIASES.get(key) or _ALIASES.get(key.replace(".", "").strip())


# Aliases that are also ordinary English words — match only their capitalised
# place-name form, never the lowercase verb ("erodes public trust" ≠ Erode).
_AMBIGUOUS_EN_ALIASES = {"erode"}

_EN_ALIAS_PATTERN: re.Pattern[str] | None = None


def _en_alias_pattern() -> re.Pattern[str]:
    """One word-boundary alternation over every ASCII alias, longest first.

    Word boundaries are load-bearing: plain substring search once matched
    'theni' inside "strengthening" and 'salem' inside "Jerusalem", flagging
    international wire stories as Tamil Nadu coverage.
    """
    global _EN_ALIAS_PATTERN
    if _EN_ALIAS_PATTERN is None:
        names = sorted((a for a in _ALIASES if a.isascii()), key=len, reverse=True)
        _EN_ALIAS_PATTERN = re.compile(
            r"\b(" + "|".join(re.escape(name) for name in names) + r")\b",
            re.IGNORECASE,
        )
    return _EN_ALIAS_PATTERN


def detect_district(*texts: str | None) -> str | None:
    """Find the first Tamil Nadu district mentioned in the given texts.

    Texts are scanned in order (pass the title first so headline mentions
    win); within a text the earliest mention wins. Returns the canonical
    district name or None.
    """
    for text in texts:
        if not text:
            continue
        best: tuple[int, str] | None = None
        for match in _en_alias_pattern().finditer(text):
            alias = match.group(1).lower()
            if alias in _AMBIGUOUS_EN_ALIASES and not match.group(1)[0].isupper():
                continue
            if best is None or match.start() < best[0]:
                best = (match.start(), _ALIASES[alias])
        for alias, canonical in _ALIASES.items():
            if alias.isascii():
                continue
            position = text.find(alias)
            if position != -1 and (best is None or position < best[0]):
                best = (position, canonical)
        if best is not None:
            return best[1]
    return None


_CATEGORY_KEYS = ("positive", "negative", "mixed", "people", "neutral")

# Which single signal colours a district tile when several are present.
_DOMINANT_ORDER = ("negative", "people", "mixed", "positive")


def _dominant(counts: dict[str, int]) -> str:
    for key in _DOMINANT_ORDER:
        if counts.get(key, 0) > 0:
            return key
    return "quiet"


def summarize_by_district(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate narrative-card payloads per canonical district.

    Returns every one of the 38 districts (zero-count tiles render grey), plus
    a statewide bucket for items whose district didn't resolve.
    """
    per: dict[str, dict[str, Any]] = {
        name: {
            "district": name,
            "short": short,
            "col": col,
            "row": row,
            "total": 0,
            "issues": Counter(),
            **{key: 0 for key in _CATEGORY_KEYS},
        }
        for name, (col, row, short) in DISTRICT_TILES.items()
    }
    unmapped = 0
    for item in items:
        name = canonical_district(item.get("district"))
        if name is None:
            unmapped += 1
            continue
        bucket = per[name]
        bucket["total"] += 1
        category = item.get("display_category") or "neutral"
        if category in _CATEGORY_KEYS:
            bucket[category] += 1
        issue = (item.get("public_issue") or "").strip()
        if issue:
            bucket["issues"][issue] += 1

    tiles: list[dict[str, Any]] = []
    for name in sorted(per):
        bucket = per[name]
        issues = [
            {"issue": issue, "count": count}
            for issue, count in bucket["issues"].most_common(3)
        ]
        tiles.append(
            {
                **{k: bucket[k] for k in ("district", "short", "col", "row", "total", *_CATEGORY_KEYS)},
                "dominant": _dominant(bucket) if bucket["total"] else "quiet",
                "top_issues": issues,
            }
        )
    return {
        "tiles": tiles,
        "unmapped_total": unmapped,
        "mapped_total": sum(t["total"] for t in tiles),
    }


# ---------------------------------------------------------------------------
# Department name normalization
# ---------------------------------------------------------------------------
#
# The Gemma LLM (src/tnmi/local_llm.py) emits a freeform ``department`` string
# per analysis, so the dashboard's Departments rail fragments into near-
# duplicates: "Tamil Nadu Government" vs "Government of Tamil Nadu", or the
# overlapping "Police" / "Law enforcement" / "Law/order". Everything funnels
# through ``canonical_department`` before the rail aggregates, collapsing those
# variants onto one stable, readable name. The canonical names mirror the
# readable departments in ``responsibility.py`` so the rail and the per-card
# "who should act" chips speak the same vocabulary.

# Placeholders the LLM emits when it can't name a department — not real owners,
# so they drop out of the rail (mirrors ``canonical_district`` returning None
# for "statewide"/"unspecified").
_GENERIC_DEPARTMENTS = {
    "", "general", "unspecified", "unknown", "none", "n/a", "na", "nil",
    "various", "multiple", "other", "others", "misc", "miscellaneous",
    "department", "dept", "ministry",
}

# Administrative affixes stripped before matching so "Department of Health",
# "Health Department" and "Health Dept." all reduce to the same core text.
_DEPARTMENT_PREFIXES = (
    "department of ", "department for ", "dept of ", "dept. of ",
    "ministry of ", "directorate general of ", "directorate of ",
    "office of the ", "office of ",
)
_DEPARTMENT_SUFFIXES = (
    " department", " dept.", " dept", " ministry", " directorate",
)

# Keyword → canonical department. First match wins, so the more specific rows
# come first and the catch-all "government" bucket comes last (mirrors the
# ordering of ``responsibility._DEPARTMENT_RULES``). Matched as substrings
# against the cleaned, lowercased department text.
_DEPARTMENT_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("drinking water", "metro water", "metrowater", "water supply",
      "water board", "twad", "maws"),
     "Municipal Admin & Water Supply"),
    (("sewage", "drainage", "garbage", "sanitation", "corporation", "municipal",
      "municipality", "town panchayat", "street light", "urban local"),
     "Municipal Administration"),
    (("electricity", "power", "tangedco", "tneb", "transformer", "energy"),
     "Energy / TANGEDCO"),
    (("road", "pothole", "traffic", "transport", "highway", "bridge", "bus "),
     "Highways & Transport"),
    (("hospital", "health", "medical", "clinic", "phc", "doctor"),
     "Health & Family Welfare"),
    (("school", "education", "student", "teacher", "college", "exam"),
     "School Education"),
    (("police", "law enforcement", "law and order", "law/order", "law & order",
      "law ", "crime", "home (police)", "home department"),
     "Home (Police)"),
    (("fire", "rescue", "disaster", "flood", "relief", "revenue"),
     "Revenue & Disaster Management"),
    (("labour", "labor", "employment", "worker", "wage", "factory"),
     "Labour & Employment"),
    (("agricultur", "farmer", "crop", "irrigation", "cauvery", "mettur",
      "paddy", "horticultur"),
     "Agriculture & Farmers' Welfare"),
    (("ration", "pds", "civil supplies", "fair price", "co-operation",
      "cooperation", "consumer"),
     "Co-operation, Food & Consumer Protection"),
    (("pension", "welfare", "widow", "disabled", "differently abled", "old age"),
     "Social Welfare"),
    (("forest", "environment", "pollution", "wildlife"),
     "Environment & Forests"),
    (("tourism", "tourist"),
     "Tourism"),
    (("rural development", "panchayat raj", "village panchayat", "rural "),
     "Rural Development"),
    # Catch-all: a bare state-government reference. Comes last so any specific
    # department above wins over "Tamil Nadu Government" / "secretariat".
    (("government", "govt", "secretariat", "cabinet", "minister", "cmo",
      "administration", "governance"),
     "Government of Tamil Nadu"),
)


def _clean_department(value: str) -> str:
    """Collapse whitespace and strip administrative affixes from a department."""
    cleaned = " ".join(value.split())
    lowered = cleaned.lower()
    for prefix in _DEPARTMENT_PREFIXES:
        if lowered.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            lowered = cleaned.lower()
            break
    for suffix in _DEPARTMENT_SUFFIXES:
        if lowered.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            break
    return cleaned


def canonical_department(value: str | None) -> str | None:
    """Map a free-text department mention to a stable canonical name, or None.

    Returns None for empty or generic placeholders ("general", "unspecified")
    so the Departments rail lists only real owners. Recognised departments
    collapse onto the readable names used in ``responsibility.py``; unrecognised
    ones are kept but cleaned (affixes stripped, title-cased) so distinct
    free-text variants of the same office — "Sports Department", "department of
    sports" — still group together.
    """
    if not value:
        return None
    cleaned = _clean_department(value)
    key = cleaned.lower()
    if key in _GENERIC_DEPARTMENTS:
        return None
    for keywords, department in _DEPARTMENT_RULES:
        if any(keyword in key for keyword in keywords):
            return department
    # Unrecognised but real: keep it, normalising case so "SPORTS" and "sports"
    # don't split into two rails.
    return cleaned.title() if cleaned.islower() or cleaned.isupper() else cleaned


def summarize_by_department(
    items: list[dict[str, Any]], *, limit: int = 14
) -> list[dict[str, Any]]:
    """Aggregate narrative-card payloads per canonical department, busiest first.

    Free-text department names are funnelled through ``canonical_department`` so
    near-duplicates collapse into one row. The ``department`` field on each row
    is the canonical name the cards also carry (see ``list_latest_items`` →
    ``department_canonical``), so the dashboard's ``data-department`` rail filter
    matches both sides.
    """
    per: dict[str, dict[str, Any]] = {}
    for item in items:
        canonical = canonical_department(item.get("department"))
        if canonical is None:
            continue
        bucket = per.get(canonical)
        if bucket is None:
            bucket = per[canonical] = {
                "department": canonical,
                "label": canonical,
                "total": 0,
                **{k: 0 for k in _CATEGORY_KEYS},
            }
        bucket["total"] += 1
        category = item.get("display_category") or "neutral"
        if category in _CATEGORY_KEYS:
            bucket[category] += 1

    ranked = sorted(per.values(), key=lambda b: (-b["total"], b["department"]))
    for bucket in ranked:
        bucket["dominant"] = _dominant(bucket)
    return ranked[:limit]
