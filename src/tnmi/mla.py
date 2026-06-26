"""MLA roster — who represents each constituency, grouped per district.

The roster lives in ``configs/mla_roster.json`` and is DATA, not code: it
records a specific assembly (name + election year are displayed in the UI so
nobody mistakes its vintage). When a new assembly is sworn in — or to model a
scenario — replace the file and restart; regenerate from source with
``scripts/build_mla_roster.py``.
"""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from tnmi.districts import canonical_district

_DEFAULT_ROSTER_PATH = Path(__file__).resolve().parents[2] / "configs" / "mla_roster.json"

# Full party name -> (short code, css color key). The css key doubles as the
# class suffix party chips use in the dashboard.
PARTY_STYLE: dict[str, tuple[str, str]] = {
    "Dravida Munnetra Kazhagam": ("DMK", "dmk"),
    "All India Anna Dravida Munnetra Kazhagam": ("AIADMK", "aiadmk"),
    "Indian National Congress": ("INC", "inc"),
    "Bharatiya Janata Party": ("BJP", "bjp"),
    "Pattali Makkal Katchi": ("PMK", "pmk"),
    "Viduthalai Chiruthaigal Katchi": ("VCK", "vck"),
    "Marumalarchi Dravida Munnetra Kazhagam": ("MDMK", "mdmk"),
    "Communist Party of India": ("CPI", "cpi"),
    "Communist Party of India (Marxist)": ("CPI(M)", "cpim"),
    "Tamilaga Vettri Kazhagam": ("TVK", "tvk"),
    "Naam Tamilar Katchi": ("NTK", "ntk"),
    "Indian Union Muslim League": ("IUML", "iuml"),
    "Desiya Murpokku Dravida Kazhagam": ("DMDK", "dmdk"),
    "Amma Makkal Munnetra Kazhagam": ("AMMK", "ammk"),
    "Vacant": ("Vacant", "vacant"),
}


def party_short(party: str) -> str:
    return PARTY_STYLE.get(party, (party, "other"))[0]


def party_css(party: str) -> str:
    return PARTY_STYLE.get(party, (party, "other"))[1]


@lru_cache(maxsize=4)
def load_roster(path: str | None = None) -> dict[str, Any]:
    """Load the roster file; returns an empty roster if the file is absent so
    the dashboard degrades gracefully on deployments without one."""
    roster_path = Path(path) if path else _DEFAULT_ROSTER_PATH
    if not roster_path.exists():
        return {"assembly": "", "elected": "", "members": []}
    return json.loads(roster_path.read_text(encoding="utf-8"))


def roster_label(roster: dict[str, Any] | None = None) -> str:
    roster = roster or load_roster()
    if not roster.get("members"):
        return ""
    elected = f", elected {roster['elected']}" if roster.get("elected") else ""
    return f"{roster.get('assembly', 'Assembly')}{elected}"


def mlas_by_district(roster: dict[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Group members by canonical district, ready for the district panel."""
    roster = roster or load_roster()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for member in roster.get("members", []):
        district = canonical_district(member.get("district"))
        if district is None:
            continue
        grouped.setdefault(district, []).append(
            {
                "no": member["no"],
                "constituency": member["constituency"],
                "mla": member["mla"],
                "party": member["party"],
                "party_short": party_short(member["party"]),
                "party_css": party_css(member["party"]),
            }
        )
    for members in grouped.values():
        members.sort(key=lambda m: m["no"])
    return grouped


def party_seat_counts(roster: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Statewide seats per party, biggest first — the default panel's tally."""
    roster = roster or load_roster()
    counts = Counter(member["party"] for member in roster.get("members", []))
    return [
        {
            "party": party,
            "party_short": party_short(party),
            "party_css": party_css(party),
            "seats": seats,
        }
        for party, seats in counts.most_common()
    ]
