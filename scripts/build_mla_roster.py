"""Generate configs/mla_roster.json from the Wikipedia members table.

Parses the "Members of Legislative Assembly" wikitable of a Tamil Nadu
Legislative Assembly page — constituency number, name, district, MLA, party —
handling the table's rowspan-folded district and party cells.

Usage:
  .venv/bin/python scripts/build_mla_roster.py [wiki_json] [assembly_label] [elected_year]

Defaults target the CURRENT assembly (17th, elected 2026). Download first:
  curl -sL "https://en.wikipedia.org/w/api.php?action=parse&page=17th%20Tamil%20Nadu%20Assembly&prop=wikitext&format=json&formatversion=2" -o /tmp/tn-17th-wiki.json
  (optional) /tmp/tn-assembly.geojson for the AC_NO → DIST_NAME cross-check

The output file is DATA, not code — when a new assembly is sworn in, re-run
this against the new page; nothing else changes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# First positional arg is the party/alliance name; later args may include
# color=…, shortname=…, rowspan=N in any order (the 17th-assembly table uses
# them all on alliance cells).
PARTY_TEMPLATE_RE = re.compile(r"\{\{Party name with color\|([^}|]+)((?:\|[^{}]*)?)\}\}", re.IGNORECASE)
ROWSPAN_ARG_RE = re.compile(r"rowspan\s*=\s*(\d+)")
# The district cell's link target always names the "<X> district" article —
# that anchor distinguishes it from rowspan'd constituency cells.
DISTRICT_CELL_RE = re.compile(r'rowspan="?(\d+)"?\s*\|\s*\[\[[^|\]]*[Dd]istrict[^|\]]*\|([^\]]+)\]\]')
LINK_RE = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]")
AC_LINK_RE = re.compile(r"\[\[([^|\]]+?)\s*(?:Assembly constituency|\(constituency\))(?:\|([^\]]+))?\]\]")


def _is_alliance(value: str) -> bool:
    # Alliance umbrella labels ("TVK-led Alliance", "Secular Progressive
    # Alliance", "AIADMK-led Alliance") share the party-template form; the
    # party column is distinguished by rowspan accounting plus this guard.
    return "alliance" in value.lower()


def main() -> int:
    wiki_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/tn-17th-wiki.json")
    assembly_label = sys.argv[2] if len(sys.argv) > 2 else "17th Tamil Nadu Legislative Assembly"
    elected_year = sys.argv[3] if len(sys.argv) > 3 else "2026"
    wiki = json.loads(wiki_path.read_text(encoding="utf-8"))
    wikitext = wiki["parse"]["wikitext"]
    page_title = wiki["parse"].get("title", "")

    start = wikitext.find('<section begin="MLA Header"/>')
    end = wikitext.find("|}", start)
    table = wikitext[start:end]

    rows = table.split("|-")[1:]  # drop the header chunk
    members: list[dict] = []
    district = ""
    district_remaining = 0
    party = ""
    party_remaining = 0
    alliance_remaining = 0

    for row in rows:
        row = row.strip()
        if not row:
            continue

        if district_remaining <= 0:
            dm = DISTRICT_CELL_RE.search(row)
            if dm:
                district_remaining = int(dm.group(1))
                district = dm.group(2).strip()

        # Plain `|58` or `|rowspan="2"|58` (members with a mid-term party
        # change get two stacked table rows for one seat).
        number_match = re.search(r"^\|\s*(?:rowspan=\"?\d+\"?\s*\|)?\s*(\d+)\s*$", row, re.MULTILINE)
        ac_match = AC_LINK_RE.search(row)

        if number_match and ac_match:
            ac_no = int(number_match.group(1))
            constituency = (ac_match.group(2) or ac_match.group(1)).strip()
            constituency = re.sub(r"\s+", " ", constituency)

            # MLA name: first wiki-link after the constituency link that is
            # not a party/alliance/district/remark artefact.
            after = row[ac_match.end():]
            name = ""
            for link in LINK_RE.finditer(after):
                text = link.group(1).strip()
                if "Assembly constituency" in link.group(0) or "district" in text.lower():
                    continue
                if text.lower().startswith(("chief minister", "speaker", "deputy", "leader of")):
                    continue
                name = text
                break
            if not name or name.lower() == "vacant":
                name = "Vacant"

            # Party via rowspan accounting: templates on a member row fill the
            # leftmost open column — party first, then alliance. Templates on
            # continuation rows (mid-term party switches) are ignored: the
            # roster records the party the seat was won under.
            for template in PARTY_TEMPLATE_RE.finditer(row):
                value = template.group(1).strip()
                extra_args = template.group(2) or ""
                rowspan_match = ROWSPAN_ARG_RE.search(extra_args)
                span = int(rowspan_match.group(1)) if rowspan_match else 1
                if party_remaining <= 0 and not _is_alliance(value):
                    party = value
                    party_remaining = span
                elif alliance_remaining <= 0:
                    alliance_remaining = span

            row_party = party if party_remaining > 0 else "Vacant"
            if name == "Vacant":
                row_party = "Vacant"

            members.append(
                {
                    "no": ac_no,
                    "constituency": constituency,
                    "district": district,
                    "mla": name,
                    "party": row_party,
                }
            )

        # rowspan counts TABLE rows, not members — a 2-row member's
        # continuation chunk must consume district/party/alliance slots too,
        # or every later district in the table drifts by one.
        party_remaining -= 1
        alliance_remaining -= 1
        district_remaining -= 1

    numbers = {m["no"] for m in members}
    missing = sorted(set(range(1, 235)) - numbers)
    if len(members) != 234 or missing:
        print(f"PARSE PROBLEM: {len(members)} rows, missing AC numbers: {missing[:10]}")
        return 1

    # Cross-check districts against the assembly shapes where available.
    try:
        geo = json.loads(Path("/tmp/tn-assembly.geojson").read_text(encoding="utf-8"))
        geo_district = {
            f["properties"]["AC_NO"]: f["properties"]["DIST_NAME"].title()
            for f in geo["features"]
        }
        mismatches = sum(
            1
            for m in members
            if geo_district.get(m["no"], "").lower()[:5] not in m["district"].lower()
            and m["district"].lower()[:5] not in geo_district.get(m["no"], "").lower()
        )
        print(f"geo cross-check: {234 - mismatches}/234 districts agree (boundary-change variance expected)")
    except FileNotFoundError:
        print("geo cross-check skipped (no /tmp/tn-assembly.geojson)")

    out = PROJECT_ROOT / "configs" / "mla_roster.json"
    out.write_text(
        json.dumps(
            {
                "assembly": assembly_label,
                "elected": elected_year,
                "source": f"en.wikipedia.org — {page_title or assembly_label} (CC BY-SA)",
                "note": "Replace this file when a new assembly is sworn in; the dashboard reads it as data.",
                "members": members,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    parties: dict[str, int] = {}
    for m in members:
        parties[m["party"]] = parties.get(m["party"], 0) + 1
    print(f"Wrote {out} — 234 members")
    for p, c in sorted(parties.items(), key=lambda kv: -kv[1]):
        print(f"  {p}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
