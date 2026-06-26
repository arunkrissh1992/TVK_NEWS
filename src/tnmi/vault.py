"""Knowledge vault — the Obsidian-compatible markdown layer derived from the DB.

One dossier per canonical entity (person, party, office, district, department,
scheme, source), each accumulating dated, evidence-cited bullets wikilinked to
every other entity the item touches. ``Home.md`` is the war-room landing page.

Design rules (docs/superpowers/specs/2026-06-10-knowledge-vault-design.md):

* The DB stays the source of truth — the vault is a derived, regenerable
  artifact. Rendering twice with the same data produces byte-identical files.
* One note per canonical object, never per article. Articles are referenced by
  ``^raw-<id>`` block anchors that carry their ``raw_items.id``.
* Counts use the same best-analysis-per-item rule as the dashboard
  (``tnmi.resolver.pick_best_analyses``) and the same relevance gate, so vault
  numbers always match the operator's screen.
* Human sections (between ``<!-- human:begin -->`` markers) survive every
  regeneration byte-for-byte. Synthesis markers reserve the slot the phase-V4
  LLM pass will fill.
* No wall-clock timestamps in rendered content — "as of" derives from the data
  itself, which is what makes double-rendering deterministic.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from tnmi.resolver import pick_best_analyses
from tnmi.storage import (
    AIAnalysisRecord,
    EntityRecord,
    ItemEntityRecord,
    RawItemRecord,
)

VAULT_VERSION = "vault-v1"

EVIDENCE_MAX_BULLETS = 50
EVIDENCE_WINDOW_DAYS = 90

_HUMAN_BEGIN = "<!-- human:begin -->"
_HUMAN_END = "<!-- human:end -->"
_HUMAN_DEFAULT = "_Notes added here survive every regeneration._"
_HUMAN_RE = re.compile(
    re.escape(_HUMAN_BEGIN) + r"(.*?)" + re.escape(_HUMAN_END), re.DOTALL
)

_SYNTH_BEGIN = "<!-- synthesis:begin -->"
_SYNTH_END = "<!-- synthesis:end -->"

# entity_type → vault folder. Folder = graph color group = mental category.
_FOLDERS: dict[str, str] = {
    "person": "Entities/People",
    "party": "Entities/Parties",
    "office": "Entities/Offices",
    "org": "Entities/Organizations",
    "source": "Sources",
    "department": "Government/Departments",
    "scheme": "Government/Schemes",
    "district": "Geography/Districts",
    "constituency": "Geography/Constituencies",
}

_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


@dataclass
class _Mention:
    item: RawItemRecord
    analysis: AIAnalysisRecord
    day: date


@dataclass
class VaultStats:
    dossiers_written: int = 0
    dossiers_unchanged: int = 0
    entities_rendered: int = 0
    meta_written: int = 0
    candidates_listed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "entities_rendered": self.entities_rendered,
            "dossiers_written": self.dossiers_written,
            "dossiers_unchanged": self.dossiers_unchanged,
            "meta_written": self.meta_written,
            "candidates_listed": self.candidates_listed,
        }


# ---------------------------------------------------------------------------
# Small helpers


def _portrayal(analysis: AIAnalysisRecord) -> str:
    return (analysis.tvk_portrayal or analysis.stance_toward_government or "neutral").lower()


def _relevant(analysis: AIAnalysisRecord) -> bool:
    return (analysis.government_relevance or "").lower() != "none"


def _item_day(item: RawItemRecord, analysis: AIAnalysisRecord) -> date:
    stamp = item.published_at or item.ingested_at or analysis.created_at
    if stamp is None:
        return date(1970, 1, 1)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc).date()


def _split_line(counts: Counter[str]) -> str:
    return (
        f"+{counts.get('positive', 0)} / −{counts.get('negative', 0)} / "
        f"±{counts.get('mixed', 0)} / 0 {counts.get('neutral', 0)}"
    )


def _link(entity: EntityRecord) -> str:
    name = entity.canonical_name
    if entity.slug == name:
        return f"[[{entity.slug}]]"
    return f"[[{entity.slug}|{name}]]"


def _truncate(value: str, limit: int = 220) -> str:
    cleaned = " ".join((value or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _frontmatter(payload: dict[str, Any]) -> str:
    body = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False, default_flow_style=False)
    return f"---\n{body}---\n"


def _write_if_changed(path: Path, content: str) -> bool:
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _existing_human_block(path: Path) -> str:
    if not path.exists():
        return _HUMAN_DEFAULT
    match = _HUMAN_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        return _HUMAN_DEFAULT
    inner = match.group(1).strip("\n")
    return inner if inner.strip() else _HUMAN_DEFAULT


# ---------------------------------------------------------------------------
# Data gathering


@dataclass
class _VaultData:
    entities: dict[int, EntityRecord]
    mentions_by_entity: dict[int, list[_Mention]]
    edges_by_item: dict[int, list[ItemEntityRecord]]
    best: dict[int, AIAnalysisRecord]
    items: dict[int, RawItemRecord]
    as_of: date
    candidates: list[tuple[EntityRecord, int]] = field(default_factory=list)


def _gather(session: Session) -> _VaultData:
    entities = {
        row.id: row for row in session.scalars(select(EntityRecord).order_by(EntityRecord.slug))
    }
    items = {row.id: row for row in session.scalars(select(RawItemRecord))}
    analyses = session.scalars(select(AIAnalysisRecord).order_by(AIAnalysisRecord.id)).all()
    best = pick_best_analyses(analyses)
    edges = session.scalars(select(ItemEntityRecord).order_by(ItemEntityRecord.id)).all()

    edges_by_item: dict[int, list[ItemEntityRecord]] = defaultdict(list)
    mentions_by_entity: dict[int, list[_Mention]] = defaultdict(list)
    seen_pairs: set[tuple[int, int]] = set()
    candidate_counts: Counter[int] = Counter()

    for edge in edges:
        edges_by_item[edge.raw_item_id].append(edge)

    as_of = date(1970, 1, 1)
    for raw_item_id, analysis in best.items():
        item = items.get(raw_item_id)
        if item is None or not _relevant(analysis):
            continue
        day = _item_day(item, analysis)
        as_of = max(as_of, day)
        for edge in edges_by_item.get(raw_item_id, []):
            entity = entities.get(edge.entity_id)
            if entity is None:
                continue
            if entity.status == "candidate":
                candidate_counts[entity.id] += 1
                continue
            pair = (edge.entity_id, raw_item_id)
            if pair in seen_pairs:
                continue  # one mention per (entity, item) even with many surfaces
            seen_pairs.add(pair)
            mentions_by_entity[edge.entity_id].append(_Mention(item=item, analysis=analysis, day=day))

    for bucket in mentions_by_entity.values():
        bucket.sort(key=lambda m: (m.day, m.item.id), reverse=True)

    candidates = sorted(
        ((entities[eid], count) for eid, count in candidate_counts.items()),
        key=lambda pair: (-pair[1], pair[0].slug),
    )
    return _VaultData(
        entities=entities,
        mentions_by_entity=dict(mentions_by_entity),
        edges_by_item=dict(edges_by_item),
        best=best,
        items=items,
        as_of=as_of,
        candidates=candidates,
    )


def _co_mentions(data: _VaultData, entity_id: int, limit: int = 6) -> list[tuple[EntityRecord, int]]:
    counts: Counter[int] = Counter()
    for mention in data.mentions_by_entity.get(entity_id, []):
        for edge in data.edges_by_item.get(mention.item.id, []):
            other = data.entities.get(edge.entity_id)
            if other is None or other.id == entity_id:
                continue
            if other.status == "candidate" or other.entity_type == "source":
                continue
            counts[other.id] += 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], data.entities[pair[0]].slug))
    return [(data.entities[eid], count) for eid, count in ranked[:limit]]


def _item_links(data: _VaultData, item_id: int, exclude_entity_id: int, limit: int = 4) -> list[str]:
    links: list[str] = []
    seen: set[int] = set()
    edges = sorted(
        data.edges_by_item.get(item_id, []),
        key=lambda e: ("source district political_actors department scheme".find(e.mention_field), e.id),
    )
    for edge in edges:
        entity = data.entities.get(edge.entity_id)
        if entity is None or entity.id in {exclude_entity_id} | seen or entity.status == "candidate":
            continue
        seen.add(entity.id)
        links.append(_link(entity))
        if len(links) >= limit:
            break
    return links


# ---------------------------------------------------------------------------
# Dossier rendering


def _render_dossier(data: _VaultData, entity: EntityRecord, human_block: str) -> str:
    mentions = data.mentions_by_entity.get(entity.id, [])
    window_start = data.as_of - timedelta(days=30)
    recent = [m for m in mentions if m.day >= window_start]
    portrayals = Counter(_portrayal(m.analysis) for m in mentions)
    recent_portrayals = Counter(_portrayal(m.analysis) for m in recent)
    severe = sum(
        1 for m in mentions if (m.analysis.severity or "").lower() in {"high", "critical"}
    )
    categories = Counter(
        (m.analysis.issue_category or "").strip().lower()
        for m in mentions
        if (m.analysis.issue_category or "").strip()
    )
    top_categories = [name for name, _ in categories.most_common(3)]
    co_mentioned = _co_mentions(data, entity.id)

    aliases: list[str] = []
    for alias in [entity.canonical_name, entity.name_ta]:
        if alias and alias not in aliases:
            aliases.append(alias)

    front: dict[str, Any] = {
        "type": entity.entity_type,
        "slug": entity.slug,
        "name": entity.canonical_name,
    }
    if entity.name_ta:
        front["name_ta"] = entity.name_ta
    front["aliases"] = aliases
    if entity.role:
        front["role"] = entity.role
    if entity.party:
        front["party"] = entity.party
    if entity.portfolio:
        front["portfolio"] = entity.portfolio
    tags = [entity.entity_type]
    if entity.is_tvk:
        tags.append("tvk")
    front["tags"] = tags
    front["mentions_total"] = len(mentions)
    front["mentions_30d"] = len(recent)
    if mentions:
        front["last_seen"] = mentions[0].day.isoformat()

    lines: list[str] = [_frontmatter(front)]
    lines.append(f"# {entity.canonical_name}")
    if entity.name_ta and entity.name_ta != entity.canonical_name:
        lines.append("")
        lines.append(f"_{entity.name_ta}_")
    holder = (entity.metadata_json or {}).get("holder")
    if holder and holder in {e.slug for e in data.entities.values()}:
        holder_entity = next(e for e in data.entities.values() if e.slug == holder)
        lines.append("")
        lines.append(f"Current holder (per deployment roster): {_link(holder_entity)}")

    lines.append("")
    lines.append("## At a glance")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| mentions (total / last 30d) | {len(mentions)} / {len(recent)} |")
    lines.append(f"| portrayal split (all time) | {_split_line(portrayals)} |")
    lines.append(f"| portrayal split (last 30d) | {_split_line(recent_portrayals)} |")
    lines.append(f"| high/critical severity items | {severe} |")
    if top_categories:
        lines.append(f"| top issue categories | {', '.join(top_categories)} |")
    if co_mentioned:
        co_text = ", ".join(f"{_link(e)} ({n})" for e, n in co_mentioned)
        lines.append(f"| most co-mentioned | {co_text} |")

    lines.append("")
    lines.append("## Current state")
    lines.append("")
    lines.append(_SYNTH_BEGIN)
    if mentions:
        dominant = portrayals.most_common(1)[0][0]
        topic = top_categories[0] if top_categories else "general coverage"
        lines.append(
            f"Deterministic placeholder — phase V4 adds grounded synthesis. "
            f"Coverage to date: {len(mentions)} items, dominant portrayal "
            f"**{dominant}**, leading category **{topic}**."
        )
    else:
        lines.append(
            "No analyzed coverage yet. This dossier exists so the roster is "
            "complete; evidence accumulates automatically."
        )
    lines.append(_SYNTH_END)

    lines.append("")
    lines.append(f"## Evidence log (last {EVIDENCE_WINDOW_DAYS} days)")
    lines.append("")
    evidence_start = data.as_of - timedelta(days=EVIDENCE_WINDOW_DAYS)
    in_window = [m for m in mentions if m.day >= evidence_start]
    shown = in_window[:EVIDENCE_MAX_BULLETS]
    if not shown:
        lines.append("_No evidence in window._")
    for mention in shown:
        analysis = mention.analysis
        summary = _truncate(analysis.summary_english or analysis.summary_original)
        links = _item_links(data, mention.item.id, entity.id)
        link_text = f" ({', '.join(links)})" if links else ""
        severity = (analysis.severity or "low").lower()
        flag = " ⚠ unverified" if analysis.needs_human_review else ""
        lines.append(
            f"- {mention.day.isoformat()} · {_portrayal(analysis)} · {severity}{flag} — "
            f"{summary}{link_text} ^raw-{mention.item.id}"
        )
    hidden = len(mentions) - len(shown)
    if hidden > 0:
        lines.append("")
        lines.append(f"_{hidden} older item(s) remain in the database — see the dashboard._")

    lines.append("")
    lines.append("## Analyst notes")
    lines.append("")
    lines.append(_HUMAN_BEGIN)
    lines.append(human_block)
    lines.append(_HUMAN_END)
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Home / war room


def _render_home(data: _VaultData) -> str:
    gated = [
        (item_id, analysis)
        for item_id, analysis in sorted(data.best.items())
        if _relevant(analysis) and data.items.get(item_id) is not None
    ]
    portrayals = Counter(_portrayal(analysis) for _, analysis in gated)
    people_issues = sum(1 for _, analysis in gated if analysis.people_issue)

    def _day(item_id: int, analysis: AIAnalysisRecord) -> date:
        return _item_day(data.items[item_id], analysis)

    risks = sorted(
        (
            (item_id, analysis)
            for item_id, analysis in gated
            if _portrayal(analysis) == "negative" or analysis.people_issue
        ),
        key=lambda pair: (
            -_SEVERITY_RANK.get((pair[1].severity or "").lower(), 0),
            pair[1].needs_human_review,
            -(_day(*pair).toordinal()),
            -pair[0],
        ),
    )[:8]

    mention_counts = sorted(
        (
            (entity_id, len(mentions))
            for entity_id, mentions in data.mentions_by_entity.items()
            if data.entities[entity_id].entity_type not in {"source", "district"}
        ),
        key=lambda pair: (-pair[1], data.entities[pair[0]].slug),
    )[:10]
    district_counts = sorted(
        (
            (entity_id, mentions)
            for entity_id, mentions in data.mentions_by_entity.items()
            if data.entities[entity_id].entity_type == "district"
        ),
        key=lambda pair: (-len(pair[1]), data.entities[pair[0]].slug),
    )[:8]

    front = {
        "type": "home",
        "as_of": data.as_of.isoformat(),
        "items_tracked": len(gated),
        "vault_version": VAULT_VERSION,
    }
    lines = [_frontmatter(front)]
    lines.append("# TVK intelligence war room")
    lines.append("")
    lines.append(
        f"As of **{data.as_of.isoformat()}** · {len(gated)} analyzed items · "
        f"see [[Calendar]] · conventions in `_meta/CONVENTIONS.md`"
    )
    lines.append("")
    lines.append("## Pulse")
    lines.append("")
    lines.append("| signal | count |")
    lines.append("|---|---|")
    lines.append(f"| positive | {portrayals.get('positive', 0)} |")
    lines.append(f"| negative | {portrayals.get('negative', 0)} |")
    lines.append(f"| mixed | {portrayals.get('mixed', 0)} |")
    lines.append(f"| neutral | {portrayals.get('neutral', 0)} |")
    lines.append(f"| people issues | {people_issues} |")
    lines.append("")
    lines.append("## Top risks")
    lines.append("")
    if not risks:
        lines.append("_No open negative / people-issue items. Quiet day._")
    for item_id, analysis in risks:
        summary = _truncate(analysis.summary_english or analysis.summary_original, 160)
        links = _item_links(data, item_id, exclude_entity_id=-1, limit=3)
        link_text = f" ({', '.join(links)})" if links else ""
        flag = " ⚠" if analysis.needs_human_review else ""
        lines.append(
            f"- {_day(item_id, analysis).isoformat()} · **{(analysis.severity or 'low').lower()}**{flag} — "
            f"{summary}{link_text} (raw {item_id})"
        )
    lines.append("")
    lines.append("## Most mentioned")
    lines.append("")
    if not mention_counts:
        lines.append("_Nothing resolved yet — run `python -m pipelines.resolve_entities`._")
    for entity_id, count in mention_counts:
        entity = data.entities[entity_id]
        splits = Counter(
            _portrayal(m.analysis) for m in data.mentions_by_entity.get(entity_id, [])
        )
        lines.append(f"- {_link(entity)} — {count} mentions ({_split_line(splits)})")
    lines.append("")
    lines.append("## Districts in the news")
    lines.append("")
    if not district_counts:
        lines.append("_No district-tagged coverage in the current window._")
    for entity_id, mentions in district_counts:
        entity = data.entities[entity_id]
        negatives = sum(1 for m in mentions if _portrayal(m.analysis) == "negative")
        lines.append(f"- {_link(entity)} — {len(mentions)} items, {negatives} negative")
    lines.append("")
    lines.append("## Candidates awaiting confirmation")
    lines.append("")
    if not data.candidates:
        lines.append("_None — every resolved surface matched the roster._")
    else:
        lines.append(
            "_Unknown surfaces the resolver queued. Confirm each in "
            "`configs/entities.seed.yaml` (or merge as an alias), then re-run "
            "`python -m pipelines.resolve_entities`._"
        )
        lines.append("")
        for entity, count in data.candidates[:20]:
            lines.append(f"- `{entity.canonical_name}` — {count} mention(s), type guess `{entity.entity_type}`")
    lines.append("")
    return "\n".join(lines)


def _render_calendar(human_block: str) -> str:
    front = {"type": "calendar"}
    lines = [_frontmatter(front)]
    lines.append("# Political calendar")
    lines.append("")
    lines.append(
        "Recurring windows that shape coverage — plan communications ahead of "
        "them instead of reacting. Add dated entries (sessions, elections, "
        "court hearings, scheme anniversaries) in the notes section."
    )
    lines.append("")
    lines.append("| window | what recurs |")
    lines.append("|---|---|")
    lines.append("| January | Pongal week — farmer/civic sentiment peak; Republic Day |")
    lines.append("| Feb–Mar | State budget window — scheme/delivery scrutiny |")
    lines.append("| Apr–May | Peak summer — water and power stress stories |")
    lines.append("| May–Jun | School results, admissions, reopening — education coverage |")
    lines.append("| Jun–Sep | Southwest monsoon — agriculture, reservoir levels |")
    lines.append("| Aug | Independence Day; festival season ramps up |")
    lines.append("| Oct–Dec | Northeast monsoon — flood preparedness is THE district story |")
    lines.append("| Oct–Nov | Deepavali — prices, travel, public order |")
    lines.append("")
    lines.append("## Dated entries")
    lines.append("")
    lines.append(_HUMAN_BEGIN)
    lines.append(human_block)
    lines.append(_HUMAN_END)
    lines.append("")
    return "\n".join(lines)


_CONVENTIONS = """\
# Vault conventions

This vault is GENERATED from the TVK_NEWS database by
`python -m pipelines.build_vault`. The database is the source of truth.

## Rules

1. Never hand-edit generated sections — your change is overwritten on the next
   build. The two exceptions, preserved byte-for-byte across regenerations:
   - anything between `<!-- human:begin -->` and `<!-- human:end -->`
   - this `_meta/` folder is generated once and then yours
2. One note per canonical object — never per article. Articles are cited as
   `^raw-<id>` block anchors carrying their `raw_items.id`; look the row up in
   the dashboard or DB for the full text and verbatim Tamil quotes.
3. Links are always `[[slug|Display name]]`. Slugs are globally unique across
   folders, so links never break when a note moves.
4. `⚠ unverified` marks evidence from analyses flagged `needs_human_review` —
   verify before acting on it.
5. Unknown surfaces become `status=candidate` entities listed on [[Home]] —
   confirm them in `configs/entities.seed.yaml`, then re-run
   `python -m pipelines.resolve_entities`. Never delete a candidate silently.

## Folder = category = graph color

| folder | meaning |
|---|---|
| `Entities/People` | named politicians and public figures |
| `Entities/Parties` | political parties |
| `Entities/Offices` | role-word mentions (CM, Minister, MLA) — not persons |
| `Entities/Organizations` | institutions (Election Commission…) |
| `Government/Departments` | the analyzer's department taxonomy |
| `Government/Schemes` | government schemes |
| `Geography/Districts` | the 38 districts (seeded from tnmi/districts.py) |
| `Sources` | media outlets — profile fields arrive in a later phase |

## Frontmatter contract

`type`, `slug`, `name`, `name_ta`, `aliases` (Obsidian resolves all of them),
`tags` (entity type + `tvk` when applicable), `mentions_total`, `mentions_30d`,
`last_seen`. Machine-readable on purpose — Dataview queries work against it.

## For AI agents

- Start at [[Home]]; it links everything that currently matters.
- `grep -rl "<topic>" vault/` then read the dossier — cheaper than RAG for
  "what do we know about X over time" questions.
- Every claim you produce from a dossier must carry its `^raw-<id>` citation.
"""


def _graph_config() -> str:
    groups = [
        ("path:Entities/People", 0x7C3AED),
        ("path:Entities/Parties", 0xE24B4A),
        ("path:Entities/Offices", 0xD4537E),
        ("path:Entities/Organizations", 0x888780),
        ("path:Government/Departments", 0x378ADD),
        ("path:Government/Schemes", 0x1D9E75),
        ("path:Geography", 0x639922),
        ("path:Sources", 0xEF9F27),
    ]
    return json.dumps(
        {
            "collapse-color-groups": False,
            "colorGroups": [
                {"query": query, "color": {"a": 1, "rgb": rgb}} for query, rgb in groups
            ],
        },
        indent=2,
    )


# ---------------------------------------------------------------------------
# Entry point


def build_vault(session: Session, vault_dir: str | Path) -> VaultStats:
    """Render the whole vault. Idempotent: unchanged dossiers are not rewritten,
    and rendering twice with the same data is byte-identical."""
    vault_path = Path(vault_dir)
    stats = VaultStats()
    data = _gather(session)

    for entity in sorted(data.entities.values(), key=lambda e: e.slug):
        if entity.status == "candidate":
            continue
        folder = _FOLDERS.get(entity.entity_type)
        if folder is None:
            continue
        path = vault_path / folder / f"{entity.slug}.md"
        content = _render_dossier(data, entity, _existing_human_block(path))
        if _write_if_changed(path, content):
            stats.dossiers_written += 1
        else:
            stats.dossiers_unchanged += 1
        stats.entities_rendered += 1

    home_path = vault_path / "Home.md"
    if _write_if_changed(home_path, _render_home(data)):
        stats.meta_written += 1
    calendar_path = vault_path / "Calendar.md"
    if _write_if_changed(calendar_path, _render_calendar(_existing_human_block(calendar_path))):
        stats.meta_written += 1

    conventions_path = vault_path / "_meta" / "CONVENTIONS.md"
    if not conventions_path.exists():
        _write_if_changed(conventions_path, _CONVENTIONS)
        stats.meta_written += 1
    graph_path = vault_path / ".obsidian" / "graph.json"
    if not graph_path.exists():
        _write_if_changed(graph_path, _graph_config())
        stats.meta_written += 1

    stats.candidates_listed = len(data.candidates)
    return stats
