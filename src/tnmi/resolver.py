"""Entity resolution — free-text analysis surfaces → canonical entities.

``ai_analysis`` stores actors, districts, departments and schemes as free text
("EPS", "எடப்பாடி பழனிசாமி", "Transport / road safety"). The knowledge vault
needs one node per real-world object, so everything funnels through here:

  surface ──normalize──► alias index ──hit──► entities row
                                   └─miss──► status='candidate' entity
                                             (queued for a human, never dropped)

Resolution writes ``item_entities`` rows — the edges of the knowledge graph.
Exactly ONE analysis's view is kept per item: the same non-mock-then-latest
winner the dashboard shows, so vault numbers always match the operator's
screen. Re-running is idempotent; a better analysis replaces the old edges.

Districts reuse ``tnmi.districts.canonical_district`` (the proven bilingual
normalizer) rather than duplicating its alias table.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml
from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from tnmi.contracts import EntitySeed, EntityStatus, EntityType
from tnmi.districts import DISTRICT_TILES, canonical_district
from tnmi.storage import (
    AIAnalysisRecord,
    EntityAliasRecord,
    EntityRecord,
    ItemEntityRecord,
    RawItemRecord,
    add_entity_alias,
    save_item_entity,
    upsert_entity,
)

RESOLVER_VERSION = "resolver-v1"

# Surfaces that carry no entity signal — never resolved, never made candidates.
_GENERIC_SURFACES = {
    "",
    "unspecified",
    "none",
    "n/a",
    "na",
    "unknown",
    "general",
    "others",
    "other",
    "statewide",
    "state-wide",
    "tamil nadu",
    "tamilnadu",
    "தமிழ்நாடு",
    "தமிழகம்",
}

_SEED_SECTIONS: tuple[tuple[str, EntityType], ...] = (
    ("people", EntityType.PERSON),
    ("offices", EntityType.OFFICE),
    ("parties", EntityType.PARTY),
    ("organizations", EntityType.ORGANIZATION),
    ("departments", EntityType.DEPARTMENT),
    ("schemes", EntityType.SCHEME),
)

_TAMIL_RE = re.compile(r"[஀-௿]")
_WS_RE = re.compile(r"\s+")
_PAREN_RE = re.compile(r"\s*\([^)]*\)\s*$")
_SLUG_KEEP_RE = re.compile(r"[^a-z0-9஀-௿]+")


def normalize_surface(value: str | None) -> str:
    """Casefolded, dot/quote-stripped, whitespace-collapsed lookup key.

    Both alias registration and lookup pass through here, so "தி.மு.க" and
    "திமுக", or "M.K. Stalin" and "MK Stalin", land on the same key.
    """
    if not value:
        return ""
    cleaned = unicodedata.normalize("NFC", value).casefold()
    cleaned = cleaned.replace(".", "").replace("’", "").replace("'", "").replace("​", "")
    cleaned = _WS_RE.sub(" ", cleaned).strip(" \t -–—:;,&")
    return cleaned


def detect_lang(value: str) -> str:
    return "ta" if _TAMIL_RE.search(value) else "en"


def slugify(value: str) -> str:
    """Filesystem/wikilink-safe slug. Tamil text is kept (Obsidian handles
    unicode filenames); everything else lowercases to ascii-ish kebab."""
    normalized = unicodedata.normalize("NFC", value).casefold().strip()
    slug = _SLUG_KEEP_RE.sub("-", normalized).strip("-")
    return slug or "unnamed"


@dataclass
class ResolveStats:
    entities_seeded: int = 0
    items_processed: int = 0
    mentions_created: int = 0
    mentions_replaced: int = 0
    resolved_surfaces: int = 0
    candidate_surfaces: int = 0
    skipped_generic: int = 0
    candidates: Counter[str] = field(default_factory=Counter)

    @property
    def resolution_rate(self) -> float:
        total = self.resolved_surfaces + self.candidate_surfaces
        return (self.resolved_surfaces / total) if total else 1.0

    def as_dict(self) -> dict[str, object]:
        return {
            "entities_seeded": self.entities_seeded,
            "items_processed": self.items_processed,
            "mentions_created": self.mentions_created,
            "mentions_replaced": self.mentions_replaced,
            "resolved_surfaces": self.resolved_surfaces,
            "candidate_surfaces": self.candidate_surfaces,
            "skipped_generic": self.skipped_generic,
            "resolution_rate": round(self.resolution_rate, 4),
            "top_candidates": self.candidates.most_common(15),
        }


# ---------------------------------------------------------------------------
# Seeding


def load_entity_seeds(path: str | Path) -> list[EntitySeed]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    seeds: list[EntitySeed] = []
    for section, entity_type in _SEED_SECTIONS:
        for raw in data.get(section, []) or []:
            seeds.append(EntitySeed.model_validate({"entity_type": entity_type, **raw}))
    return seeds


def _register(session: Session, entity: EntityRecord, *aliases: str) -> None:
    seen: set[str] = set()
    for alias in aliases:
        alias = (alias or "").strip()
        normalized = normalize_surface(alias)
        if not alias or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        add_entity_alias(
            session,
            entity_id=entity.id,
            alias=alias,
            normalized=normalized,
            lang=detect_lang(alias),
        )


def sync_seed_entities(session: Session, seed_path: str | Path) -> int:
    """Upsert the curated roster. Idempotent; safe to re-run after every edit."""
    count = 0
    for seed in load_entity_seeds(seed_path):
        entity = upsert_entity(
            session,
            entity_type=seed.entity_type.value,
            slug=seed.slug,
            canonical_name=seed.canonical_name,
            name_ta=seed.name_ta,
            role=seed.role,
            party=seed.party,
            district=seed.district,
            portfolio=seed.portfolio,
            is_tvk=seed.is_tvk,
            status=EntityStatus.ACTIVE.value,
            metadata=seed.metadata,
        )
        _register(session, entity, seed.canonical_name, seed.name_ta, *seed.aliases)
        count += 1
    return count


def sync_district_entities(session: Session) -> int:
    """Seed the 38 districts from tnmi.districts (single source of truth).

    Resolution itself goes through canonical_district(), which already knows
    every spelling variant — DB aliases exist for dossier frontmatter display.
    """
    from tnmi.districts import _EXTRA_ALIASES  # registry module's alias table

    tamil_names: dict[str, str] = {}
    for alias, canonical in _EXTRA_ALIASES.items():
        if _TAMIL_RE.search(alias) and canonical not in tamil_names:
            tamil_names[canonical] = alias
    count = 0
    for name in sorted(DISTRICT_TILES):
        entity = upsert_entity(
            session,
            entity_type=EntityType.DISTRICT.value,
            slug=slugify(name),
            canonical_name=name,
            name_ta=tamil_names.get(name, ""),
            status=EntityStatus.ACTIVE.value,
        )
        _register(session, entity, name, tamil_names.get(name, ""))
        count += 1
    return count


def sync_source_entities(session: Session) -> int:
    """One source entity per observed outlet. Sources are facts (the item came
    from somewhere), so they auto-create as ACTIVE — profile fields (leaning,
    reliability) start empty in metadata_json for later curation."""
    names = session.scalars(select(RawItemRecord.source_name).distinct()).all()
    count = 0
    for name in sorted({n for n in names if n and n.strip()}):
        entity = upsert_entity(
            session,
            entity_type=EntityType.SOURCE.value,
            slug=f"source-{slugify(name)}",
            canonical_name=name,
            status=EntityStatus.ACTIVE.value,
        )
        _register(session, entity, name)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Best-analysis selection (mirrors the dashboard's dedupe rule exactly)


def _beats(challenger: AIAnalysisRecord, incumbent: AIAnalysisRecord) -> bool:
    challenger_mock = challenger.model_name == "mock"
    incumbent_mock = incumbent.model_name == "mock"
    if incumbent_mock and not challenger_mock:
        return True
    if incumbent_mock != challenger_mock:
        return False
    if challenger.created_at is None:
        return False
    if incumbent.created_at is None:
        return True
    if challenger.created_at != incumbent.created_at:
        return challenger.created_at > incumbent.created_at
    return challenger.id > incumbent.id


def pick_best_analyses(analyses: Iterable[AIAnalysisRecord]) -> dict[int, AIAnalysisRecord]:
    """raw_item_id → the analysis the dashboard would show (non-mock > mock,
    then most recent). Keeping this rule identical is load-bearing: vault
    numbers must match what the operator sees on screen."""
    best: dict[int, AIAnalysisRecord] = {}
    for row in analyses:
        incumbent = best.get(row.raw_item_id)
        if incumbent is None or _beats(row, incumbent):
            best[row.raw_item_id] = row
    return best


# ---------------------------------------------------------------------------
# Resolution


class AliasIndex:
    """normalized surface → EntityRecord, with a parenthetical-stripping
    fallback so unseen variants like "AIADMK (opposition)" still resolve."""

    def __init__(self, session: Session) -> None:
        self._session = session
        self._by_norm: dict[str, EntityRecord] = {}
        rows = session.execute(
            select(EntityAliasRecord, EntityRecord)
            .join(EntityRecord, EntityRecord.id == EntityAliasRecord.entity_id)
            .order_by(EntityRecord.id.asc(), EntityAliasRecord.id.asc())
        ).all()
        for alias, entity in rows:
            # First registration wins; later duplicates would be config errors.
            self._by_norm.setdefault(alias.normalized, entity)

    def lookup(self, surface: str) -> EntityRecord | None:
        normalized = normalize_surface(surface)
        if not normalized:
            return None
        hit = self._by_norm.get(normalized)
        if hit is not None:
            return hit
        stripped = _PAREN_RE.sub("", surface).strip()
        if stripped and stripped != surface:
            return self._by_norm.get(normalize_surface(stripped))
        return None

    def add(self, entity: EntityRecord, *aliases: str) -> None:
        _register(self._session, entity, *aliases)
        for alias in aliases:
            normalized = normalize_surface(alias)
            if normalized:
                self._by_norm.setdefault(normalized, entity)


def _candidate(
    session: Session,
    index: AliasIndex,
    surface: str,
    entity_type: EntityType,
    stats: ResolveStats,
) -> EntityRecord:
    """Unknown surface → candidate entity awaiting human confirmation."""
    slug = f"candidate-{slugify(surface)}"
    entity = upsert_entity(
        session,
        entity_type=entity_type.value,
        slug=slug,
        canonical_name=surface.strip(),
        status=EntityStatus.CANDIDATE.value,
    )
    index.add(entity, surface)
    stats.candidates[surface.strip()] += 1
    return entity


def _is_generic(surface: str) -> bool:
    return normalize_surface(surface) in _GENERIC_SURFACES


def _mention_targets(
    session: Session,
    item: RawItemRecord,
    analysis: AIAnalysisRecord,
    index: AliasIndex,
    stats: ResolveStats,
) -> list[tuple[EntityRecord, str, str]]:
    """(entity, mention_field, surface) triples for one item's best analysis."""
    targets: list[tuple[EntityRecord, str, str]] = []

    if item.source_name and item.source_name.strip():
        source = index.lookup(item.source_name)
        if source is None:
            source = upsert_entity(
                session,
                entity_type=EntityType.SOURCE.value,
                slug=f"source-{slugify(item.source_name)}",
                canonical_name=item.source_name.strip(),
                status=EntityStatus.ACTIVE.value,
            )
            index.add(source, item.source_name)
        targets.append((source, "source", item.source_name.strip()))

    for surface in analysis.political_actors or []:
        if not isinstance(surface, str) or _is_generic(surface):
            stats.skipped_generic += 1
            continue
        entity = index.lookup(surface)
        if entity is None:
            entity = _candidate(session, index, surface, EntityType.PERSON, stats)
            stats.candidate_surfaces += 1
        else:
            stats.resolved_surfaces += 1
        targets.append((entity, "political_actors", surface.strip()))

    district_name = canonical_district(analysis.district)
    if district_name and district_name in DISTRICT_TILES:
        district = index.lookup(district_name)
        if district is not None:
            stats.resolved_surfaces += 1
            targets.append((district, "district", analysis.district.strip()))

    department = (analysis.department or "").strip()
    if department and not _is_generic(department):
        entity = index.lookup(department)
        if entity is None:
            entity = _candidate(session, index, department, EntityType.DEPARTMENT, stats)
            stats.candidate_surfaces += 1
        else:
            stats.resolved_surfaces += 1
        targets.append((entity, "department", department))
    elif department:
        stats.skipped_generic += 1

    scheme = (analysis.scheme or "").strip()
    if scheme and not _is_generic(scheme):
        entity = index.lookup(scheme)
        if entity is None:
            entity = _candidate(session, index, scheme, EntityType.SCHEME, stats)
            stats.candidate_surfaces += 1
        else:
            stats.resolved_surfaces += 1
        targets.append((entity, "scheme", scheme))

    return targets


def resolve_items(session: Session, *, stats: ResolveStats | None = None) -> ResolveStats:
    """Resolve every item's best analysis into item_entities edges.

    Idempotent: re-running creates nothing new. When a better analysis has
    appeared for an item, the old analysis's edges are replaced wholesale so
    each item contributes exactly one view to the graph.
    """
    stats = stats or ResolveStats()
    index = AliasIndex(session)

    analyses = session.scalars(select(AIAnalysisRecord).order_by(AIAnalysisRecord.id.asc())).all()
    items_by_id = {
        row.id: row
        for row in session.scalars(select(RawItemRecord).order_by(RawItemRecord.id.asc()))
    }
    best = pick_best_analyses(analyses)

    for raw_item_id in sorted(best):
        analysis = best[raw_item_id]
        item = items_by_id.get(raw_item_id)
        if item is None:
            continue
        stale = session.execute(
            delete(ItemEntityRecord).where(
                ItemEntityRecord.raw_item_id == raw_item_id,
                ItemEntityRecord.analysis_id.is_not(None),
                ItemEntityRecord.analysis_id != analysis.id,
            )
        )
        stats.mentions_replaced += stale.rowcount or 0

        existing = {
            (row.entity_id, row.mention_field, row.surface)
            for row in session.scalars(
                select(ItemEntityRecord).where(ItemEntityRecord.raw_item_id == raw_item_id)
            )
        }
        for entity, mention_field, surface in _mention_targets(session, item, analysis, index, stats):
            key = (entity.id, mention_field, surface)
            if key in existing:
                continue
            save_item_entity(
                session,
                raw_item_id=raw_item_id,
                analysis_id=analysis.id,
                entity_id=entity.id,
                mention_field=mention_field,
                surface=surface,
                resolver_version=RESOLVER_VERSION,
            )
            existing.add(key)
            stats.mentions_created += 1
        stats.items_processed += 1

    session.flush()
    return stats


def resolve_all(session: Session, *, seed_path: str | Path) -> ResolveStats:
    """Seed roster + districts + sources, then resolve every item. The single
    entry point pipelines use; every step is idempotent."""
    stats = ResolveStats()
    stats.entities_seeded += sync_seed_entities(session, seed_path)
    stats.entities_seeded += sync_district_entities(session)
    stats.entities_seeded += sync_source_entities(session)
    return resolve_items(session, stats=stats)
