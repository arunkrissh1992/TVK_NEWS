"""JSON views over the entity knowledge graph — the live API behind the
war-room modal and actor scorecards.

The knowledge vault (``tnmi.vault``) already resolves every article's
free-text actors/districts/departments/sources into 88 canonical entities with
1,200+ mention links and renders them as markdown dossiers. This module reuses
the *same* aggregation (``_gather`` / ``_co_mentions``) but returns dicts, so
the web dashboard can show what was previously markdown-only: click any chip →
portrayal split, co-mention network, weekly trend, and cited evidence.

A short process-wide TTL cache keeps the all-items gather cheap across the
handful of entity requests a dashboard load fires.
"""

from __future__ import annotations

import time as _time
from collections import Counter, defaultdict
from datetime import date, timedelta
from typing import Any

from tnmi.storage import AIAnalysisRecord, RawItemRecord
from tnmi.vault import _VaultData, _Mention, _co_mentions, _gather, _portrayal

_PORTRAYALS = ("positive", "negative", "mixed", "neutral")
# Tie-break order among decisive categories — negative wins a tie (cautious).
_DECISIVE = ("negative", "positive", "mixed")

_CACHE: dict[str, Any] = {"ts": 0.0, "data": None}
_TTL_SECONDS = 120.0


def invalidate_entity_cache() -> None:
    """Drop the cached graph so the next request rebuilds it (call after ingest)."""
    _CACHE.update(ts=0.0, data=None)


def _vault_data(session) -> _VaultData:
    now = _time.time()
    cached = _CACHE["data"]
    if cached is not None and (now - _CACHE["ts"]) < _TTL_SECONDS:
        return cached
    data = _gather(session)
    _CACHE.update(ts=now, data=data)
    return data


def _split(counter: Counter[str]) -> dict[str, int]:
    return {key: counter.get(key, 0) for key in _PORTRAYALS}


def _dominant(counter: Counter[str]) -> str:
    """The net lean — the largest of positive/negative/mixed, neutral only when
    there is no decisive coverage at all. (Unlike the district map, which flags
    any negative; here a mostly-favourable figure should read favourable.)"""
    best = max(_DECISIVE, key=lambda key: counter.get(key, 0))
    return best if counter.get(best, 0) > 0 else "neutral"


def favorability(split: dict[str, int]) -> int | None:
    """Map a portrayal split to a 0–100 favorability score (50 = neutral).

    None when there is no positive/negative signal at all, so the UI can show
    "—" instead of a misleading 50.
    """
    pos, neg = split.get("positive", 0), split.get("negative", 0)
    decisive = pos + neg
    if decisive == 0:
        return None
    return round(50 + 50 * (pos - neg) / decisive)


def _entity_brief(entity, mentions: list[_Mention]) -> dict[str, Any]:
    split = _split(Counter(_portrayal(m.analysis) for m in mentions))
    return {
        "slug": entity.slug,
        "name": entity.canonical_name,
        "name_ta": entity.name_ta or "",
        "entity_type": entity.entity_type,
        "role": entity.role or "",
        "party": entity.party or "",
        "district": entity.district or "",
        "portfolio": entity.portfolio or "",
        "is_tvk": bool(entity.is_tvk),
        "mention_count": len(mentions),
        "portrayal_split": split,
        "dominant": _dominant(Counter(split)),
        "favorability": favorability(split),
    }


def list_entities(
    session,
    *,
    entity_type: str | None = None,
    limit: int = 200,
    min_mentions: int = 1,
) -> list[dict[str, Any]]:
    """Active entities ranked by mention volume, each with its portrayal split."""
    data = _vault_data(session)
    rows: list[dict[str, Any]] = []
    for entity_id, entity in data.entities.items():
        if entity.status != "active":
            continue
        if entity_type and entity.entity_type != entity_type:
            continue
        mentions = data.mentions_by_entity.get(entity_id, [])
        if len(mentions) < min_mentions:
            continue
        rows.append(_entity_brief(entity, mentions))
    rows.sort(key=lambda r: (-r["mention_count"], r["name"]))
    return rows[: max(0, limit)]


def _evidence_item(mention: _Mention) -> dict[str, Any]:
    item: RawItemRecord = mention.item
    analysis: AIAnalysisRecord = mention.analysis
    return {
        "raw_item_id": item.id,
        "analysis_id": analysis.id,
        "date": mention.day.isoformat(),
        "portrayal": _portrayal(analysis),
        "severity": (analysis.severity or "").lower(),
        "title": item.title or "",
        "source_name": item.source_name or "",
        "source_url": item.source_url or "",
        "district": analysis.district or "",
        "public_issue": analysis.public_issue or "",
        "summary": (analysis.summary_english or analysis.summary_original or "").strip(),
        "needs_review": bool(analysis.needs_human_review),
    }


def _weekly_timeseries(mentions: list[_Mention], *, as_of: date, weeks: int) -> list[dict[str, Any]]:
    """Portrayal counts per ISO week for the trailing ``weeks`` window, oldest
    first — the data behind a scorecard sparkline."""
    # Monday of the most recent week; walk back `weeks` buckets.
    end_monday = as_of - timedelta(days=as_of.weekday())
    starts = [end_monday - timedelta(weeks=offset) for offset in range(weeks - 1, -1, -1)]
    buckets: dict[date, Counter[str]] = {start: Counter() for start in starts}
    earliest = starts[0]
    for mention in mentions:
        if mention.day < earliest:
            continue
        bucket_start = mention.day - timedelta(days=mention.day.weekday())
        if bucket_start in buckets:
            buckets[bucket_start][_portrayal(mention.analysis)] += 1
    series: list[dict[str, Any]] = []
    for start in starts:
        split = _split(buckets[start])
        series.append(
            {
                "week_start": start.isoformat(),
                **split,
                "total": sum(split.values()),
                "favorability": favorability(split),
            }
        )
    return series


def entity_dossier(
    session,
    slug: str,
    *,
    evidence_limit: int = 12,
    weeks: int = 8,
) -> dict[str, Any] | None:
    """Full JSON dossier for one entity, or None if the slug is unknown."""
    data = _vault_data(session)
    entity = next((e for e in data.entities.values() if e.slug == slug), None)
    if entity is None:
        return None

    mentions = data.mentions_by_entity.get(entity.id, [])
    window_start = data.as_of - timedelta(days=30)
    recent = [m for m in mentions if m.day >= window_start]

    all_split = _split(Counter(_portrayal(m.analysis) for m in mentions))
    recent_split = _split(Counter(_portrayal(m.analysis) for m in recent))
    severe = sum(1 for m in mentions if (m.analysis.severity or "").lower() in {"high", "critical"})
    categories = Counter(
        (m.analysis.issue_category or "").strip().lower()
        for m in mentions
        if (m.analysis.issue_category or "").strip()
    )
    districts = Counter(
        (m.analysis.district or "").strip()
        for m in mentions
        if (m.analysis.district or "").strip() and m.analysis.district != "unspecified"
    )
    co = [
        {
            "slug": other.slug,
            "name": other.canonical_name,
            "entity_type": other.entity_type,
            "party": other.party or "",
            "count": count,
        }
        for other, count in _co_mentions(data, entity.id, limit=8)
    ]

    return {
        "slug": entity.slug,
        "name": entity.canonical_name,
        "name_ta": entity.name_ta or "",
        "entity_type": entity.entity_type,
        "role": entity.role or "",
        "party": entity.party or "",
        "district": entity.district or "",
        "portfolio": entity.portfolio or "",
        "is_tvk": bool(entity.is_tvk),
        "mention_count": len(mentions),
        "mention_count_30d": len(recent),
        "portrayal_split": all_split,
        "portrayal_split_30d": recent_split,
        "dominant": _dominant(Counter(all_split)),
        "favorability": favorability(all_split),
        "favorability_30d": favorability(recent_split),
        "severe_count": severe,
        "top_categories": [name for name, _ in categories.most_common(4)],
        "top_districts": [{"district": d, "count": c} for d, c in districts.most_common(5)],
        "co_mentions": co,
        "timeseries": _weekly_timeseries(mentions, as_of=data.as_of, weeks=weeks),
        "evidence": [_evidence_item(m) for m in mentions[: max(0, evidence_limit)]],
        "as_of": data.as_of.isoformat(),
    }


def actor_scorecards(
    session,
    *,
    limit: int = 12,
    weeks: int = 8,
    min_mentions: int = 2,
) -> list[dict[str, Any]]:
    """Ranked persona cards for people — the "who is winning the narrative" view.

    Each carries an all-time + 30d portrayal split, a favorability score, a
    weekly trend (for a sparkline), and the momentum delta (recent vs prior
    favorability) so leadership sees who is rising and who is slipping.
    """
    data = _vault_data(session)
    cards: list[dict[str, Any]] = []
    for entity_id, entity in data.entities.items():
        if entity.status != "active" or entity.entity_type != "person":
            continue
        mentions = data.mentions_by_entity.get(entity_id, [])
        if len(mentions) < min_mentions:
            continue
        window_start = data.as_of - timedelta(days=30)
        recent = [m for m in mentions if m.day >= window_start]
        all_split = _split(Counter(_portrayal(m.analysis) for m in mentions))
        recent_split = _split(Counter(_portrayal(m.analysis) for m in recent))
        series = _weekly_timeseries(mentions, as_of=data.as_of, weeks=weeks)
        # Momentum: favorability of the last half vs the first half of the window.
        half = max(1, len(series) // 2)
        first = Counter()
        last = Counter()
        for bucket in series[:half]:
            for key in _PORTRAYALS:
                first[key] += bucket[key]
        for bucket in series[half:]:
            for key in _PORTRAYALS:
                last[key] += bucket[key]
        fav_first = favorability(_split(first))
        fav_last = favorability(_split(last))
        momentum = (
            (fav_last - fav_first)
            if (fav_first is not None and fav_last is not None)
            else None
        )
        cards.append(
            {
                "slug": entity.slug,
                "name": entity.canonical_name,
                "name_ta": entity.name_ta or "",
                "role": entity.role or "",
                "party": entity.party or "",
                "is_tvk": bool(entity.is_tvk),
                "mention_count": len(mentions),
                "mention_count_30d": len(recent),
                "portrayal_split": all_split,
                "portrayal_split_30d": recent_split,
                "favorability": favorability(all_split),
                "favorability_30d": favorability(recent_split),
                "momentum": momentum,
                "dominant": _dominant(Counter(all_split)),
                "timeseries": series,
            }
        )
    # Rank by relevance now (recent mentions), then total volume.
    cards.sort(key=lambda c: (-c["mention_count_30d"], -c["mention_count"], c["name"]))
    return cards[: max(0, limit)]
