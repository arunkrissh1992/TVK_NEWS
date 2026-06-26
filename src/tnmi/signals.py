"""Spike detection — the forward-looking WARN layer.

A war-room needs to see a negative surge *forming*, not read about it the next
morning. For every entity (figure, party, district, department) this compares
recent daily coverage volume and negative share against a trailing baseline and
flags statistically unusual, negative-leaning surges as "emerging threats".

It reuses the same gathered entity graph as ``tnmi.entity_api`` (so a spike on a
district or a person is detected the same way) and emits alerts in the shape the
priority-alert rail already renders, plus a headline story to drill into.
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta
from typing import Any

from tnmi.entity_api import _vault_data
from tnmi.vault import _portrayal

# Entity types worth watching for a surge. Sources are excluded — a newspaper
# publishing more is not itself a threat.
_WATCH_TYPES = ("person", "party", "district", "department", "office", "scheme")


def _severity_rank(value: str) -> int:
    return {"critical": 3, "high": 2, "medium": 1}.get((value or "").lower(), 0)


def detect_spikes(
    session,
    *,
    recent_days: int = 3,
    baseline_days: int = 10,
    min_recent: int = 4,
    z_threshold: float = 1.0,
    threat_negative_share: float = 0.3,
    threats_only: bool = False,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return ranked emerging-signal alerts.

    An entity *surges* when its recent per-day mention rate is ``z_threshold``
    standard deviations above its baseline daily rate — coverage forming faster
    than usual. A surge whose recent coverage is at least
    ``threat_negative_share`` negative is tagged an **emerging threat** (the
    react-before-the-news-cycle case); the rest are **developing stories**.
    Threats always rank above developing stories. Set ``threats_only`` to drop
    non-hostile surges entirely.
    """
    data = _vault_data(session)
    as_of = data.as_of
    recent_start = as_of - timedelta(days=recent_days - 1)
    baseline_start = recent_start - timedelta(days=baseline_days)
    baseline_day_list = [baseline_start + timedelta(days=i) for i in range(baseline_days)]

    alerts: list[dict[str, Any]] = []
    for entity_id, entity in data.entities.items():
        if entity.status != "active" or entity.entity_type not in _WATCH_TYPES:
            continue
        mentions = data.mentions_by_entity.get(entity_id, [])
        recent = [m for m in mentions if m.day >= recent_start]
        if len(recent) < min_recent:
            continue
        baseline = [m for m in mentions if baseline_start <= m.day < recent_start]

        baseline_daily = Counter(m.day for m in baseline)
        counts = [baseline_daily.get(day, 0) for day in baseline_day_list]
        mean = sum(counts) / len(counts)
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        # Floor the std so a flat-zero baseline still yields a finite, sortable
        # z — a jump from nothing to a burst is the strongest spike, not a NaN.
        std = max(variance ** 0.5, 0.75)
        recent_rate = len(recent) / recent_days
        z = (recent_rate - mean) / std

        if z < z_threshold:
            continue
        neg_recent = sum(1 for m in recent if _portrayal(m.analysis) == "negative")
        neg_share = neg_recent / len(recent)
        is_threat = neg_share >= threat_negative_share
        if threats_only and not is_threat:
            continue

        severe = sum(1 for m in recent if (m.analysis.severity or "").lower() in {"high", "critical"})
        # Headline: most recent negative, highest-severity story to drill into
        # (falls back to the most severe recent story when none are negative).
        neg_mentions = [m for m in recent if _portrayal(m.analysis) == "negative"]
        headline = max(
            neg_mentions or recent,
            key=lambda m: (_severity_rank(m.analysis.severity), m.day, m.item.id),
        )
        # Threats outrank developing stories; within each, sharper + more
        # negative + more severe ranks first.
        score = (10 if is_threat else 0) + z * (0.5 + neg_share) * (1 + severe)
        alerts.append(
            {
                "kind": "emerging_threat" if is_threat else "developing_story",
                "is_threat": is_threat,
                "label": "Emerging threat" if is_threat else "Developing story",
                "slug": entity.slug,
                "name": entity.canonical_name,
                "entity_type": entity.entity_type,
                "recent_mentions": len(recent),
                "recent_negative": neg_recent,
                "negative_share": round(neg_share, 2),
                "baseline_daily_avg": round(mean, 2),
                "recent_daily_avg": round(recent_rate, 2),
                "z_score": round(z, 2),
                "severe_count": severe,
                "score": round(score, 3),
                "headline": {
                    "raw_item_id": headline.item.id,
                    "title": headline.item.title or "",
                    "source_name": headline.item.source_name or "",
                    "source_url": headline.item.source_url or "",
                    "district": headline.analysis.district or "",
                    "public_issue": headline.analysis.public_issue or "",
                    "summary": (headline.analysis.summary_english or headline.analysis.summary_original or "").strip(),
                    "severity": (headline.analysis.severity or "").lower(),
                },
            }
        )

    alerts.sort(key=lambda a: a["score"], reverse=True)
    return alerts[: max(0, limit)]
