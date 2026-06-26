from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import Select, case, func, select
from sqlalchemy.orm import Session

from functools import lru_cache

from tnmi.clusters import (
    ArticleCluster,
    ArticleMember,
    RecurringThemesReport,
    ThemeCluster,
    cluster_all_articles_for_briefing,
    find_recurring_themes,
)
from tnmi.districts import canonical_department, canonical_district
from tnmi.mla import mlas_by_district
from tnmi.responsibility import resolve_responsibility
from tnmi.gdelt import (
    GdeltCrossReference,
    build_query_for_theme,
    search_articles as gdelt_search_articles,
)
from tnmi.storage import (
    AIAnalysisRecord,
    ChunkEmbeddingRecord,
    DocumentChunkRecord,
    RawItemRecord,
    ReviewDecisionRecord,
    get_latest_review_decision,
)


def _count_values(values: list[str | None]) -> dict[str, int]:
    return dict(Counter(value for value in values if value))


def _stance_label(stance: str | None) -> str:
    return {
        "positive": "Positive / நேர்மறை",
        "negative": "Negative / எதிர்மறை",
        "mixed": "Mixed / கலப்பு",
        "neutral": "Neutral / நடுநிலை",
        "people": "People Issue / மக்கள் பிரச்சனை",
    }.get(stance or "", "Review / மதிப்பாய்வு")


def _portrayal_kind(stance: str | None) -> str:
    if stance == "people":
        return "people"
    if stance == "positive":
        return "positive"
    if stance == "negative":
        return "negative"
    if stance == "mixed":
        return "mixed"
    return "neutral"


def _tvk_portrayal_value(value: str | None, fallback: str | None = None) -> str:
    value = (value or "").lower()
    if value in {"positive", "negative", "mixed", "neutral"}:
        return value
    fallback = (fallback or "").lower()
    if fallback in {"positive", "negative", "mixed", "neutral"}:
        return fallback
    return "neutral"


def _display_category(*, tvk_portrayal: str | None, people_issue: bool) -> str:
    portrayal = _tvk_portrayal_value(tvk_portrayal)
    if people_issue and portrayal == "neutral":
        return "people"
    return portrayal


def _display_list(values: list[str] | None, *, fallback: str = "") -> list[str]:
    cleaned = [value.strip() for value in values or [] if value and value.strip()]
    if cleaned:
        return cleaned[:3]
    return [fallback] if fallback else []


def _top_counts(counts: dict[str, int], *, limit: int = 8) -> list[dict[str, int | str]]:
    return [
        {"label": label, "count": count}
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _latest_datetime(values: list[datetime | None]) -> datetime | None:
    candidates = [value for value in values if value is not None]
    if not candidates:
        return None
    return max(candidates, key=_utc_sort_key)


def _utc_sort_key(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _is_openai_model(model_name: str | None) -> bool:
    name = (model_name or "").lower()
    return name.startswith(("gpt-", "gpt", "o1", "o3", "o4"))


def _is_semantic_model(model_name: str | None) -> bool:
    name = (model_name or "").lower()
    return _is_openai_model(name) or "gemma" in name or "sentence-transformers" in name or "indic-bert" in name


def _is_fallback_model(model_name: str | None) -> bool:
    return not _is_semantic_model(model_name)


def get_dashboard_summary(session: Session) -> dict[str, Any]:
    analyses = session.scalars(select(AIAnalysisRecord).order_by(AIAnalysisRecord.id)).all()
    items = session.scalars(select(RawItemRecord).order_by(RawItemRecord.id)).all()
    embeddings = session.scalars(select(ChunkEmbeddingRecord).order_by(ChunkEmbeddingRecord.id)).all()
    reviewed_analysis_ids = set(session.scalars(select(ReviewDecisionRecord.analysis_id)).all())
    # Dedupe to one analysis per article so KPI counts match the narrative
    # grid below. Prefer real (non-mock) models; among equal, prefer most
    # recently created.
    dedup_by_raw: dict[int, AIAnalysisRecord] = {}
    for row in analyses:
        existing = dedup_by_raw.get(row.raw_item_id)
        if existing is None:
            dedup_by_raw[row.raw_item_id] = row
            continue
        existing_is_mock = existing.model_name == "mock"
        row_is_mock = row.model_name == "mock"
        if existing_is_mock and not row_is_mock:
            dedup_by_raw[row.raw_item_id] = row
        elif existing_is_mock == row_is_mock:
            existing_at = existing.created_at
            row_at = row.created_at
            if row_at is not None and (existing_at is None or row_at > existing_at):
                dedup_by_raw[row.raw_item_id] = row
    # Same relevance gate as list_latest_items — KPI counts should match the
    # cards / table rows the operator actually sees on screen.
    unique_analyses = [
        row for row in dedup_by_raw.values()
        if (row.government_relevance or "").lower() != "none"
    ]
    latest_models = [row.model_name for row in unique_analyses]
    tvk_portrayals = [
        _tvk_portrayal_value(row.tvk_portrayal, row.stance_toward_government)
        for row in unique_analyses
    ]
    source_counts = _count_values([row.source_name for row in items])
    model_counts = _count_values([row.model_name for row in analyses])
    embedding_provider_counts = _count_values(
        [f"{row.provider_name}/{row.model_name}" for row in embeddings]
    )
    latest_ingested_at = _latest_datetime([row.ingested_at for row in items])
    latest_analysis_at = _latest_datetime([row.created_at for row in analyses])
    return {
        "total_items": len(items),
        "total_analyses": len(analyses),
        "source_count": len(source_counts),
        "total_chunks": session.scalar(select(func.count()).select_from(DocumentChunkRecord)) or 0,
        "total_embeddings": len(embeddings),
        "openai_analyses": sum(1 for model_name in latest_models if _is_openai_model(model_name)),
        "semantic_analyses": sum(1 for model_name in latest_models if _is_semantic_model(model_name)),
        "fallback_analyses": sum(1 for model_name in latest_models if _is_fallback_model(model_name)),
        "keyword_analyses": sum(1 for model_name in latest_models if model_name == "local-tamil-keywords"),
        "mock_analyses": model_counts.get("mock", 0),
        "needs_human_review": sum(1 for row in unique_analyses if row.needs_human_review),
        "reviewed": len(reviewed_analysis_ids),
        "pending_review": sum(
            1
            for row in unique_analyses
            if row.needs_human_review and row.id not in reviewed_analysis_ids
        ),
        "positive_count": sum(1 for value in tvk_portrayals if value == "positive"),
        "negative_count": sum(1 for value in tvk_portrayals if value == "negative"),
        "mixed_count": sum(1 for value in tvk_portrayals if value == "mixed"),
        "neutral_count": sum(1 for value in tvk_portrayals if value == "neutral"),
        "people_issue_count": sum(1 for row in unique_analyses if row.people_issue),
        "stance_counts": _count_values(tvk_portrayals),
        "severity_counts": _count_values([row.severity for row in unique_analyses]),
        "department_counts": _count_values([row.department for row in unique_analyses]),
        "district_counts": _count_values([row.district for row in unique_analyses]),
        "source_counts": source_counts,
        "top_sources": _top_counts(source_counts),
        "analysis_model_counts": model_counts,
        "embedding_provider_counts": embedding_provider_counts,
        "latest_ingested_at": latest_ingested_at,
        "latest_analysis_at": latest_analysis_at,
    }


def _queue_query() -> Select[tuple[AIAnalysisRecord, RawItemRecord]]:
    severity_rank = case(
        (AIAnalysisRecord.severity == "critical", 4),
        (AIAnalysisRecord.severity == "high", 3),
        (AIAnalysisRecord.severity == "medium", 2),
        (AIAnalysisRecord.severity == "low", 1),
        else_=0,
    )
    return (
        select(AIAnalysisRecord, RawItemRecord)
        .join(RawItemRecord, RawItemRecord.id == AIAnalysisRecord.raw_item_id)
        .where(AIAnalysisRecord.needs_human_review.is_(True))
        .order_by(
            severity_rank.desc(),
            AIAnalysisRecord.confidence.asc(),
            AIAnalysisRecord.created_at.desc(),
            AIAnalysisRecord.id.desc(),
        )
    )


def list_review_queue(session: Session, *, limit: int = 50) -> list[dict[str, Any]]:
    bounded_limit = max(1, min(limit, 200))
    rows = session.execute(_queue_query().limit(bounded_limit)).all()
    queue: list[dict[str, Any]] = []
    for analysis, item in rows:
        latest = get_latest_review_decision(session, analysis.id)
        if latest is not None:
            continue
        queue.append(
            {
                "analysis_id": analysis.id,
                "raw_item_id": item.id,
                "review_status": "pending",
                "source_name": item.source_name,
                "source_url": item.source_url,
                "title": item.title,
                "published_at": item.published_at,
                "language": item.language,
                "stance": analysis.stance_toward_government,
                "severity": analysis.severity,
                "department": analysis.department,
                "district": analysis.district,
                "summary": analysis.summary_english or analysis.summary_original,
                "confidence": analysis.confidence,
                "evidence": analysis.evidence_quotes_english or analysis.evidence_quotes_original,
                "recommended_step": (analysis.recommended_step or "").strip(),
                "risk_if_ignored": (analysis.risk_if_ignored or "").strip(),
                "talking_points": list(analysis.talking_points or []),
                "verification_checklist": list(analysis.verification_checklist or []),
                "draft_statement_original": (analysis.draft_statement_original or "").strip(),
                "draft_statement_english": (analysis.draft_statement_english or "").strip(),
            }
        )
    return queue


# Small TTL caches. Clustering all articles is ~10s on the demo DB; nothing
# in the result depends on the request, so a per-process cache shared across
# dashboard hits is correct as long as we invalidate after each ingest.
import time as _time

_BRIEFING_CACHE: dict[str, Any] = {"timestamp": 0.0, "payload": None, "limit": 0}
_THEMES_CACHE: dict[str, Any] = {}  # (limit, cross_ref) -> {timestamp, payload}
_CACHE_TTL_SECONDS = 300.0  # 5 minutes — invalidated whenever ingest writes new rows


def _briefing_cache_get(limit: int) -> list[dict[str, Any]] | None:
    now = _time.time()
    if (
        _BRIEFING_CACHE["payload"] is not None
        and _BRIEFING_CACHE["limit"] >= limit
        and (now - _BRIEFING_CACHE["timestamp"]) < _CACHE_TTL_SECONDS
    ):
        return _BRIEFING_CACHE["payload"][:limit]
    return None


def _briefing_cache_set(payload: list[dict[str, Any]], limit: int) -> None:
    _BRIEFING_CACHE.update(timestamp=_time.time(), payload=payload, limit=limit)


def _themes_cache_get(key: tuple[int, bool]) -> dict[str, Any] | None:
    entry = _THEMES_CACHE.get(key)
    if not entry:
        return None
    if (_time.time() - entry["timestamp"]) < _CACHE_TTL_SECONDS:
        return entry["payload"]
    return None


def _themes_cache_set(key: tuple[int, bool], payload: dict[str, Any]) -> None:
    _THEMES_CACHE[key] = {"timestamp": _time.time(), "payload": payload}


def list_latest_items(session: Session, *, limit: int = 25) -> list[dict[str, Any]]:
    """Return one card payload per *narrative* (not per article).

    Articles covering the same story across multiple newspapers collapse
    into a single card whose ``sources`` list names every contributing
    newspaper. Unique stories become singleton clusters that look identical
    to the previous "one card per article" layout.

    The card's summary, headline, briefing lines and evidence are taken
    from the cluster's "representative" article — the highest-relevance
    member, preferring those flagged for human review and longer titles.
    """
    # Upper bound is generous: every relevant story should be reachable via
    # pagination rather than silently truncated. The page stays short because
    # the grid is paginated client-side, not because we drop cards here.
    bounded_limit = max(1, min(limit, 1500))
    cached = _briefing_cache_get(bounded_limit)
    if cached is not None:
        return cached
    clusters = cluster_all_articles_for_briefing(session)
    overrides = _load_operator_stance_overrides(session)
    district_mlas = mlas_by_district()  # cached; used to name who acts per card

    payloads: list[dict[str, Any]] = []
    for cluster in clusters:
        rep = cluster.representative
        if (rep.relevance or "").lower() == "none":
            continue
        rep_evidence = _display_list(rep.evidence_quotes_original, fallback=rep.summary_original)

        # Source chips: one per distinct newspaper, in cluster-member order.
        sources: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for member in cluster.members:
            if member.source_name in seen_names:
                continue
            seen_names.add(member.source_name)
            sources.append(
                {
                    "name": member.source_name,
                    "stance": member.stance,
                    "stance_label": _stance_label(member.stance),
                    "url": member.source_url,
                    "raw_item_id": member.raw_item_id,
                    "analysis_id": member.analysis_id,
                }
            )

        tvk_portrayal = _tvk_portrayal_value(rep.tvk_portrayal, rep.stance)
        # Operator correction wins over AI classification.
        member_ids = [m.raw_item_id for m in cluster.members]
        operator_corrected = next(
            (overrides[rid] for rid in member_ids if rid in overrides),
            None,
        )
        if operator_corrected:
            tvk_portrayal = operator_corrected
        people_issue = any(m.people_issue for m in cluster.members)
        display_category = _display_category(
            tvk_portrayal=tvk_portrayal,
            people_issue=people_issue,
        )

        card = {
                "raw_item_id": rep.raw_item_id,
                "analysis_id": rep.analysis_id,
                "source_name": rep.source_name,
                "source_url": rep.source_url,
                "ingested_at": cluster.latest_ingested_at,
                "title": rep.title,
                "published_at": cluster.latest_published_at or rep.published_at,
                "language": "ta",  # rep language not tracked — Tamil-default
                "stance": tvk_portrayal,
                "government_stance": rep.stance,
                "tvk_portrayal": tvk_portrayal,
                "tvk_relevance": rep.tvk_relevance,
                "display_category": display_category,
                "stance_label": _stance_label(display_category),
                "portrayal_kind": _portrayal_kind(display_category),
                "stance_breakdown": cluster.stance_breakdown,
                "severity": rep.action_priority or None,
                "target": None,
                "people_issue": people_issue,
                "public_issue": rep.public_issue,
                "political_actors": rep.political_actors,
                "action_owner": rep.action_owner,
                "action_type": rep.action_type,
                "action_priority": rep.action_priority,
                "department": rep.department,
                "department_canonical": canonical_department(rep.department) or "",
                "district": rep.district,
                "district_canonical": canonical_district(rep.district) or "",
                "summary_original": rep.summary_original,
                "summary_english": rep.summary_english,
                "summary": rep.summary_english or rep.summary_original,
                "party_action": (rep.party_action or "").strip(),
                "people_impact": (rep.people_impact or "").strip(),
                "root_cause": (rep.root_cause or "").strip(),
                "recommended_step": (rep.recommended_step or "").strip(),
                "risk_if_ignored": (rep.risk_if_ignored or "").strip(),
                "talking_points": list(rep.talking_points or []),
                "verification_checklist": list(rep.verification_checklist or []),
                "draft_statement_original": (rep.draft_statement_original or "").strip(),
                "draft_statement_english": (rep.draft_statement_english or "").strip(),
                "positive_points": [],
                "negative_points": [],
                "evidence_original": rep_evidence,
                "evidence_english": [],
                "issue_category": None,
                "confidence": None,
                "needs_human_review": any(m.needs_human_review for m in cluster.members),
                "model_name": rep.model_name or None,
                # Heuristic = produced by the keyword fallback (not a semantic LLM).
                # The card labels its action playbook so operators never mistake
                # rule-based boilerplate for genuine per-article analysis.
                "is_heuristic": _is_fallback_model(rep.model_name),
                "prompt_version": rep.prompt_version or None,
                # Cluster metadata for the new multi-source UI.
                "cluster_size": cluster.size,
                "sources": sources,
                "source_count": cluster.distinct_source_count,
                "member_ids": [m.raw_item_id for m in cluster.members],
                # Only show the "N newspapers" badge when multiple DISTINCT
                # newspapers cover the same story — two near-duplicate articles
                # from one paper shouldn't claim cross-source coverage.
                "is_consolidated": cluster.distinct_source_count > 1,
                "is_operator_corrected": bool(operator_corrected),
        }
        # Who in government should act on this — department + district + CM escalation.
        card["responsibility"] = resolve_responsibility(card, district_mlas)
        payloads.append(card)
        if len(payloads) >= bounded_limit:
            break
    _briefing_cache_set(payloads, bounded_limit)
    return payloads


_PRIORITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


def select_priority_alerts(
    items: list[dict[str, Any]], *, limit: int = 5
) -> list[dict[str, Any]]:
    """Pick the items the leadership office must act on first.

    The dashboard is a monitor: among the narrative cards, the few that are
    high/critical-priority problems (negative for TVK, or a serious people
    issue) — or anything still awaiting human review — surface to the top as
    alerts. Positive and neutral coverage never raises an alert.
    """
    scored: list[tuple[tuple[int, int, float], dict[str, Any]]] = []
    for item in items:
        category = item.get("display_category")
        if category not in {"negative", "people"}:
            continue
        priority = (item.get("action_priority") or item.get("severity") or "").lower()
        priority_rank = _PRIORITY_RANK.get(priority, 0)
        needs_review = bool(item.get("needs_human_review"))
        # Only urgent problems become alerts: high/critical priority, or an
        # item still flagged for human review. Low-priority chatter stays in
        # the main feed.
        if priority_rank < 3 and not needs_review:
            continue
        when = item.get("published_at") or item.get("ingested_at")
        recency = _utc_sort_key(when).timestamp() if isinstance(when, datetime) else 0.0
        urgency = priority_rank + (1 if needs_review else 0)
        scored.append(((urgency, priority_rank, recency), item))

    scored.sort(key=lambda entry: entry[0], reverse=True)

    alerts: list[dict[str, Any]] = []
    for _key, item in scored[: max(0, limit)]:
        alerts.append(
            {
                "raw_item_id": item.get("raw_item_id"),
                "analysis_id": item.get("analysis_id"),
                "title": item.get("title"),
                "source_name": item.get("source_name"),
                "source_count": item.get("source_count", 1),
                "display_category": item.get("display_category"),
                "stance_label": item.get("stance_label"),
                "portrayal_kind": item.get("portrayal_kind"),
                "action_priority": item.get("action_priority") or item.get("severity"),
                "needs_human_review": bool(item.get("needs_human_review")),
                "public_issue": item.get("public_issue") or "",
                "summary": item.get("summary") or "",
                "risk_if_ignored": item.get("risk_if_ignored") or "",
                "recommended_step": item.get("recommended_step") or "",
                "published_at": item.get("published_at"),
            }
        )
    return alerts


def summarize_briefing_categories(latest_items: list[dict[str, Any]]) -> dict[str, int]:
    """Filter-deck counts derived directly from the briefing cards.

    The KPI deck is a *filter* over the cards, so each tile must equal the
    number of cards its filter reveals. Every card carries exactly one
    ``display_category`` in {positive, negative, mixed, neutral, people}, so the
    five buckets sum to the total number of cards. Deriving the headline numbers
    here — from the very list the operator sees — makes them impossible to drift
    from the grid (the old deck mixed an ungated raw-item total with per-item
    portrayal sub-counts, so "All Coverage" never matched its own breakdown).
    """
    counts = Counter((item.get("display_category") or "neutral") for item in latest_items)
    return {
        "briefing_total": len(latest_items),
        "positive_count": counts.get("positive", 0),
        "negative_count": counts.get("negative", 0),
        "mixed_count": counts.get("mixed", 0),
        "neutral_count": counts.get("neutral", 0),
        "people_issue_count": counts.get("people", 0),
    }


def compose_brief(
    *,
    summary: dict[str, Any],
    emerging_signals: list[dict[str, Any]],
    priority_alerts: list[dict[str, Any]],
    district_summary: dict[str, Any],
    actors: list[dict[str, Any]],
    subject: str = "TVK",
) -> list[dict[str, Any]]:
    """The "so what" — rank the few things a leader must know today out of
    signals already computed elsewhere. Pure + deterministic: no LLM, no new
    queries, just synthesis of the dashboard's own numbers into a decision."""
    lines: list[dict[str, Any]] = []

    pos = summary.get("positive_count", 0)
    neg = summary.get("negative_count", 0)
    if pos + neg:
        fav = round(50 + 50 * (pos - neg) / (pos + neg))
        tone = "good" if fav >= 55 else "bad" if fav <= 45 else "watch"
        lines.append(
            {
                "kind": "standing",
                "tone": tone,
                "title": f"{subject} standing: {fav}/100 favourability",
                "detail": f"{pos} positive vs {neg} negative across current coverage",
            }
        )

    threats = [s for s in emerging_signals if s.get("is_threat")]
    pick = (threats or emerging_signals)[:1]
    if pick:
        s = pick[0]
        lines.append(
            {
                "kind": "signal",
                "tone": "bad" if s.get("is_threat") else "watch",
                "title": f"{s['label']}: {s['name']}",
                "detail": f"{s['recent_mentions']} stories in 3 days, {s['z_score']}σ above its baseline",
                "slug": s.get("slug"),
            }
        )

    if priority_alerts:
        a = priority_alerts[0]
        lines.append(
            {
                "kind": "alert",
                "tone": "bad",
                "title": "Act now: " + (a.get("title") or a.get("summary") or "")[:90],
                "detail": a.get("risk_if_ignored") or a.get("recommended_step") or "",
            }
        )

    tiles = [t for t in district_summary.get("tiles", []) if t.get("total")]
    if tiles:
        hot = max(tiles, key=lambda t: t.get("negative", 0) + t.get("people", 0))
        concerns = hot.get("negative", 0) + hot.get("people", 0)
        if concerns:
            top_issue = hot["top_issues"][0]["issue"] if hot.get("top_issues") else ""
            lines.append(
                {
                    "kind": "hotspot",
                    "tone": "watch",
                    "title": f"Hotspot: {hot['district']}",
                    "detail": f"{concerns} negative/people-issue stories"
                    + (f" · top: {top_issue}" if top_issue else ""),
                }
            )

    rivals = [a for a in actors if not a.get("is_tvk")]
    if rivals:
        r = rivals[0]
        mom = r.get("momentum")
        arrow = "↑" if (mom or 0) > 0 else "↓" if (mom or 0) < 0 else "→"
        lines.append(
            {
                "kind": "rival",
                "tone": "watch",
                "title": f"Rival watch: {r['name']} ({r.get('party', '')})",
                "detail": f"favourability {r.get('favorability')}/100 {arrow}"
                + (str(abs(mom)) if mom else ""),
                "slug": r.get("slug"),
            }
        )

    return lines


def build_daily_brief(session: Session, *, subject: str = "TVK") -> list[dict[str, Any]]:
    """Assemble the inputs and compose the brief — one place the dashboard route
    and the email/WhatsApp sender both call, so they never drift."""
    from tnmi.districts import summarize_by_district
    from tnmi.entity_api import actor_scorecards
    from tnmi.signals import detect_spikes

    latest = list_latest_items(session, limit=200)
    return compose_brief(
        summary=get_dashboard_summary(session),
        emerging_signals=detect_spikes(session, limit=6),
        priority_alerts=select_priority_alerts(latest, limit=5),
        district_summary=summarize_by_district(latest),
        actors=actor_scorecards(session, limit=8),
        subject=subject,
    )


def invalidate_briefing_cache() -> None:
    """Called after an ingest writes new rows so the next dashboard hit
    rebuilds the briefing instead of serving stale data. Clears both the
    narrative-card cache and the recurring-themes cache."""
    _BRIEFING_CACHE.update(timestamp=0.0, payload=None, limit=0)
    _THEMES_CACHE.clear()


def _load_operator_stance_overrides(session: Session) -> dict[int, str]:
    """Return raw_item_id -> corrected_stance, latest override wins.

    Operators can mark a card as the wrong stance via the dashboard; that
    decision lands in review_decisions with corrected_stance set. The
    briefing trusts that override above the AI assignment forever after,
    so once you fix a misclassification it stays fixed across rebuilds."""
    rows = session.execute(
        select(
            ReviewDecisionRecord.analysis_id,
            ReviewDecisionRecord.corrected_stance,
            ReviewDecisionRecord.created_at,
            AIAnalysisRecord.raw_item_id,
        )
        .join(AIAnalysisRecord, AIAnalysisRecord.id == ReviewDecisionRecord.analysis_id)
        .where(ReviewDecisionRecord.corrected_stance.is_not(None))
        .order_by(ReviewDecisionRecord.created_at.desc())
    ).all()
    overrides: dict[int, str] = {}
    for _analysis_id, corrected, _created_at, raw_item_id in rows:
        if raw_item_id in overrides:
            continue  # we ordered by created_at DESC, first hit is latest
        overrides[raw_item_id] = corrected
    return overrides


def list_recurring_themes(
    session: Session,
    *,
    limit: int = 4,
    min_cluster_size: int = 2,
    similarity_threshold: float = 0.7,
    cross_reference_global: bool = True,
) -> dict[str, Any]:
    """Dashboard-ready payload for the Recurring Themes panel.

    When ``cross_reference_global`` is True we hit GDELT for each visible
    theme to surface a global-signal badge. Calls are cached per process so
    a dashboard reload doesn't re-fetch.
    """
    cache_key = (limit, bool(cross_reference_global))
    cached = _themes_cache_get(cache_key)
    if cached is not None:
        return cached

    report: RecurringThemesReport = find_recurring_themes(
        session,
        similarity_threshold=similarity_threshold,
        min_cluster_size=min_cluster_size,
        limit=limit,
    )

    serialised: list[dict[str, Any]] = []
    for cluster in report.themes:
        payload = _serialise_theme(cluster)
        if cross_reference_global and payload.get("sample_title"):
            payload["global_signal"] = _gdelt_signal_for_title(payload["sample_title"])
        else:
            payload["global_signal"] = None
        serialised.append(payload)

    payload = {
        "has_themes": report.has_themes,
        "diagnostic": report.diagnostic,
        "total_articles_indexed": report.total_articles_indexed,
        "themes": serialised,
    }
    _themes_cache_set(cache_key, payload)
    return payload


@lru_cache(maxsize=256)
def _gdelt_signal_for_title(theme_title: str) -> dict[str, Any]:
    """One GDELT lookup per unique theme title, cached for the process
    lifetime. The dashboard renders several panels per request — we don't
    want each one to repeat the same network call."""
    try:
        query = build_query_for_theme(theme_title)
        cross: GdeltCrossReference = gdelt_search_articles(query, timespan="24h", max_records=8)
        return {
            "has_signal": cross.has_signal,
            "article_count": cross.article_count,
            "distinct_domains": cross.distinct_domains,
            "top_match_title": cross.matches[0].title if cross.matches else None,
            "top_match_url": cross.matches[0].url if cross.matches else None,
            "query": query,
        }
    except Exception:  # noqa: BLE001 — never let GDELT break the dashboard
        return {
            "has_signal": False,
            "article_count": 0,
            "distinct_domains": 0,
            "top_match_title": None,
            "top_match_url": None,
            "query": theme_title,
        }


def _serialise_theme(cluster: ThemeCluster) -> dict[str, Any]:
    stance = cluster.dominant_stance
    return {
        "representative_id": cluster.representative_id,
        "sample_title": cluster.sample_title or "Untitled item",
        "sample_summary": cluster.sample_summary,
        "size": cluster.size,
        "source_count": cluster.source_count,
        "dominant_stance": stance,
        "portrayal_kind": _portrayal_kind(stance),
        "stance_label": _stance_label(stance),
        "stance_breakdown": cluster.stance_breakdown,
        "sources": sorted({m.source_name for m in cluster.members}),
        "latest_published_at": cluster.latest_published_at,
        "member_ids": [m.raw_item_id for m in cluster.members],
        # Phase E
        "coordination_score": cluster.coordination_score,
        "is_coordinated": cluster.is_coordinated,
        "momentum": cluster.momentum(),
    }


# ---------------------------------------------------------------------------
# Trends + breakdowns (Batch 2)
# ---------------------------------------------------------------------------

_STANCE_KEYS = ("positive", "negative", "mixed", "neutral")


def get_dashboard_trends(session: Session, *, days: int = 14) -> dict[str, Any]:
    """One trip to the DB; returns:
      - stance_timeseries: per-day stance counts for the last N days
      - department_breakdown: top departments by article count, split by stance
      - district_breakdown: same shape for districts
    All counts dedupe to one analysis per raw_item (real OpenAI > mock; latest
    wins) so the totals line up with the KPI/filter deck above.
    """
    rows = session.execute(
        select(
            AIAnalysisRecord.raw_item_id,
            AIAnalysisRecord.stance_toward_government,
            AIAnalysisRecord.department,
            AIAnalysisRecord.district,
            AIAnalysisRecord.government_relevance,
            AIAnalysisRecord.model_name,
            AIAnalysisRecord.created_at,
            RawItemRecord.published_at,
            RawItemRecord.ingested_at,
        )
        .join(RawItemRecord, RawItemRecord.id == AIAnalysisRecord.raw_item_id)
        .where(RawItemRecord.source_type == "news")
    ).all()

    dedup: dict[int, dict[str, Any]] = {}
    for raw_id, stance, department, district, relevance, model, created_at, published_at, ingested_at in rows:
        candidate = {
            "stance": stance,
            "department": (department or "unspecified").strip() or "unspecified",
            "district": (district or "unspecified").strip() or "unspecified",
            "relevance": (relevance or "").strip().lower(),
            "model": model,
            "created_at": created_at,
            "published_at": published_at,
            "ingested_at": ingested_at,
        }
        existing = dedup.get(raw_id)
        if existing is None:
            dedup[raw_id] = candidate
            continue
        existing_mock = existing["model"] == "mock"
        cand_mock = candidate["model"] == "mock"
        if existing_mock and not cand_mock:
            dedup[raw_id] = candidate
        elif existing_mock == cand_mock:
            if candidate["created_at"] and (
                existing["created_at"] is None or candidate["created_at"] > existing["created_at"]
            ):
                dedup[raw_id] = candidate

    # Only the briefing-relevant stories count, same gate as the cards/KPIs, so
    # the district/department breakdowns don't drown in off-topic "Unclassified".
    items = [item for item in dedup.values() if item["relevance"] != "none"]

    return {
        "stance_timeseries": _stance_timeseries(items, days=days),
        "department_breakdown": _categorical_breakdown(items, attribute="department", limit=6),
        "district_breakdown": _categorical_breakdown(items, attribute="district", limit=6),
        "total_items": len(items),
    }


def _stance_timeseries(items: list[dict[str, Any]], *, days: int) -> list[dict[str, Any]]:
    days = max(1, days)
    today_utc = datetime.now(timezone.utc).date()
    start = today_utc - timedelta(days=days - 1)

    buckets: dict[date, dict[str, int]] = {
        start + timedelta(days=i): {k: 0 for k in _STANCE_KEYS}
        for i in range(days)
    }

    for item in items:
        when = item["published_at"] or item["ingested_at"]
        if when is None:
            continue
        when_utc = when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)
        day = when_utc.date()
        if day < start or day > today_utc:
            continue
        stance = item["stance"] if item["stance"] in _STANCE_KEYS else "neutral"
        buckets[day][stance] += 1

    series: list[dict[str, Any]] = []
    for day, counts in buckets.items():
        total = sum(counts.values())
        series.append(
            {
                "date": day.isoformat(),
                "label_short": day.strftime("%d %b"),
                "label_long": day.strftime("%A, %d %B"),
                "total": total,
                **counts,
            }
        )
    return series


def _categorical_breakdown(
    items: list[dict[str, Any]], *, attribute: str, limit: int
) -> list[dict[str, Any]]:
    bucket: dict[str, dict[str, int]] = defaultdict(lambda: {k: 0 for k in _STANCE_KEYS})
    for item in items:
        label = (item.get(attribute) or "Unclassified").strip() or "Unclassified"
        # Normalise demo/placeholder labels so the breakdown stays informative
        # in mock mode but doesn't shout "unspecified" at the operator.
        if label.lower() in {"unspecified", "unknown", "none", "n/a"}:
            label = "Unclassified"
        elif label.lower() == "general":
            label = "General"
        stance = item["stance"] if item["stance"] in _STANCE_KEYS else "neutral"
        bucket[label][stance] += 1

    rows: list[dict[str, Any]] = []
    for label, counts in bucket.items():
        total = sum(counts.values())
        rows.append({"label": label, "total": total, **counts})

    rows.sort(key=lambda row: (-row["total"], row["label"]))
    return rows[: max(0, limit)]
