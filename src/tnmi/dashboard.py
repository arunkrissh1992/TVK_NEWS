from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Select, case, func, select
from sqlalchemy.orm import Session

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


def get_dashboard_summary(session: Session) -> dict[str, Any]:
    analyses = session.scalars(select(AIAnalysisRecord).order_by(AIAnalysisRecord.id)).all()
    items = session.scalars(select(RawItemRecord).order_by(RawItemRecord.id)).all()
    embeddings = session.scalars(select(ChunkEmbeddingRecord).order_by(ChunkEmbeddingRecord.id)).all()
    reviewed_analysis_ids = set(session.scalars(select(ReviewDecisionRecord.analysis_id)).all())
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
        "total_chunks": session.scalar(select(func.count()).select_from(DocumentChunkRecord)) or 0,
        "total_embeddings": len(embeddings),
        "openai_analyses": sum(1 for row in analyses if row.model_name != "mock"),
        "mock_analyses": model_counts.get("mock", 0),
        "needs_human_review": sum(1 for row in analyses if row.needs_human_review),
        "reviewed": len(reviewed_analysis_ids),
        "pending_review": sum(1 for row in analyses if row.needs_human_review and row.id not in reviewed_analysis_ids),
        "stance_counts": _count_values([row.stance_toward_government for row in analyses]),
        "severity_counts": _count_values([row.severity for row in analyses]),
        "department_counts": _count_values([row.department for row in analyses]),
        "district_counts": _count_values([row.district for row in analyses]),
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
            }
        )
    return queue


def list_latest_items(session: Session, *, limit: int = 25) -> list[dict[str, Any]]:
    bounded_limit = max(1, min(limit, 100))
    rows = session.execute(
        select(RawItemRecord, AIAnalysisRecord)
        .join(AIAnalysisRecord, AIAnalysisRecord.raw_item_id == RawItemRecord.id)
        .where(RawItemRecord.source_type == "news")
        .order_by(
            RawItemRecord.ingested_at.desc(),
            RawItemRecord.id.desc(),
            case((AIAnalysisRecord.model_name == "mock", 0), else_=1).desc(),
            AIAnalysisRecord.created_at.desc(),
            AIAnalysisRecord.id.desc(),
        )
    ).all()

    latest_by_raw_item: dict[int, dict[str, Any]] = {}
    for item, analysis in rows:
        if item.id in latest_by_raw_item:
            continue
        latest_by_raw_item[item.id] = {
            "raw_item_id": item.id,
            "analysis_id": analysis.id,
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
            "needs_human_review": analysis.needs_human_review,
            "model_name": analysis.model_name,
            "prompt_version": analysis.prompt_version,
        }
        if len(latest_by_raw_item) >= bounded_limit:
            break
    return list(latest_by_raw_item.values())
