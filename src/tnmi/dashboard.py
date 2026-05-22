from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import Select, case, func, select
from sqlalchemy.orm import Session

from tnmi.storage import AIAnalysisRecord, RawItemRecord, ReviewDecisionRecord, get_latest_review_decision


def _count_values(values: list[str | None]) -> dict[str, int]:
    return dict(Counter(value for value in values if value))


def get_dashboard_summary(session: Session) -> dict[str, Any]:
    analyses = session.scalars(select(AIAnalysisRecord).order_by(AIAnalysisRecord.id)).all()
    reviewed_analysis_ids = set(session.scalars(select(ReviewDecisionRecord.analysis_id)).all())
    return {
        "total_items": session.scalar(select(func.count()).select_from(RawItemRecord)) or 0,
        "total_analyses": len(analyses),
        "needs_human_review": sum(1 for row in analyses if row.needs_human_review),
        "reviewed": len(reviewed_analysis_ids),
        "pending_review": sum(1 for row in analyses if row.needs_human_review and row.id not in reviewed_analysis_ids),
        "stance_counts": _count_values([row.stance_toward_government for row in analyses]),
        "severity_counts": _count_values([row.severity for row in analyses]),
        "department_counts": _count_values([row.department for row in analyses]),
        "district_counts": _count_values([row.district for row in analyses]),
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
