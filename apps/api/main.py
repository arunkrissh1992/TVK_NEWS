from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from sqlalchemy import select

from tnmi import __version__
from tnmi.config import Settings, load_newspaper_sources
from tnmi.storage import AIAnalysisRecord, RawItemRecord, create_session_factory

app = FastAPI(title="Tamil Nadu Media Intelligence API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": __version__}


@app.get("/sources")
def sources() -> list[dict[str, object]]:
    settings = Settings()
    return [source.model_dump(mode="json") for source in load_newspaper_sources(settings.news_source_config)]


@app.get("/items")
def items(limit: int = 50) -> list[dict[str, object]]:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    bounded_limit = max(1, min(limit, 200))
    with session_factory() as session:
        records = session.scalars(
            select(RawItemRecord).order_by(RawItemRecord.ingested_at.desc(), RawItemRecord.id.desc()).limit(bounded_limit)
        ).all()
    return [
        {
            "id": record.id,
            "source_name": record.source_name,
            "source_url": record.source_url,
            "title": record.title,
            "language": record.language,
            "published_at": record.published_at,
        }
        for record in records
    ]


@app.get("/analyses")
def analyses(limit: int = 50) -> list[dict[str, object]]:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    bounded_limit = max(1, min(limit, 200))
    with session_factory() as session:
        records = session.scalars(
            select(AIAnalysisRecord)
            .order_by(AIAnalysisRecord.created_at.desc(), AIAnalysisRecord.id.desc())
            .limit(bounded_limit)
        ).all()
    return [
        {
            "id": record.id,
            "raw_item_id": record.raw_item_id,
            "stance": record.stance_toward_government,
            "relevance": record.government_relevance,
            "summary": record.summary_english or record.summary_original,
            "confidence": record.confidence,
            "needs_human_review": record.needs_human_review,
        }
        for record in records
    ]


@app.get("/reports")
def reports() -> list[dict[str, str]]:
    settings = Settings()
    report_dir = Path(settings.report_output_dir)
    if not report_dir.exists():
        return []
    return [{"filename": path.name} for path in sorted(report_dir.glob("*.md")) if path.is_file()]
