from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from tnmi import __version__
from tnmi.config import Settings, load_newspaper_sources, load_x_handle_sources
from tnmi.contracts import ReviewDecisionCreate
from tnmi.dashboard import get_dashboard_summary, list_latest_items, list_review_queue
from tnmi.storage import (
    AIAnalysisRecord,
    RawItemRecord,
    ReviewDecisionRecord,
    create_session_factory,
    get_latest_review_decision,
    save_review_decision,
)

app = FastAPI(title="Tamil Nadu Media Intelligence API", version=__version__)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def require_operator(x_tnmi_operator_token: str | None = Header(default=None)) -> None:
    token = Settings().operator_api_token
    if token and x_tnmi_operator_token != token:
        raise HTTPException(status_code=401, detail="Operator token required")


def _review_decision_payload(record: ReviewDecisionRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "analysis_id": record.analysis_id,
        "reviewer_name": record.reviewer_name,
        "status": record.status,
        "note": record.note,
        "corrected_stance": record.corrected_stance,
        "corrected_relevance": record.corrected_relevance,
        "corrected_summary": record.corrected_summary,
        "created_at": record.created_at,
    }


def _list_report_files(report_dir: Path) -> list[dict[str, object]]:
    if not report_dir.exists():
        return []
    return [
        {
            "filename": path.name,
            "updated_at": path.stat().st_mtime,
        }
        for path in sorted(report_dir.glob("*.md"), key=lambda item: item.stat().st_mtime, reverse=True)
        if path.is_file()
    ]


def _settings_status(settings: Settings) -> dict[str, object]:
    openai_configured = bool(getattr(settings, "openai_api_key", None))
    database_url = getattr(settings, "database_url", "")
    return {
        "openai_configured": openai_configured,
        "openai_secret_status": "Configured and hidden" if openai_configured else "Not configured",
        "item_classifier_model": getattr(settings, "openai_model_item_classifier", "not configured"),
        "report_model": getattr(settings, "openai_model_report", "not configured"),
        "embedding_model": getattr(settings, "openai_embedding_model", "not configured"),
        "embedding_dimension": getattr(settings, "openai_embedding_dimension", "not configured"),
        "database_kind": database_url.split(":", 1)[0] if database_url else "not configured",
        "report_output_dir": str(getattr(settings, "report_output_dir", "")),
        "operator_guard": "Enabled" if getattr(settings, "operator_api_token", None) else "Local demo mode",
    }


def _dashboard_audit_events(
    *,
    summary: dict[str, object],
    settings_status: dict[str, object],
    report_files: list[dict[str, object]],
) -> list[dict[str, str]]:
    openai_ready = bool(settings_status["openai_configured"])
    openai_count = int(summary.get("openai_analyses", 0))
    latest_report = report_files[0]["filename"] if report_files else "No report generated"
    ai_status = "OpenAI live" if openai_ready and openai_count else "Configured, waiting for live run"
    if not openai_ready:
        ai_status = "Mock fallback only"
    return [
        {
            "stage": "Source Registry",
            "status": "Authorized public feeds",
            "detail": f"{summary.get('total_items', 0)} newspaper evidence records stored with source URLs.",
        },
        {
            "stage": "AI Classification",
            "status": ai_status,
            "detail": f"{openai_count} OpenAI analyses and {summary.get('mock_analyses', 0)} mock analyses retained for audit comparison.",
        },
        {
            "stage": "RAG Evidence Index",
            "status": "Ready",
            "detail": f"{summary.get('total_chunks', 0)} chunks and {summary.get('total_embeddings', 0)} embeddings available for retrieval.",
        },
        {
            "stage": "Human Review",
            "status": "Controlled escalation",
            "detail": f"{summary.get('pending_review', 0)} pending review items; sensitive claims remain review-gated.",
        },
        {
            "stage": "Report Package",
            "status": str(latest_report),
            "detail": "Daily report output is generated from stored evidence and AI analysis, not hand-entered demo text.",
        },
    ]


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


@app.get("/sources/x")
def x_sources() -> list[dict[str, object]]:
    settings = Settings()
    return [
        {**source.model_dump(mode="json"), "source_name": source.source_name}
        for source in load_x_handle_sources(settings.x_source_config)
    ]


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
    return [{"filename": str(item["filename"])} for item in _list_report_files(Path(settings.report_output_dir))]


@app.get("/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_operator)])
def dashboard_page(request: Request) -> HTMLResponse:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        summary = get_dashboard_summary(session)
        queue = list_review_queue(session, limit=50)
        latest_items = list_latest_items(session, limit=20)
    settings_status = _settings_status(settings)
    report_files = _list_report_files(Path(settings.report_output_dir))
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "summary": summary,
            "queue": queue,
            "latest_items": latest_items,
            "settings_status": settings_status,
            "audit_events": _dashboard_audit_events(
                summary=summary,
                settings_status=settings_status,
                report_files=report_files,
            ),
            "report_files": report_files,
        },
    )


@app.get("/settings", response_class=HTMLResponse, dependencies=[Depends(require_operator)])
def settings_page(request: Request) -> HTMLResponse:
    settings = Settings()
    return templates.TemplateResponse(
        request,
        "settings.html",
        {"settings_status": _settings_status(settings)},
    )


@app.get("/settings/status", dependencies=[Depends(require_operator)])
def settings_status() -> dict[str, object]:
    return _settings_status(Settings())


@app.get("/dashboard/summary", dependencies=[Depends(require_operator)])
def dashboard_summary() -> dict[str, object]:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        return get_dashboard_summary(session)


@app.get("/review/queue", dependencies=[Depends(require_operator)])
def review_queue(limit: int = 50) -> list[dict[str, object]]:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        return list_review_queue(session, limit=limit)


@app.post("/review/decisions", dependencies=[Depends(require_operator)])
def create_review_decision(decision: ReviewDecisionCreate) -> dict[str, object]:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        record = save_review_decision(session, decision)
        session.commit()
        return _review_decision_payload(record)


@app.get("/review/decisions/{analysis_id}", dependencies=[Depends(require_operator)])
def latest_review_decision(analysis_id: int) -> dict[str, object]:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        record = get_latest_review_decision(session, analysis_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Review decision not found")
        return _review_decision_payload(record)
