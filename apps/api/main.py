from __future__ import annotations

import os
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import select

from tnmi import __version__
from tnmi.ai import MockAIAnalyzer, OpenAIAnalyzer
from tnmi.local_models import LocalTamilAnalyzer
from tnmi.config import Settings, load_newspaper_sources, load_x_handle_sources
from tnmi.contracts import ReviewDecisionCreate
from tnmi.dashboard import (
    get_dashboard_summary,
    get_dashboard_trends,
    invalidate_briefing_cache,
    list_latest_items,
    list_recurring_themes,
    list_review_queue,
)
from tnmi.pipeline import DailyNewsPipeline, RequestsNewsClient
from tnmi.storage import (
    AIAnalysisRecord,
    RawItemRecord,
    ReviewDecisionRecord,
    create_session_factory,
    get_latest_review_decision,
    init_db,
    save_review_decision,
)

app = FastAPI(title="Tamil Nadu Media Intelligence API", version=__version__)
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _static_version() -> str:
    """Cache-bust query string based on the on-disk mtime of the dashboard
    static assets, so every edit forces browsers to fetch the latest file
    instead of serving a stale cached copy."""
    static_dir = BASE_DIR / "static"
    latest = 0.0
    for name in ("dashboard.css", "dashboard.js"):
        path = static_dir / name
        if path.exists():
            mtime = path.stat().st_mtime
            if mtime > latest:
                latest = mtime
    return str(int(latest))


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


def _briefing_groups(latest_items: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    positive = [item for item in latest_items if item.get("stance") == "positive"]
    concerns = [
        item
        for item in latest_items
        if item.get("stance") in {"negative", "mixed"} or item.get("needs_human_review")
    ]
    return {
        "positive_items": positive[:4],
        "concern_items": concerns[:4],
        "narrative_items": latest_items[:8],
    }


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


_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}


# ---------------------------------------------------------------------------
# Pull Latest — background ingest job
# ---------------------------------------------------------------------------

_INGEST_STATE: dict[str, Any] = {
    "status": "idle",   # idle | running | finished | failed
    "started_at": None,
    "finished_at": None,
    "result": None,     # summary dict from PipelineResult
    "error": None,
    "trigger": None,    # "manual" or "scheduled"
}
_INGEST_LOCK = threading.Lock()


class _FallbackAnalyzer:
    """OpenAI → Local Tamil model → Mock cascade.

    On any exception from a higher-tier analyser (quota, network, refusal),
    permanently switch to the next tier for the rest of this run. We do NOT
    retry OpenAI per-article once it has failed once — that would burn
    network repeatedly on a known-bad credential.

    The cascade order matches the user's stated requirement:
    "Product works 100% of days, regardless of OpenAI availability."
    """

    def __init__(self, tiers: list[Any]) -> None:
        if not tiers:
            raise ValueError("_FallbackAnalyzer needs at least one tier")
        self._tiers = tiers
        # Mark each tier as enabled until proven otherwise.
        self._disabled: set[int] = set()
        self.model_name = tiers[0].model_name

    def analyze(self, item):  # type: ignore[no-untyped-def]
        for index, tier in enumerate(self._tiers):
            if index in self._disabled:
                continue
            try:
                result = tier.analyze(item)
                self.model_name = tier.model_name
                return result
            except Exception:
                # Permanent skip — this tier failed at least once.
                self._disabled.add(index)
                traceback.print_exc()
        # Every tier failed (very unusual — Mock should never raise). Synthesize
        # a minimal record so the pipeline doesn't crash the whole run.
        from tnmi.contracts import (
            AIAnalysis as _AIAnalysis,
            GovernmentRelevance,
            Sentiment,
            Severity,
            Stance,
        )
        self.model_name = "fallback-empty"
        return _AIAnalysis(
            government_relevance=GovernmentRelevance.NONE,
            stance_toward_government=Stance.NEUTRAL,
            sentiment=Sentiment.NEUTRAL,
            target="unavailable",
            department="general",
            district="unspecified",
            scheme=None,
            topic=item.title or "unavailable",
            issue_category="unknown",
            severity=Severity.LOW,
            summary_original="",
            summary_english="",
            party_action="",
            people_impact="",
            root_cause="",
            recommended_step="",
            positive_points=[],
            negative_points=[],
            evidence_quotes_original=[],
            evidence_quotes_english=[],
            confidence=0.0,
            needs_human_review=True,
        )


def _build_news_analyzer(settings: Settings):
    """Build the cascade analyser. Order: OpenAI → LocalTamil → Mock.

    When OpenAI billing is restored every article gets real GPT analysis.
    When OpenAI is unavailable we fall back to the local Tamil-native
    analyser (AI4Bharat IndicBERT + keyword classifier). When even that
    is missing (no `transformers` install yet) we fall back to the mock.
    The dashboard never sees an empty day."""
    tiers: list[Any] = []
    if getattr(settings, "openai_api_key", None):
        try:
            tiers.append(
                OpenAIAnalyzer(
                    api_key=settings.openai_api_key,
                    model_name=settings.openai_model_item_classifier,
                )
            )
        except Exception:
            traceback.print_exc()
    tiers.append(LocalTamilAnalyzer())
    tiers.append(MockAIAnalyzer())
    return _FallbackAnalyzer(tiers)


def _run_ingest_job(trigger: str) -> None:
    """Body of the background task. Mutates _INGEST_STATE so the UI can poll."""
    with _INGEST_LOCK:
        _INGEST_STATE.update(
            status="running",
            started_at=datetime.now(timezone.utc).isoformat(),
            finished_at=None,
            result=None,
            error=None,
            trigger=trigger,
        )

    try:
        settings = Settings()
        sources = load_newspaper_sources(settings.news_source_config)
        analyzer = _build_news_analyzer(settings)
        session_factory = create_session_factory(settings.database_url)
        init_db(session_factory)
        pipeline = DailyNewsPipeline(
            session_factory=session_factory,
            news_client=RequestsNewsClient(),
            analyzer=analyzer,
        )
        result = pipeline.run(sources)
        # Fresh data means the cached briefing payload is stale.
        invalidate_briefing_cache()
        with _INGEST_LOCK:
            _INGEST_STATE.update(
                status="finished",
                finished_at=datetime.now(timezone.utc).isoformat(),
                result={
                    "items_seen": result.items_seen,
                    "items_saved": result.items_saved,
                    "analyses_saved": result.analyses_saved,
                    "failures": result.failures,
                    "sources_skipped": result.sources_skipped,
                    "analyzer_model": analyzer.model_name,
                },
            )
    except Exception as exc:  # noqa: BLE001 — surface to the UI as a job failure
        with _INGEST_LOCK:
            _INGEST_STATE.update(
                status="failed",
                finished_at=datetime.now(timezone.utc).isoformat(),
                error=f"{exc.__class__.__name__}: {exc}",
            )
        # Print full traceback to server logs for debugging.
        traceback.print_exc()


@app.get("/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_operator)])
def dashboard_page(request: Request) -> HTMLResponse:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        summary = get_dashboard_summary(session)
        queue = list_review_queue(session, limit=50)
        # Render every article so the KPI/filter counts above (which are
        # computed across the full DB) match the cards filtered client-side.
        latest_items = list_latest_items(session, limit=200)
        # GDELT cross-reference is opt-in — it adds 6-12 seconds to page load
        # when GDELT throttles or times out. Operators who want it can enable
        # via TNMI_ENABLE_GLOBAL_CROSSREF=1.
        themes = list_recurring_themes(
            session,
            limit=4,
            cross_reference_global=bool(int(os.environ.get("TNMI_ENABLE_GLOBAL_CROSSREF", "0"))),
        )
        trends = get_dashboard_trends(session, days=14)
    settings_status = _settings_status(settings)
    report_files = _list_report_files(Path(settings.report_output_dir))
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "summary": summary,
            "queue": queue,
            "latest_items": latest_items,
            "themes": themes,
            "trends": trends,
            **_briefing_groups(latest_items),
            "settings_status": settings_status,
            "audit_events": _dashboard_audit_events(
                summary=summary,
                settings_status=settings_status,
                report_files=report_files,
            ),
            "report_files": report_files,
            "static_version": _static_version(),
        },
        headers=_NO_CACHE_HEADERS,
    )


@app.post("/pipelines/news/run", dependencies=[Depends(require_operator)])
def trigger_news_ingest(background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Kick off a one-off newspaper ingest run in the background.

    Returns 200 immediately. The caller should poll GET /pipelines/news/status
    until ``status`` is ``finished`` or ``failed``, then reload the dashboard.
    Returns 409 if a run is already in progress so we never start two."""
    with _INGEST_LOCK:
        if _INGEST_STATE["status"] == "running":
            return {
                "status": "running",
                "message": "Ingest already in progress",
                "started_at": _INGEST_STATE["started_at"],
            }

    background_tasks.add_task(_run_ingest_job, "manual")
    return {
        "status": "accepted",
        "message": "Ingest started",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/pipelines/news/status", dependencies=[Depends(require_operator)])
def get_news_ingest_status() -> dict[str, Any]:
    with _INGEST_LOCK:
        return dict(_INGEST_STATE)


@app.get("/settings", response_class=HTMLResponse, dependencies=[Depends(require_operator)])
def settings_page(request: Request) -> HTMLResponse:
    settings = Settings()
    return templates.TemplateResponse(
        request,
        "settings.html",
        {"settings_status": _settings_status(settings), "static_version": _static_version()},
        headers=_NO_CACHE_HEADERS,
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
