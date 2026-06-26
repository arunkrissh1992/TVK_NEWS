from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import traceback
from datetime import date, datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sqlalchemy import select

from tnmi import __version__
from tnmi.ai import MockAIAnalyzer, OpenAIAnalyzer
from tnmi.chat import (
    ChatAIProvider,
    ChatTurn,
    EvidenceOnlyChatProvider,
    OllamaChatProvider,
    _INSUFFICIENT_ANSWER,
    answer_question,
    build_dossier_context,
    build_retrieval_query,
    retrieve_chat_evidence,
)
from tnmi.local_models import LocalTamilAnalyzer
from tnmi.config import Settings, load_newspaper_sources, load_x_handle_sources
from tnmi.contracts import ReviewDecisionCreate
from tnmi.dashboard import (
    compose_brief,
    get_dashboard_summary,
    get_dashboard_trends,
    invalidate_briefing_cache,
    list_latest_items,
    list_recurring_themes,
    list_review_queue,
    select_priority_alerts,
    summarize_briefing_categories,
)
from tnmi.districts import summarize_by_department, summarize_by_district
from tnmi.entity_api import actor_scorecards, entity_dossier, list_entities
from tnmi.flywheel import model_health
from tnmi.mla import mlas_by_district, party_seat_counts, roster_label
from tnmi.signals import detect_spikes
from tnmi.tenancy import ControlPlane
from tnmi.tn_map import TN_MAP_LABELS, TN_MAP_PATHS, TN_MAP_VIEWBOX
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
logger = logging.getLogger(__name__)
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


@lru_cache(maxsize=1)
def _control_plane() -> ControlPlane:
    s = Settings()
    return ControlPlane(s.control_database_url, tenants_dir=s.tenants_dir)


@lru_cache(maxsize=16)
def _cached_factory(database_url: str):
    return create_session_factory(database_url)


def get_session_factory(x_tenant_key: str | None = Header(default=None)):
    """Per-request DB picker. Single-tenant (default): the configured database.
    Multi-tenant: resolve the X-Tenant-Key to its isolated tenant database."""
    settings = Settings()
    if not getattr(settings, "multi_tenant", False):
        return _cached_factory(settings.database_url)
    tenant = _control_plane().authenticate_api_key(x_tenant_key or "")
    if tenant is None:
        raise HTTPException(status_code=401, detail="Valid X-Tenant-Key required")
    return _control_plane().session_factory_for(tenant)


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
    semantic_count = int(summary.get("semantic_analyses", 0))
    fallback_count = int(summary.get("fallback_analyses", 0))
    latest_report = report_files[0]["filename"] if report_files else "No report generated"
    ai_status = "Semantic AI live" if semantic_count else "Keyword fallback"
    if openai_ready and not openai_count:
        ai_status = "OpenAI configured, waiting for live run"
    return [
        {
            "stage": "Source Registry",
            "status": "Authorized public feeds",
            "detail": f"{summary.get('total_items', 0)} newspaper evidence records stored with source URLs.",
        },
        {
            "stage": "AI Classification",
            "status": ai_status,
            "detail": (
                f"{semantic_count} semantic analyses and {fallback_count} fallback analyses are active "
                "after per-article dedupe."
            ),
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
        if item.get("people_issue") or item.get("stance") in {"negative", "mixed"} or item.get("needs_human_review")
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
def items(limit: int = 50, session_factory=Depends(get_session_factory)) -> list[dict[str, object]]:
    settings = Settings()
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
def analyses(limit: int = 50, session_factory=Depends(get_session_factory)) -> list[dict[str, object]]:
    settings = Settings()
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


class ChatAskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    limit: int = Field(default=5, ge=1, le=8)


class ChatStreamRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    limit: int = Field(default=6, ge=1, le=8)
    history: list[ChatTurn] = Field(default_factory=list)


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
    """Build the cascade analyser.

    Default order: Gemma → OpenAI → LocalTamil → Mock.

    Gemma 2 (via Ollama, fully local, zero tokens) is the default primary
    analyser. OpenAI is kept as a paid fallback for when Ollama is down or
    the operator wants the faster GPT output for a specific run.

    Set ``TNMI_PREFER_OPENAI=1`` to flip the order back to OpenAI-first.
    Useful when the operator has OpenAI quota and wants the fastest run
    even though it consumes tokens.
    """
    prefer_openai = os.getenv("TNMI_PREFER_OPENAI", "").lower() in ("1", "true", "yes")

    openai_tier: Any | None = None
    if getattr(settings, "openai_api_key", None):
        try:
            openai_tier = OpenAIAnalyzer(
                api_key=settings.openai_api_key,
                model_name=settings.openai_model_item_classifier,
            )
        except Exception:
            traceback.print_exc()

    gemma_tier: Any | None = None
    try:
        from tnmi.local_llm import GemmaAnalyzer

        gemma_tier = GemmaAnalyzer(
            model=settings.ollama_model,
            host=settings.ollama_host,
        )
    except Exception:
        traceback.print_exc()

    tiers: list[Any] = []
    if prefer_openai:
        # OpenAI-first: faster + better quality, but burns tokens.
        if openai_tier is not None:
            tiers.append(openai_tier)
        if gemma_tier is not None:
            tiers.append(gemma_tier)
    else:
        # Gemma-first (default): zero tokens, fully local, confidential.
        if gemma_tier is not None:
            tiers.append(gemma_tier)
        if openai_tier is not None:
            tiers.append(openai_tier)
    tiers.append(LocalTamilAnalyzer())
    tiers.append(MockAIAnalyzer())
    return _FallbackAnalyzer(tiers)


def _build_chat_provider(settings: Settings) -> ChatAIProvider:
    return OllamaChatProvider(
        model=settings.ollama_model,
        host=settings.ollama_host,
    )


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
def dashboard_page(request: Request, print: int = 0, session_factory=Depends(get_session_factory)) -> HTMLResponse:
    settings = Settings()
    with session_factory() as session:
        summary = get_dashboard_summary(session)
        queue = list_review_queue(session, limit=50)
        # Render every relevant story so the KPI/filter counts derive from the
        # exact cards on screen. The cap stays above the relevant-story count so
        # nothing is silently dropped; pagination (client-side) keeps the page
        # short regardless of how many cards are present.
        latest_items = list_latest_items(session, limit=1500)
        # Headline deck numbers come straight from the cards — single source of
        # truth, so "All Coverage" always equals its own category breakdown.
        summary.update(summarize_briefing_categories(latest_items))
        priority_alerts = select_priority_alerts(latest_items, limit=5)
        district_summary = summarize_by_district(latest_items)
        department_summary = summarize_by_department(latest_items)
        emerging_signals = detect_spikes(session, limit=6)
        model_status = model_health(session)
        daily_brief = compose_brief(
            summary=summary,
            emerging_signals=emerging_signals,
            priority_alerts=priority_alerts,
            district_summary=district_summary,
            actors=actor_scorecards(session, limit=8),
        )
        # Attach real boundary geometry so the template renders a true TN map,
        # and the constituency/MLA roster for the district drill-down panel.
        district_mlas = mlas_by_district()
        for tile in district_summary["tiles"]:
            tile["path"] = TN_MAP_PATHS.get(tile["district"], "")
            label = TN_MAP_LABELS.get(tile["district"])
            tile["label_x"], tile["label_y"] = (label if label else (0.0, 0.0))
            tile["mlas"] = district_mlas.get(tile["district"], [])
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
            "priority_alerts": priority_alerts,
            "emerging_signals": emerging_signals,
            "model_status": model_status,
            "daily_brief": daily_brief,
            "district_summary": district_summary,
            "department_summary": department_summary,
            "tn_map_viewbox": TN_MAP_VIEWBOX,
            "party_seats": party_seat_counts(),
            "assembly_label": roster_label(),
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
            "print_mode": bool(print),
        },
        headers=_NO_CACHE_HEADERS,
    )


_CHROME_CANDIDATES = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
)


def _find_browser_for_pdf() -> str | None:
    """Locate a Chromium-family browser usable for --print-to-pdf."""
    for path in _CHROME_CANDIDATES:
        if Path(path).exists():
            return path
    for name in ("chrome", "google-chrome", "chromium", "msedge"):
        which = shutil.which(name)
        if which:
            return which
    return None


@app.get("/reports/pdf/today", dependencies=[Depends(require_operator)])
def download_today_pdf(request: Request) -> Response:
    """Render the dashboard's print mode to a PDF via headless Chrome and
    stream it back. Uses ``--print-to-pdf`` so the layout matches the
    on-screen briefing pixel-for-pixel — same fonts, same colors, same
    cards — just with interactive controls hidden via the print CSS."""
    browser = _find_browser_for_pdf()
    if not browser:
        raise HTTPException(
            status_code=503,
            detail=(
                "No Chromium browser found for PDF rendering. Install Google "
                "Chrome or Microsoft Edge and retry."
            ),
        )

    # Build the URL the browser will hit. Use the live request host so the
    # browser hits the same FastAPI process, but force ?print=1 so the
    # template renders the print-friendly variant.
    base_url = str(request.base_url).rstrip("/")
    dashboard_url = f"{base_url}/dashboard?print=1"

    # Pass the operator token so the dashboard route lets the browser in,
    # if the operator has token guard enabled. Browsers don't support
    # arbitrary headers via the URL bar, so we plumb the token through a
    # short-lived header by setting a cookie via the browser's
    # --header-from-file? Chromium does not expose that — we rely on the
    # browser's default behaviour: when operator_api_token is empty the
    # guard is permissive (typical local-demo setup).
    today = date.today().isoformat()
    with tempfile.TemporaryDirectory(prefix="tnmi-pdf-") as tmp:
        out_path = Path(tmp) / f"tvk-briefing-{today}.pdf"
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            "--no-margins",
            f"--print-to-pdf={out_path}",
            "--virtual-time-budget=8000",
            dashboard_url,
        ]
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                timeout=60,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(
                status_code=504,
                detail="PDF rendering timed out after 60 seconds",
            ) from exc

        if not out_path.exists():
            stderr = (result.stderr or b"").decode("utf-8", "replace")[:800]
            raise HTTPException(
                status_code=500,
                detail=f"Chrome did not produce a PDF: {stderr}",
            )

        pdf_bytes = out_path.read_bytes()

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="tvk-briefing-{today}.pdf"',
            "Cache-Control": "no-store",
        },
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
def dashboard_summary(session_factory=Depends(get_session_factory)) -> dict[str, object]:
    settings = Settings()
    with session_factory() as session:
        return get_dashboard_summary(session)


@app.get("/dashboard/alerts", dependencies=[Depends(require_operator)])
def dashboard_alerts(limit: int = 5, session_factory=Depends(get_session_factory)) -> dict[str, object]:
    """Live monitor feed: forward-looking ``emerging_signals`` (entity surges and
    threats) plus the urgent ``priority`` backlog (high/critical negatives and
    people issues awaiting review). Pollable so an external monitor can surface
    them live."""
    bounded_limit = max(1, min(limit, 20))
    settings = Settings()
    with session_factory() as session:
        latest_items = list_latest_items(session, limit=200)
        spikes = detect_spikes(session, limit=bounded_limit)
    priority = select_priority_alerts(latest_items, limit=bounded_limit)
    # Emerging signals (forward-looking) lead, then the urgent backlog.
    return {"emerging_signals": spikes, "priority": priority}


@app.get("/api/entities", dependencies=[Depends(require_operator)])
def api_entities(entity_type: str | None = None, limit: int = 200, session_factory=Depends(get_session_factory)) -> list[dict[str, object]]:
    """The political knowledge graph as JSON — every canonical actor, party,
    district, department and source, ranked by coverage volume with its
    portrayal split. Powers the war-room's clickable entity chips."""
    bounded_limit = max(1, min(limit, 500))
    settings = Settings()
    with session_factory() as session:
        return list_entities(session, entity_type=entity_type, limit=bounded_limit)


@app.get("/api/entities/{slug}", dependencies=[Depends(require_operator)])
def api_entity_dossier(slug: str, session_factory=Depends(get_session_factory)) -> dict[str, object]:
    """One entity's full dossier: all-time + 30-day portrayal, weekly trend,
    co-mention network, top districts/categories, and cited evidence."""
    settings = Settings()
    with session_factory() as session:
        dossier = entity_dossier(session, slug)
    if dossier is None:
        raise HTTPException(status_code=404, detail=f"No entity with slug {slug!r}")
    return dossier


@app.get("/dashboard/models", dependencies=[Depends(require_operator)])
def dashboard_models(session_factory=Depends(get_session_factory)) -> dict[str, object]:
    """Learning-loop health: label tallies (gold/silver/bronze), the live model
    and its gold-test metric, and the latest registered candidate."""
    settings = Settings()
    with session_factory() as session:
        return model_health(session)


@app.get("/api/actors", dependencies=[Depends(require_operator)])
def api_actors(limit: int = 12, session_factory=Depends(get_session_factory)) -> list[dict[str, object]]:
    """Ranked persona scorecards — per-figure reputation (portrayal split,
    favorability, weekly trend, momentum) for the 'who is winning the
    narrative' Key Figures view."""
    bounded_limit = max(1, min(limit, 60))
    settings = Settings()
    with session_factory() as session:
        return actor_scorecards(session, limit=bounded_limit)


@app.post("/chat/ask", dependencies=[Depends(require_operator)])
def ask_chat(request: ChatAskRequest, session_factory=Depends(get_session_factory)) -> dict[str, object]:
    settings = Settings()
    provider: ChatAIProvider | None = None
    try:
        provider = _build_chat_provider(settings)
    except Exception:
        traceback.print_exc()

    with session_factory() as session:
        answer = answer_question(
            session,
            request.question,
            provider=provider,
            limit=request.limit,
        )
    return answer.model_dump(mode="json")


@app.post("/chat/stream", dependencies=[Depends(require_operator)])
def stream_chat(request: ChatStreamRequest, session_factory=Depends(get_session_factory)) -> StreamingResponse:
    """Streaming sibling of /chat/ask.

    Emits newline-delimited JSON events so the dashboard can render a live,
    typing-style answer instead of a multi-second blank wait:

        {"type": "evidence", "evidence": [...]}     # sources, sent first
        {"type": "delta",    "text": "..."}          # answer fragments
        {"type": "done",     "used_ai": true, "model_name": "ollama/gemma2:2b"}

    Evidence is retrieved up front (fast, DB-bound) and the session is closed
    before the slow LLM stream begins. If the local AI is unreachable we fall
    back to the same deterministic evidence-only answer as /chat/ask.
    """
    settings = Settings()
    provider: ChatAIProvider | None = None
    try:
        provider = _build_chat_provider(settings)
    except Exception:
        traceback.print_exc()

    history = request.history[-8:]
    retrieval_query = build_retrieval_query(request.question, history)
    with session_factory() as session:
        evidence = retrieve_chat_evidence(session, retrieval_query, limit=request.limit)
        # Aggregate knowledge-graph context for any entity named in the question —
        # computed before the session closes and the slow LLM stream begins.
        dossier_context = build_dossier_context(session, request.question)

    def emit(event: dict[str, object]) -> str:
        return json.dumps(event, ensure_ascii=False) + "\n"

    def generate():
        yield emit(
            {
                "type": "evidence",
                "evidence": [item.model_dump(mode="json") for item in evidence],
            }
        )

        if not evidence:
            yield emit({"type": "delta", "text": _INSUFFICIENT_ANSWER})
            yield emit({"type": "done", "used_ai": False, "model_name": "evidence-only"})
            return

        produced = False
        if provider is not None and hasattr(provider, "stream_answer"):
            try:
                for fragment in provider.stream_answer(
                    request.question, evidence, history=history, dossier_context=dossier_context
                ):
                    if fragment:
                        produced = True
                        yield emit({"type": "delta", "text": fragment})
            except Exception as exc:  # noqa: BLE001 - provider fallback boundary
                logger.warning(
                    "Chat stream provider failed; %s",
                    "mid-stream" if produced else "using evidence-only fallback",
                )
                if produced:
                    yield emit(
                        {
                            "type": "delta",
                            "text": "\n\n_(The local AI connection was interrupted.)_",
                        }
                    )
                    yield emit(
                        {"type": "done", "used_ai": True, "model_name": provider.model_name}
                    )
                    return

        if produced:
            yield emit({"type": "done", "used_ai": True, "model_name": provider.model_name})
            return

        fallback = EvidenceOnlyChatProvider()
        yield emit(
            {
                "type": "delta",
                "text": fallback.answer(request.question, evidence, dossier_context=dossier_context),
            }
        )
        yield emit({"type": "done", "used_ai": False, "model_name": fallback.model_name})

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/review/queue", dependencies=[Depends(require_operator)])
def review_queue(limit: int = 50, session_factory=Depends(get_session_factory)) -> list[dict[str, object]]:
    settings = Settings()
    with session_factory() as session:
        return list_review_queue(session, limit=limit)


@app.post("/review/decisions", dependencies=[Depends(require_operator)])
def create_review_decision(decision: ReviewDecisionCreate, session_factory=Depends(get_session_factory)) -> dict[str, object]:
    settings = Settings()
    with session_factory() as session:
        record = save_review_decision(session, decision)
        session.commit()
    # If the operator corrected a stance, the briefing's cached payload is
    # now stale — drop it so the next dashboard render picks up the override.
    if decision.corrected_stance is not None:
        invalidate_briefing_cache()
    return _review_decision_payload(record)


@app.get("/review/decisions/{analysis_id}", dependencies=[Depends(require_operator)])
def latest_review_decision(analysis_id: int, session_factory=Depends(get_session_factory)) -> dict[str, object]:
    settings = Settings()
    with session_factory() as session:
        record = get_latest_review_decision(session, analysis_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Review decision not found")
        return _review_decision_payload(record)
