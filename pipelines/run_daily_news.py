from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tnmi.ai import MockAIAnalyzer, OpenAIAnalyzer
from tnmi.config import Settings, load_newspaper_sources
from tnmi.pipeline import DailyNewsPipeline, RequestsNewsClient
from tnmi.reports import build_daily_report_data, render_daily_news_markdown, write_report
from tnmi.storage import create_session_factory, init_db


def parse_news_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def build_analyzer(
    settings: Settings,
    *,
    mock_ai: bool,
    subject: str = "TVK",
    leader: str = "Vijay",
    governing: bool = True,
):
    if mock_ai:
        return MockAIAnalyzer()
    if settings.openai_api_key:
        return OpenAIAnalyzer(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model_item_classifier,
            subject=subject,
            leader=leader,
            governing=governing,
        )
    raise RuntimeError("OPENAI_API_KEY is required unless --mock-ai is provided")


def ingest_all_tenants(settings: Settings, *, mock_ai: bool) -> None:
    """Run the daily pipeline once per active tenant, each through its OWN lens
    and into its OWN database. ponytail: sources are re-fetched per tenant —
    LLM classification (unavoidably per-lens) dominates cost, so fetch-sharing
    is not worth the caching complexity until proven otherwise."""
    from tnmi.tenancy import ControlPlane

    control = ControlPlane(settings.control_database_url, tenants_dir=settings.tenants_dir)
    sources = load_newspaper_sources(settings.news_source_config)
    client = RequestsNewsClient()
    for tenant in control.list_tenants():
        if tenant.status != "active":
            continue
        cfg = control.tenant_config(tenant)
        analyzer = build_analyzer(
            settings,
            mock_ai=mock_ai,
            subject=cfg.subject_party,
            leader=cfg.subject_leader,
            governing=cfg.governing,
        )
        result = DailyNewsPipeline(
            session_factory=control.session_factory_for(tenant),
            news_client=client,
            analyzer=analyzer,
        ).run(sources)
        print(
            f"tenant={tenant.slug} lens={cfg.subject_party}/"
            f"{'gov' if cfg.governing else 'opp'} items_saved={result.items_saved} "
            f"analyses_saved={result.analyses_saved}"
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=parse_news_date, default=date.today())
    parser.add_argument("--mock-ai", action="store_true")
    parser.add_argument("--all-tenants", action="store_true", help="ingest every active SaaS tenant")
    args = parser.parse_args(argv)

    settings = Settings()
    # Multi-tenant deployments fan out per tenant automatically, so the nightly
    # DAG (which calls main([])) needs no change to serve a fleet.
    if args.all_tenants or getattr(settings, "multi_tenant", False):
        return ingest_all_tenants(settings, mock_ai=args.mock_ai)

    sources = load_newspaper_sources(settings.news_source_config)
    try:
        analyzer = build_analyzer(settings, mock_ai=args.mock_ai)
    except RuntimeError as exc:
        parser.error(str(exc))

    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    pipeline = DailyNewsPipeline(
        session_factory=session_factory,
        news_client=RequestsNewsClient(),
        analyzer=analyzer,
    )
    result = pipeline.run(sources)
    with session_factory() as session:
        report_data = build_daily_report_data(session, args.date)
    report_markdown = render_daily_news_markdown(
        report_date=args.date,
        stance_counts=report_data["stance_counts"],
        top_items=report_data["top_items"],
    )
    report_path = write_report(
        report_markdown,
        settings.report_output_dir,
        f"daily-news-{args.date.isoformat()}.md",
    )
    print(
        f"date={args.date.isoformat()} items_seen={result.items_seen} "
        f"items_saved={result.items_saved} analyses_saved={result.analyses_saved} "
        f"failures={result.failures} sources_skipped={result.sources_skipped} "
        f"report_path={report_path}"
    )


if __name__ == "__main__":
    main()
