from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tnmi.ai import MockAIAnalyzer, OpenAIAnalyzer
from tnmi.config import Settings, load_x_handle_sources
from tnmi.storage import create_session_factory, init_db
from tnmi.x_ingestion import DailyXPipeline, TweepyXClient


def build_analyzer(settings: Settings, *, mock_ai: bool):
    if mock_ai:
        return MockAIAnalyzer()
    if settings.openai_api_key:
        return OpenAIAnalyzer(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model_item_classifier,
        )
    raise RuntimeError("OPENAI_API_KEY is required unless --mock-ai is provided")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-handles", type=int, default=None)
    parser.add_argument("--max-results", type=int, default=50)
    parser.add_argument("--mock-ai", action="store_true")
    args = parser.parse_args(argv)

    settings = Settings()
    if not settings.x_bearer_token:
        parser.error("X_BEARER_TOKEN is required for live X ingestion")

    sources = load_x_handle_sources(settings.x_source_config)
    if args.limit_handles is not None:
        sources = sources[: max(0, args.limit_handles)]

    try:
        analyzer = build_analyzer(settings, mock_ai=args.mock_ai)
    except RuntimeError as exc:
        parser.error(str(exc))

    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    pipeline = DailyXPipeline(
        session_factory=session_factory,
        x_client=TweepyXClient(settings.x_bearer_token),
        analyzer=analyzer,
    )
    result = pipeline.run(sources, max_results=max(10, min(args.max_results, 100)))
    print(
        f"handles_seen={result.handles_seen} handles_skipped={result.handles_skipped} "
        f"posts_seen={result.posts_seen} items_saved={result.items_saved} "
        f"analyses_saved={result.analyses_saved} failures={result.failures}"
    )


if __name__ == "__main__":
    main()
