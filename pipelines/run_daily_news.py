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
from tnmi.storage import create_session_factory, init_db


def parse_news_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO date: {value}") from exc


def build_analyzer(settings: Settings):
    if settings.openai_api_key:
        return OpenAIAnalyzer(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model_item_classifier,
        )
    return MockAIAnalyzer()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=parse_news_date, default=date.today())
    args = parser.parse_args(argv)

    settings = Settings()
    sources = load_newspaper_sources(settings.news_source_config)
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    pipeline = DailyNewsPipeline(
        session_factory=session_factory,
        news_client=RequestsNewsClient(),
        analyzer=build_analyzer(settings),
    )
    result = pipeline.run(sources)
    print(
        f"date={args.date.isoformat()} items_seen={result.items_seen} "
        f"items_saved={result.items_saved} analyses_saved={result.analyses_saved} "
        f"failures={result.failures}"
    )


if __name__ == "__main__":
    main()
