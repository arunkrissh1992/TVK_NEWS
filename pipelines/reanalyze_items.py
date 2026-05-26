"""Re-run AI analysis on existing raw_items.

Use this when the prompt version has changed and you want fresh briefing-style
analyses to replace older mock or earlier-prompt-version results, without
re-fetching the underlying newspaper articles.

Example:

    python -m pipelines.reanalyze_items --limit 100
    python -m pipelines.reanalyze_items --mock-ai          # development fallback
    python -m pipelines.reanalyze_items --only-mock        # only items whose
                                                            # latest analysis is mock

The command never re-runs an analysis that already exists for the same
(raw_item_id, model_name, prompt_version) tuple, so it is safe to invoke twice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from tnmi.ai import AIAnalyzer, MockAIAnalyzer, OpenAIAnalyzer, PROMPT_VERSION
from tnmi.local_models import LocalTamilAnalyzer
from tnmi.config import Settings
from tnmi.contracts import NormalizedItem, SourceType
from tnmi.storage import (
    AIAnalysisRecord,
    RawItemRecord,
    create_session_factory,
    get_ai_analysis,
    init_db,
    save_ai_analysis,
)


def build_analyzer(
    settings: Settings,
    *,
    mock_ai: bool,
    local_tamil: bool = False,
    gemma: bool = False,
) -> AIAnalyzer:
    if gemma:
        # Lazy import — keeps `ollama` an optional dependency.
        from tnmi.local_llm import GemmaAnalyzer

        return GemmaAnalyzer(
            model=settings.ollama_model,
            host=settings.ollama_host,
            timeout=240.0,
        )
    if local_tamil:
        return LocalTamilAnalyzer()
    if mock_ai:
        return MockAIAnalyzer()
    if settings.openai_api_key:
        return OpenAIAnalyzer(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model_item_classifier,
        )
    raise RuntimeError(
        "OPENAI_API_KEY is required unless --mock-ai / --local-tamil / --gemma is provided"
    )


def _to_normalized_item(record: RawItemRecord) -> NormalizedItem:
    try:
        source_type = SourceType(record.source_type)
    except ValueError:
        source_type = SourceType.NEWS
    return NormalizedItem(
        source_type=source_type,
        source_name=record.source_name,
        source_url=record.source_url,
        published_at=record.published_at,
        language=record.language,
        title=record.title,
        raw_text_original=record.raw_text_original,
        clean_text_original=record.clean_text_original,
        metadata=record.metadata_json or {},
    )


def _candidate_items(
    session_factory: sessionmaker, *, only_mock: bool, limit: int | None
) -> list[RawItemRecord]:
    with session_factory() as session:
        query = select(RawItemRecord).order_by(RawItemRecord.ingested_at.desc(), RawItemRecord.id.desc())
        if only_mock:
            mock_subquery = (
                select(AIAnalysisRecord.raw_item_id)
                .where(AIAnalysisRecord.model_name == "mock")
                .distinct()
            )
            query = query.where(RawItemRecord.id.in_(mock_subquery))
        if limit is not None:
            query = query.limit(limit)
        return list(session.scalars(query))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Maximum items to (re)analyse")
    parser.add_argument("--mock-ai", action="store_true", help="Use the MockAIAnalyzer instead of OpenAI")
    parser.add_argument(
        "--local-tamil",
        action="store_true",
        help="Use LocalTamilAnalyzer (multilingual encoder + keyword classifier) — fast, stance-only.",
    )
    parser.add_argument(
        "--gemma",
        action="store_true",
        help="Use GemmaAnalyzer (Gemma 2 2B via Ollama) — full briefing-quality LLM, fully local, no tokens. Slow on CPU (~30-60s per article).",
    )
    parser.add_argument(
        "--only-mock",
        action="store_true",
        help="Only re-analyse items that currently have a mock analysis",
    )
    parser.add_argument(
        "--prompt-version",
        default=PROMPT_VERSION,
        help="Prompt version tag for the new analyses (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        analyzer = build_analyzer(
            settings,
            mock_ai=args.mock_ai,
            local_tamil=args.local_tamil,
            gemma=args.gemma,
        )
    except RuntimeError as exc:
        parser.error(str(exc))

    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)

    items = _candidate_items(session_factory, only_mock=args.only_mock, limit=args.limit)
    if not items:
        print("no items found; nothing to do")
        return

    saved = 0
    skipped = 0
    failed = 0

    for record in items:
        normalized = _to_normalized_item(record)
        with session_factory() as session:
            existing = get_ai_analysis(
                session,
                record.id,
                model_name=analyzer.model_name,
                prompt_version=args.prompt_version,
            )
            if existing:
                skipped += 1
                continue
            try:
                analysis = analyzer.analyze(normalized)
                save_ai_analysis(
                    session,
                    record.id,
                    analysis,
                    model_name=analyzer.model_name,
                    prompt_version=args.prompt_version,
                )
                session.commit()
                saved += 1
                print(
                    f"  [{saved + skipped + failed}/{len(items)}] saved "
                    f"raw_item_id={record.id} source={record.source_name}"
                )
            except Exception as exc:  # noqa: BLE001 — surface unknown failures, keep going
                failed += 1
                session.rollback()
                print(
                    f"  [{saved + skipped + failed}/{len(items)}] FAILED "
                    f"raw_item_id={record.id} error={exc.__class__.__name__}: {exc}"
                )

    print(
        f"done items={len(items)} saved={saved} skipped={skipped} failed={failed} "
        f"model={analyzer.model_name} prompt_version={args.prompt_version}"
    )


if __name__ == "__main__":
    main()
