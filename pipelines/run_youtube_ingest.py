"""Pull recent YouTube videos from registered TN news channels, transcribe
the Tamil audio with faster-whisper, and persist the result through the same
storage + AIAnalysis pipeline as newspaper articles.

Usage:

    # Inspect what would run (no API calls)
    python -m pipelines.run_youtube_ingest --dry-run

    # Run for real (requires YOUTUBE_API_KEY in env or .env)
    python -m pipelines.run_youtube_ingest --max-videos 2

    # Use a smaller / faster Whisper model
    python -m pipelines.run_youtube_ingest --whisper-model tiny

Requirements (installed earlier in this Phase D pass):
    pip install yt-dlp faster-whisper google-api-python-client
Plus ffmpeg on PATH (yt-dlp needs it for audio extraction).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tnmi.ai import MockAIAnalyzer, OpenAIAnalyzer, PROMPT_VERSION
from tnmi.config import Settings
from tnmi.local_models import LocalTamilAnalyzer
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item
from tnmi.youtube import ingest_channel, load_youtube_channels


def _pick_analyzer(settings: Settings):
    """Same cascade order as the API: OpenAI > LocalTamil > Mock."""
    if getattr(settings, "openai_api_key", None):
        try:
            return OpenAIAnalyzer(
                api_key=settings.openai_api_key,
                model_name=settings.openai_model_item_classifier,
            )
        except Exception:
            pass
    return LocalTamilAnalyzer()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "sources.youtube.yaml"),
        help="Path to YouTube channel registry (default: configs/sources.youtube.yaml)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=2,
        help="Max videos to ingest per channel per run (default: 2)",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="faster-whisper model size (default: small ≈470MB)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would run without making API calls or downloading audio",
    )
    parser.add_argument(
        "--mock-ai",
        action="store_true",
        help="Bypass OpenAI / LocalTamil and use MockAIAnalyzer for the transcripts",
    )
    args = parser.parse_args(argv)

    channels = load_youtube_channels(Path(args.config))
    if not channels:
        parser.error(f"No channels found in {args.config}")

    active = [c for c in channels if c.active]
    print(f"channels_in_registry={len(channels)} active={len(active)}")
    if args.dry_run:
        for c in channels:
            flag = "ACTIVE" if c.active else "inactive"
            print(f"  [{flag}] {c.name}  channel_id={c.channel_id}  scope={c.coverage_scope}")
        if not active:
            print("Nothing to ingest — flip `active: true` on a channel after verifying its ID.")
        return

    if not active:
        parser.error("No active channels. Edit configs/sources.youtube.yaml and flip an entry to `active: true`.")

    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key:
        parser.error("YOUTUBE_API_KEY not set in environment. Get a free key from Google Cloud Console.")

    settings = Settings()
    analyzer = MockAIAnalyzer() if args.mock_ai else _pick_analyzer(settings)
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)

    with tempfile.TemporaryDirectory(prefix="tnmi-yt-") as tmp:
        audio_dir = Path(tmp)
        total_seen = total_saved = total_analyses = total_failed = 0
        for channel in active:
            print(f"\n== {channel.name} ==")
            result = ingest_channel(
                channel,
                youtube_api_key=api_key,
                audio_dir=audio_dir,
                max_videos_per_channel=args.max_videos,
                whisper_model_size=args.whisper_model,
            )
            total_seen += result.discovered
            total_failed += result.failed
            print(
                f"  discovered={result.discovered} transcribed={result.transcribed} failed={result.failed}"
            )

            for item in result.items:
                try:
                    with session_factory() as session:
                        raw = save_raw_item(session, item)
                        analysis = analyzer.analyze(item)
                        save_ai_analysis(
                            session,
                            raw.id,
                            analysis,
                            model_name=analyzer.model_name,
                            prompt_version=PROMPT_VERSION,
                        )
                        session.commit()
                    total_saved += 1
                    total_analyses += 1
                    print(f"    saved: {item.title[:80]!r}")
                except Exception as exc:  # noqa: BLE001
                    total_failed += 1
                    print(f"    FAILED to persist: {exc}")

        print(
            f"\ndone discovered={total_seen} saved={total_saved} analyses={total_analyses} "
            f"failed={total_failed} model={analyzer.model_name}"
        )


if __name__ == "__main__":
    main()
