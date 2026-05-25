"""YouTube ingestion — Phase D.

Pipeline shape mirrors the newspaper pipeline:

    YouTube Data API (channels.list / search.list)
        ↓
    Discover recent videos per registered channel
        ↓
    yt-dlp downloads the audio (mp3/m4a) to a temp file
        ↓
    faster-whisper transcribes the audio (Tamil model)
        ↓
    Normalised into a NormalizedItem with source_type = YOUTUBE
        ↓
    Same AIAnalysis pipeline as newspapers (stance / Party / People / Why / Next)

This module is intentionally split into small functions you can run
piecewise from notebooks or the dashboard's Pull Latest button.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tnmi.contracts import NormalizedItem, SourceType, YouTubeChannelSource


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YouTube Data API — list recent uploads per channel
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YouTubeVideoRef:
    video_id: str
    channel_id: str
    title: str
    description: str
    published_at: datetime | None
    url: str


def list_recent_videos(
    channel: YouTubeChannelSource,
    *,
    api_key: str,
    max_results: int = 5,
) -> list[YouTubeVideoRef]:
    """Use the official YouTube Data API to list the channel's newest uploads.

    Requires an API key from Google Cloud Console (free tier covers 10k
    units/day; one search.list call costs 100 units). Quota-aware callers
    should keep ``max_results`` small per channel and batch across days.
    """
    try:
        from googleapiclient.discovery import build  # type: ignore[import-not-found]
    except ImportError as exc:  # noqa: TRY003
        raise RuntimeError(
            "google-api-python-client not installed. "
            "Install with: pip install google-api-python-client"
        ) from exc

    if not api_key:
        raise ValueError("YouTube Data API key is required")

    youtube = build("youtube", "v3", developerKey=api_key, cache_discovery=False)

    response = (
        youtube.search()
        .list(
            channelId=channel.channel_id,
            part="snippet",
            order="date",
            maxResults=max(1, min(max_results, 25)),
            type="video",
        )
        .execute()
    )

    videos: list[YouTubeVideoRef] = []
    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue
        published_raw = snippet.get("publishedAt")
        published_at: datetime | None = None
        if published_raw:
            try:
                # API returns 2024-01-15T08:30:00Z
                published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
            except ValueError:
                published_at = None
        videos.append(
            YouTubeVideoRef(
                video_id=video_id,
                channel_id=channel.channel_id,
                title=snippet.get("title", "").strip(),
                description=snippet.get("description", "").strip(),
                published_at=published_at,
                url=f"https://www.youtube.com/watch?v={video_id}",
            )
        )
    return videos


# ---------------------------------------------------------------------------
# Audio download via yt-dlp
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DownloadedAudio:
    path: Path
    duration_seconds: float | None
    file_size_bytes: int


def download_audio(
    video: YouTubeVideoRef,
    *,
    output_dir: Path,
    timeout_seconds: int = 180,
) -> DownloadedAudio:
    """Download the audio track of a YouTube video to a local mp3 file.

    Uses yt-dlp's Python API. Output is an m4a/mp3 file under ``output_dir``
    named ``<video_id>.<ext>``. Caller is responsible for cleaning up.
    """
    try:
        import yt_dlp  # type: ignore[import-not-found]
    except ImportError as exc:  # noqa: TRY003
        raise RuntimeError(
            "yt-dlp not installed. Install with: pip install yt-dlp"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    target_template = str(output_dir / f"{video.video_id}.%(ext)s")

    options = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": target_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": timeout_seconds,
        # Prefer not to mux video frames — audio is enough for transcription
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
    }

    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(video.url, download=True)

    # Resolve the actual output filename yt-dlp produced.
    candidates = sorted(output_dir.glob(f"{video.video_id}.*"))
    if not candidates:
        raise FileNotFoundError(f"yt-dlp did not produce a file for {video.video_id}")
    audio_path = candidates[0]
    size = audio_path.stat().st_size
    duration = float(info.get("duration") or 0) or None
    return DownloadedAudio(path=audio_path, duration_seconds=duration, file_size_bytes=size)


# ---------------------------------------------------------------------------
# Transcription via faster-whisper
# ---------------------------------------------------------------------------


_WHISPER_MODEL_CACHE: dict[str, object] = {}


def transcribe_tamil(
    audio_path: Path,
    *,
    model_size: str = "small",
    compute_type: str = "int8",
) -> str:
    """Transcribe a Tamil audio file using faster-whisper.

    First call downloads the model (small: ~470MB, medium: ~1.5GB) to
    ~/.cache/huggingface/. Subsequent calls reuse the in-memory model.
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:  # noqa: TRY003
        raise RuntimeError(
            "faster-whisper not installed. Install with: pip install faster-whisper"
        ) from exc

    cache_key = f"{model_size}:{compute_type}"
    model = _WHISPER_MODEL_CACHE.get(cache_key)
    if model is None:
        logger.info(
            "Loading faster-whisper %s (compute_type=%s); first run downloads the model.",
            model_size,
            compute_type,
        )
        model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
        _WHISPER_MODEL_CACHE[cache_key] = model

    segments_iter, _info = model.transcribe(  # type: ignore[attr-defined]
        str(audio_path),
        language="ta",
        beam_size=5,
        condition_on_previous_text=False,
    )

    return " ".join((segment.text or "").strip() for segment in segments_iter).strip()


# ---------------------------------------------------------------------------
# Normalisation — wrap a transcript into a NormalizedItem so it flows through
# the existing AIAnalysis pipeline alongside newspaper articles.
# ---------------------------------------------------------------------------


def normalise_youtube_transcript(
    channel: YouTubeChannelSource,
    video: YouTubeVideoRef,
    transcript: str,
) -> NormalizedItem:
    description = video.description or ""
    body = "\n\n".join(filter(None, [description, transcript])).strip()
    return NormalizedItem(
        source_type=SourceType.YOUTUBE,
        source_name=channel.name,
        source_url=video.url,
        published_at=video.published_at,
        language=channel.language_hint,
        title=video.title,
        raw_text_original=body,
        clean_text_original=body,
        metadata={
            "youtube_video_id": video.video_id,
            "youtube_channel_id": video.channel_id,
            "transcript_chars": len(transcript),
            "transcript_source": "faster-whisper",
        },
    )


# ---------------------------------------------------------------------------
# High-level pipeline: discover → download → transcribe → normalise
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YouTubeIngestResult:
    channel: str
    discovered: int
    transcribed: int
    failed: int
    items: list[NormalizedItem]


def ingest_channel(
    channel: YouTubeChannelSource,
    *,
    youtube_api_key: str,
    audio_dir: Path,
    max_videos_per_channel: int = 3,
    whisper_model_size: str = "small",
    keep_audio: bool = False,
) -> YouTubeIngestResult:
    """End-to-end: pull latest videos, transcribe, return NormalizedItems.

    Caller persists the items via tnmi.storage.save_raw_item and runs the
    AIAnalyzer over each one to produce the same Party/People/Why/Next-step
    briefing lines used for newspaper articles.
    """
    videos = list_recent_videos(channel, api_key=youtube_api_key, max_results=max_videos_per_channel)
    items: list[NormalizedItem] = []
    failed = 0

    for video in videos:
        try:
            downloaded = download_audio(video, output_dir=audio_dir)
            try:
                transcript = transcribe_tamil(downloaded.path, model_size=whisper_model_size)
            finally:
                if not keep_audio:
                    try:
                        downloaded.path.unlink(missing_ok=True)
                    except OSError:
                        pass
            if not transcript:
                logger.warning("Empty transcript for %s (%s); skipping", video.video_id, channel.name)
                failed += 1
                continue
            items.append(normalise_youtube_transcript(channel, video, transcript))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to ingest video %s: %s", video.video_id, exc)
            failed += 1

    return YouTubeIngestResult(
        channel=channel.name,
        discovered=len(videos),
        transcribed=len(items),
        failed=failed,
        items=items,
    )


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_youtube_channels(config_path: Path) -> list[YouTubeChannelSource]:
    """Parse the YouTube channel registry yaml."""
    import yaml

    if not config_path.exists():
        return []

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_channels: Iterable[dict] = data.get("channels", []) or []
    return [YouTubeChannelSource.model_validate(row) for row in raw_channels]
