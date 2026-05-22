from __future__ import annotations

from datetime import date
from pathlib import Path
import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from tnmi.storage import AIAnalysisRecord, RawItemRecord


_MARKDOWN_CONTROL_CHARS = "\\*_[]()#<>|`"
_REQUIRED_TOP_ITEM_KEYS = {"source_name", "title", "stance", "summary", "url"}


def _normalize_inline_text(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    return "".join(
        f"\\{char}" if char in _MARKDOWN_CONTROL_CHARS else char
        for char in normalized
    )


def _validate_top_item(item: dict[str, str], index: int) -> None:
    missing_keys = sorted(_REQUIRED_TOP_ITEM_KEYS - item.keys())
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise ValueError(f"top item {index} is missing required keys: {missing}")


def render_daily_news_markdown(
    *,
    report_date: date,
    stance_counts: dict[str, int],
    top_items: list[dict[str, str]],
) -> str:
    lines = [
        f"# Daily Newspaper Intelligence Report - {report_date.isoformat()}",
        "",
        "## Stance Split",
        "",
        f"- Positive: {stance_counts.get('positive', 0)}",
        f"- Negative: {stance_counts.get('negative', 0)}",
        f"- Neutral: {stance_counts.get('neutral', 0)}",
        f"- Mixed: {stance_counts.get('mixed', 0)}",
        "",
        "## Top Items",
        "",
    ]
    for index, item in enumerate(top_items):
        _validate_top_item(item, index)
        lines.extend(
            [
                f"### {_normalize_inline_text(item['title'])}",
                "",
                f"- Source: {_normalize_inline_text(item['source_name'])}",
                f"- Stance: {_normalize_inline_text(item['stance'])}",
                f"- Summary: {_normalize_inline_text(item['summary'])}",
                f"- URL: {_normalize_inline_text(item['url'])}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _record_report_date(raw: RawItemRecord) -> date:
    timestamp = raw.published_at or raw.ingested_at
    return timestamp.date()


def build_daily_report_data(session: Session, report_date: date) -> dict[str, object]:
    rows = session.execute(
        select(RawItemRecord, AIAnalysisRecord)
        .join(AIAnalysisRecord, AIAnalysisRecord.raw_item_id == RawItemRecord.id)
        .order_by(RawItemRecord.id)
    ).all()

    stance_counts = {"positive": 0, "negative": 0, "neutral": 0, "mixed": 0}
    top_items: list[dict[str, str]] = []
    for raw, analysis in rows:
        if _record_report_date(raw) != report_date:
            continue
        stance = analysis.stance_toward_government
        stance_counts[stance] = stance_counts.get(stance, 0) + 1
        top_items.append(
            {
                "source_name": raw.source_name,
                "title": raw.title or raw.source_url,
                "stance": stance,
                "summary": analysis.summary_english or analysis.summary_original,
                "url": raw.source_url,
            }
        )
    return {"stance_counts": stance_counts, "top_items": top_items}


def write_report(markdown: str, output_dir: Path, filename: str) -> Path:
    filename_path = Path(filename)
    if filename_path.is_absolute() or filename_path.name != filename:
        raise ValueError("unsafe filename: must be a plain filename")

    resolved_output_dir = output_dir.resolve()
    path = (resolved_output_dir / filename).resolve()
    try:
        path.relative_to(resolved_output_dir)
    except ValueError as exc:
        raise ValueError("unsafe filename: resolved path escapes output directory") from exc

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path
