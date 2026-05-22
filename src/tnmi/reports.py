from __future__ import annotations

from datetime import date
from pathlib import Path


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
    for item in top_items:
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"- Source: {item['source_name']}",
                f"- Stance: {item['stance']}",
                f"- Summary: {item['summary']}",
                f"- URL: {item['url']}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_report(markdown: str, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(markdown, encoding="utf-8")
    return path
