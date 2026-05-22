from datetime import date

from tnmi.reports import render_daily_news_markdown


def test_render_daily_news_markdown_includes_stance_counts():
    markdown = render_daily_news_markdown(
        report_date=date(2026, 5, 21),
        stance_counts={"positive": 2, "negative": 1, "neutral": 3, "mixed": 1},
        top_items=[
            {
                "source_name": "Example Tamil Daily",
                "title": "தமிழக அரசு புதிய திட்டம்",
                "stance": "positive",
                "summary": "சாதகமான செய்தி.",
                "url": "https://example.com/a",
            }
        ],
    )

    assert "# Daily Newspaper Intelligence Report - 2026-05-21" in markdown
    assert "- Positive: 2" in markdown
    assert "தமிழக அரசு புதிய திட்டம்" in markdown
