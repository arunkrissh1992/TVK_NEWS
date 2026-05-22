from datetime import date

import pytest

from tnmi.reports import render_daily_news_markdown, write_report


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


def test_write_report_writes_inside_output_dir(tmp_path):
    path = write_report("# Report\n", tmp_path, "daily.md")

    assert path == tmp_path / "daily.md"
    assert path.read_text(encoding="utf-8") == "# Report\n"


def test_write_report_rejects_parent_path_filename(tmp_path):
    with pytest.raises(ValueError, match="unsafe filename"):
        write_report("# Report\n", tmp_path, "../outside.md")


def test_write_report_rejects_absolute_filename(tmp_path):
    with pytest.raises(ValueError, match="unsafe filename"):
        write_report("# Report\n", tmp_path, str(tmp_path.parent / "outside.md"))


def test_render_daily_news_markdown_normalizes_and_escapes_item_text():
    markdown = render_daily_news_markdown(
        report_date=date(2026, 5, 21),
        stance_counts={},
        top_items=[
            {
                "source_name": "Daily\tName",
                "title": "# Bad\n- item [link](https://example.com) *bold* `code`",
                "stance": "neutral",
                "summary": "line 1\r\n## heading <tag> | _value_",
                "url": "https://example.com/a\n#fragment",
            }
        ],
    )

    assert "### \\# Bad - item \\[link\\]\\(https://example.com\\) \\*bold\\* \\`code\\`" in markdown
    assert "- Source: Daily Name" in markdown
    assert "- Summary: line 1 \\#\\# heading \\<tag\\> \\| \\_value\\_" in markdown
    assert "- URL: https://example.com/a \\#fragment" in markdown


def test_render_daily_news_markdown_requires_top_item_fields():
    with pytest.raises(ValueError, match="item 0.*summary"):
        render_daily_news_markdown(
            report_date=date(2026, 5, 21),
            stance_counts={},
            top_items=[
                {
                    "source_name": "Example Tamil Daily",
                    "title": "Title",
                    "stance": "positive",
                    "url": "https://example.com/a",
                }
            ],
        )
