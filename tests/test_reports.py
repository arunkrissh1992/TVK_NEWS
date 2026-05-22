from datetime import date, datetime, timezone

import pytest

from tnmi.reports import build_daily_report_data, render_daily_news_markdown, write_report
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item
from tests.test_storage import make_analysis, make_item


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


def test_build_daily_report_data_uses_published_date_and_ingested_fallback(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)

    published_item = make_item().model_copy(
        update={
            "source_name": "Published Source",
            "source_url": "https://example.com/published",
            "title": "Published Item",
            "published_at": datetime(2026, 5, 21, 8, 0, tzinfo=timezone.utc),
        }
    )
    fallback_item = make_item().model_copy(
        update={
            "source_name": "Fallback Source",
            "source_url": "https://example.com/fallback",
            "title": "Fallback Item",
            "published_at": None,
        }
    )
    other_item = make_item().model_copy(
        update={
            "source_url": "https://example.com/other",
            "published_at": datetime(2026, 5, 20, 8, 0, tzinfo=timezone.utc),
        }
    )

    with session_factory() as session:
        published_raw = save_raw_item(session, published_item)
        fallback_raw = save_raw_item(session, fallback_item)
        other_raw = save_raw_item(session, other_item)
        fallback_raw.ingested_at = datetime(2026, 5, 21, 9, 0, tzinfo=timezone.utc)
        save_ai_analysis(session, published_raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        save_ai_analysis(session, fallback_raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        save_ai_analysis(session, other_raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        session.commit()

        report_data = build_daily_report_data(session, date(2026, 5, 21))

    assert report_data["stance_counts"]["positive"] == 2
    assert [item["title"] for item in report_data["top_items"]] == ["Published Item", "Fallback Item"]
    assert report_data["top_items"][0]["url"] == "https://example.com/published"
