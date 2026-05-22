from pathlib import Path

from tnmi.contracts import NewspaperSource
from tnmi.news import extract_article_text, parse_feed_entries


def test_parse_feed_entries_reads_rss_fixture():
    feed_xml = Path("tests/fixtures/sample_feed.xml").read_text(encoding="utf-8")
    source = NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])

    entries = parse_feed_entries(source, feed_xml)

    assert len(entries) == 1
    assert entries[0].url == "https://example.com/news/tamil-nadu-scheme"
    assert "தமிழக அரசு" in entries[0].title


def test_extract_article_text_from_html_fixture():
    html = Path("tests/fixtures/sample_article.html").read_text(encoding="utf-8")

    article = extract_article_text("https://example.com/news/tamil-nadu-scheme", html)

    assert article.title == "தமிழக அரசு புதிய திட்டம்"
    assert "நலத்திட்டத்தை" in article.clean_text


def test_parse_feed_entries_skips_missing_link_entry_without_raising():
    feed_xml = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <item>
      <title>No URL</title>
      <pubDate>Thu, 21 May 2026 06:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
"""
    source = NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])

    entries = parse_feed_entries(source, feed_xml)

    assert entries == []


def test_parse_feed_entries_invalid_date_is_none_without_raising():
    feed_xml = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <item>
      <title>Invalid Date</title>
      <link>https://example.com/news/invalid-date</link>
      <pubDate>not a real date</pubDate>
    </item>
  </channel>
</rss>
"""
    source = NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])

    entries = parse_feed_entries(source, feed_xml)

    assert len(entries) == 1
    assert entries[0].published_at is None


def test_parse_feed_entries_uses_updated_date_when_published_absent():
    feed_xml = """<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <item>
      <title>Updated Date</title>
      <link>https://example.com/news/updated-date</link>
      <updated>2026-05-21T08:30:00Z</updated>
    </item>
  </channel>
</rss>
"""
    source = NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])

    entries = parse_feed_entries(source, feed_xml)

    assert len(entries) == 1
    assert entries[0].published_at is not None
    assert entries[0].published_at.year == 2026
    assert entries[0].published_at.hour == 8


def test_extract_article_text_fallback_marks_primary_extraction_failure():
    html = """<!doctype html>
<html>
  <head>
    <title>Fallback Article</title>
    <meta name="description" content="Fallback summary from metadata.">
  </head>
  <body></body>
</html>
"""

    article = extract_article_text("https://example.com/news/fallback", html)

    assert article.extraction_succeeded is False
    assert article.clean_text == "Fallback summary from metadata."
