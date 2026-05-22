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
