from pathlib import Path

from tnmi.ai import MockAIAnalyzer
from tnmi.contracts import NewspaperSource
from tnmi.pipeline import DailyNewsPipeline, InMemoryNewsClient
from tnmi.storage import create_session_factory, init_db


def test_daily_news_pipeline_processes_feed_and_article(tmp_path):
    feed_xml = Path("tests/fixtures/sample_feed.xml").read_text(encoding="utf-8")
    article_html = Path("tests/fixtures/sample_article.html").read_text(encoding="utf-8")
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    client = InMemoryNewsClient(
        feeds={"https://example.com/rss": feed_xml},
        articles={"https://example.com/news/tamil-nadu-scheme": article_html},
    )
    pipeline = DailyNewsPipeline(
        session_factory=session_factory,
        news_client=client,
        analyzer=MockAIAnalyzer(),
    )

    result = pipeline.run([NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])])

    assert result.items_seen == 1
    assert result.items_saved == 1
    assert result.analyses_saved == 1
