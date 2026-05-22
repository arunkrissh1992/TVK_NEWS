from pathlib import Path

from sqlalchemy import func, select

from tnmi.ai import MockAIAnalyzer
from tnmi.contracts import NewspaperSource
from tnmi.pipeline import (
    DailyNewsPipeline,
    InMemoryNewsClient,
    RequestsNewsClient,
    is_safe_resolved_article_url,
    normalize_source_url,
)
from tnmi.storage import AIAnalysisRecord, RawItemRecord, create_session_factory, init_db


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


def test_daily_news_pipeline_rerun_does_not_duplicate_rows(tmp_path):
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
    sources = [NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])]

    first = pipeline.run(sources)
    second = pipeline.run(sources)

    with session_factory() as session:
        raw_count = session.scalar(select(func.count()).select_from(RawItemRecord))
        analysis_count = session.scalar(select(func.count()).select_from(AIAnalysisRecord))

    assert first.items_seen == 1
    assert first.items_saved == 1
    assert first.analyses_saved == 1
    assert second.items_seen == 1
    assert second.items_saved == 1
    assert second.analyses_saved == 0
    assert raw_count == 1
    assert analysis_count == 1


def test_daily_news_pipeline_rerun_does_not_call_analyzer_for_existing_analysis(tmp_path):
    class CountingAnalyzer(MockAIAnalyzer):
        def __init__(self) -> None:
            self.calls = 0

        def analyze(self, item):
            self.calls += 1
            return super().analyze(item)

    feed_xml = Path("tests/fixtures/sample_feed.xml").read_text(encoding="utf-8")
    article_html = Path("tests/fixtures/sample_article.html").read_text(encoding="utf-8")
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    client = InMemoryNewsClient(
        feeds={"https://example.com/rss": feed_xml},
        articles={"https://example.com/news/tamil-nadu-scheme": article_html},
    )
    analyzer = CountingAnalyzer()
    pipeline = DailyNewsPipeline(
        session_factory=session_factory,
        news_client=client,
        analyzer=analyzer,
    )
    sources = [NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])]

    pipeline.run(sources)
    pipeline.run(sources)

    assert analyzer.calls == 1


def test_daily_news_pipeline_rolls_back_raw_item_when_analysis_fails(tmp_path):
    class RaisingAnalyzer:
        model_name = "raising"

        def analyze(self, item):
            raise RuntimeError("analysis failed")

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
        analyzer=RaisingAnalyzer(),
    )

    result = pipeline.run([NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])])

    with session_factory() as session:
        raw_count = session.scalar(select(func.count()).select_from(RawItemRecord))
        analysis_count = session.scalar(select(func.count()).select_from(AIAnalysisRecord))

    assert result.items_seen == 1
    assert result.items_saved == 0
    assert result.analyses_saved == 0
    assert result.failures == 1
    assert raw_count == 0
    assert analysis_count == 0


def test_daily_news_pipeline_stores_normalized_source_url(tmp_path):
    article_url = "https://Example.com/news/tamil-nadu-scheme?utm_source=rss&district=chennai&fbclid=abc#comments"
    feed_xml = f"""
    <rss version="2.0">
      <channel>
        <item>
          <title>Tamil Nadu scheme</title>
          <link>{article_url}</link>
        </item>
      </channel>
    </rss>
    """
    article_html = Path("tests/fixtures/sample_article.html").read_text(encoding="utf-8")
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    client = InMemoryNewsClient(
        feeds={"https://example.com/rss": feed_xml},
        articles={article_url: article_html},
    )
    pipeline = DailyNewsPipeline(
        session_factory=session_factory,
        news_client=client,
        analyzer=MockAIAnalyzer(),
    )

    pipeline.run([NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])])

    with session_factory() as session:
        raw = session.scalar(select(RawItemRecord))

    assert raw.source_url == "https://example.com/news/tamil-nadu-scheme?district=chennai"
    assert normalize_source_url(article_url) == raw.source_url


def test_daily_news_pipeline_blocks_localhost_article_url(tmp_path):
    feed_xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Localhost attack</title>
          <link>http://localhost/admin</link>
        </item>
      </channel>
    </rss>
    """
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    client = InMemoryNewsClient(feeds={"https://example.com/rss": feed_xml}, articles={})
    pipeline = DailyNewsPipeline(
        session_factory=session_factory,
        news_client=client,
        analyzer=MockAIAnalyzer(),
    )

    result = pipeline.run([NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])])

    assert result.items_seen == 1
    assert result.items_saved == 0
    assert result.analyses_saved == 0
    assert result.failures == 1


def test_daily_news_pipeline_skips_article_host_outside_source_domains(tmp_path):
    feed_xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Outside host</title>
          <link>https://evil.example.net/news</link>
        </item>
      </channel>
    </rss>
    """
    article_html = Path("tests/fixtures/sample_article.html").read_text(encoding="utf-8")
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    client = InMemoryNewsClient(
        feeds={"https://example.com/rss": feed_xml},
        articles={"https://evil.example.net/news": article_html},
    )
    pipeline = DailyNewsPipeline(
        session_factory=session_factory,
        news_client=client,
        analyzer=MockAIAnalyzer(),
    )

    result = pipeline.run([NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])])

    assert result.items_seen == 1
    assert result.items_saved == 0
    assert result.analyses_saved == 0
    assert result.failures == 1


def test_daily_news_pipeline_allows_article_on_source_domain(tmp_path):
    feed_xml = """
    <rss version="2.0">
      <channel>
        <item>
          <title>Allowed source host</title>
          <link>https://news.example.com/news/tamil-nadu-scheme</link>
        </item>
      </channel>
    </rss>
    """
    article_html = Path("tests/fixtures/sample_article.html").read_text(encoding="utf-8")
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    client = InMemoryNewsClient(
        feeds={"https://example.com/rss": feed_xml},
        articles={"https://news.example.com/news/tamil-nadu-scheme": article_html},
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


def test_daily_news_pipeline_reports_skipped_inactive_and_unconfigured_sources(tmp_path):
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
    sources = [
        NewspaperSource(name="Active Source", rss_urls=["https://example.com/rss"]),
        NewspaperSource(name="Inactive Source", active=False),
        NewspaperSource(name="Unconfigured Source", active=True, rss_urls=[]),
    ]

    result = pipeline.run(sources)

    assert result.items_seen == 1
    assert result.items_saved == 1
    assert result.analyses_saved == 1
    assert result.sources_skipped == 2


def test_requests_news_client_blocks_redirect_to_disallowed_article_host(monkeypatch):
    calls: list[str] = []

    class FakeResponse:
        is_redirect = True
        headers = {"Location": "http://127.0.0.1/admin"}

    class FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url, **kwargs):
            calls.append(url)
            return FakeResponse()

    monkeypatch.setattr("tnmi.pipeline.requests.Session", FakeSession)

    client = RequestsNewsClient()

    try:
        client.fetch_text("https://example.com/news", allowed_hosts={"example.com"})
    except ValueError as exc:
        assert "blocked article URL" in str(exc)
    else:
        raise AssertionError("redirect to localhost should be blocked")

    assert calls == ["https://example.com/news"]


def test_requests_news_client_blocks_hosts_resolving_to_private_ips(monkeypatch):
    monkeypatch.setattr(
        "tnmi.pipeline.socket.getaddrinfo",
        lambda host, port: [(None, None, None, None, ("10.0.0.5", 0))],
    )

    assert is_safe_resolved_article_url("https://example.com/news") is False
