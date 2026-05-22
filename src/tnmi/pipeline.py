from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests
from sqlalchemy.orm import Session, sessionmaker

from tnmi.ai import AIAnalyzer, PROMPT_VERSION
from tnmi.contracts import NewspaperSource, NormalizedItem, SourceType
from tnmi.language import detect_language
from tnmi.news import extract_article_text, parse_feed_entries
from tnmi.storage import save_ai_analysis, save_raw_item


class NewsClient(Protocol):
    def fetch_text(self, url: str) -> str:
        ...


class RequestsNewsClient:
    def fetch_text(self, url: str) -> str:
        response = requests.get(
            url,
            timeout=30,
            headers={"User-Agent": "tn-media-intelligence/0.1"},
        )
        response.raise_for_status()
        return response.text


class InMemoryNewsClient:
    def __init__(self, *, feeds: dict[str, str], articles: dict[str, str]) -> None:
        self.feeds = feeds
        self.articles = articles

    def fetch_text(self, url: str) -> str:
        if url in self.feeds:
            return self.feeds[url]
        return self.articles[url]


@dataclass(frozen=True)
class PipelineResult:
    items_seen: int = 0
    items_saved: int = 0
    analyses_saved: int = 0
    failures: int = 0


class DailyNewsPipeline:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        news_client: NewsClient,
        analyzer: AIAnalyzer,
    ) -> None:
        self.session_factory = session_factory
        self.news_client = news_client
        self.analyzer = analyzer

    def run(self, sources: list[NewspaperSource]) -> PipelineResult:
        items_seen = 0
        items_saved = 0
        analyses_saved = 0
        failures = 0

        with self.session_factory() as session:
            for source in sources:
                if not source.active:
                    continue
                for rss_url in source.rss_urls:
                    try:
                        feed_xml = self.news_client.fetch_text(str(rss_url))
                        entries = parse_feed_entries(source, feed_xml)
                    except Exception:
                        failures += 1
                        continue

                    for entry in entries:
                        items_seen += 1
                        try:
                            html = self.news_client.fetch_text(entry.url)
                            article = extract_article_text(entry.url, html)
                            text = article.clean_text.strip()
                            if not text:
                                failures += 1
                                continue
                            item = NormalizedItem(
                                source_type=SourceType.NEWS,
                                source_name=source.name,
                                source_url=entry.url,
                                published_at=entry.published_at,
                                language=detect_language(text),
                                title=article.title or entry.title,
                                raw_text_original=article.raw_text,
                                clean_text_original=text,
                                metadata={
                                    **article.metadata,
                                    "extraction_succeeded": article.extraction_succeeded,
                                },
                            )
                            raw = save_raw_item(session, item)
                            items_saved += 1
                            analysis = self.analyzer.analyze(item)
                            save_ai_analysis(
                                session,
                                raw.id,
                                analysis,
                                model_name=self.analyzer.model_name,
                                prompt_version=PROMPT_VERSION,
                            )
                            analyses_saved += 1
                        except Exception:
                            failures += 1

            session.commit()

        return PipelineResult(
            items_seen=items_seen,
            items_saved=items_saved,
            analyses_saved=analyses_saved,
            failures=failures,
        )
