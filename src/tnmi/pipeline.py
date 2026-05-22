from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import requests
from sqlalchemy.orm import Session, sessionmaker

from tnmi.ai import AIAnalyzer, PROMPT_VERSION
from tnmi.contracts import NewspaperSource, NormalizedItem, SourceType
from tnmi.language import detect_language
from tnmi.news import FeedEntry, extract_article_text, parse_feed_entries, parse_listing_entries
from tnmi.storage import get_ai_analysis, save_ai_analysis, save_raw_item


_TRACKING_QUERY_PARAMS = {"fbclid", "gclid"}
_LOCAL_HOSTNAMES = {"localhost"}


def normalize_source_url(url: str) -> str:
    parsed = urlsplit(url)
    query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in _TRACKING_QUERY_PARAMS
    ]
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            urlencode(query, doseq=True),
            "",
        )
    )


class NewsClient(Protocol):
    def fetch_text(self, url: str, *, allowed_hosts: set[str] | None = None) -> str:
        ...


class RequestsNewsClient:
    def fetch_text(self, url: str, *, allowed_hosts: set[str] | None = None) -> str:
        current_url = url
        with requests.Session() as session:
            for _ in range(10):
                if allowed_hosts is not None and not is_allowed_article_url(current_url, allowed_hosts):
                    raise ValueError("blocked article URL")
                if not is_safe_resolved_article_url(current_url):
                    raise ValueError("blocked article URL")
                response = session.get(
                    current_url,
                    timeout=30,
                    headers={"User-Agent": "tn-media-intelligence/0.1"},
                    allow_redirects=False,
                )
                if not response.is_redirect:
                    response.raise_for_status()
                    return response.text
                location = response.headers.get("Location")
                if not location:
                    raise ValueError("redirect response missing Location header")
                current_url = urljoin(current_url, location)
            raise ValueError("too many redirects")


class InMemoryNewsClient:
    def __init__(self, *, feeds: dict[str, str], articles: dict[str, str]) -> None:
        self.feeds = feeds
        self.articles = articles

    def fetch_text(self, url: str, *, allowed_hosts: set[str] | None = None) -> str:
        if url in self.feeds:
            return self.feeds[url]
        return self.articles[url]


def _normalized_hostname(url: str) -> str | None:
    try:
        hostname = urlsplit(url).hostname
    except ValueError:
        return None
    if not hostname:
        return None
    return hostname.rstrip(".").lower()


def _is_blocked_ip_literal(hostname: str) -> bool:
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return any(
        (
            address.is_loopback,
            address.is_private,
            address.is_link_local,
            address.is_multicast,
            address.is_unspecified,
            address.is_reserved,
        )
    )


def _hostname_resolves_to_blocked_ip(hostname: str) -> bool:
    try:
        results = socket.getaddrinfo(hostname, None)
    except OSError:
        return False

    for result in results:
        address = result[4][0]
        try:
            parsed = ipaddress.ip_address(address)
        except ValueError:
            continue
        if any(
            (
                parsed.is_loopback,
                parsed.is_private,
                parsed.is_link_local,
                parsed.is_multicast,
                parsed.is_unspecified,
                parsed.is_reserved,
            )
        ):
            return True
    return False


def _is_local_hostname(hostname: str) -> bool:
    return hostname in _LOCAL_HOSTNAMES or hostname.endswith(".localhost")


def _host_matches_allowed(hostname: str, allowed_hosts: set[str]) -> bool:
    return any(hostname == allowed or hostname.endswith(f".{allowed}") for allowed in allowed_hosts)


def allowed_article_hosts(source: NewspaperSource) -> set[str]:
    hosts: set[str] = set()
    for url in [*source.rss_urls, *source.sitemap_urls, *source.section_urls]:
        hostname = _normalized_hostname(str(url))
        if hostname and not _is_local_hostname(hostname) and not _is_blocked_ip_literal(hostname):
            hosts.add(hostname)
    return hosts


def is_allowed_article_url(url: str, allowed_hosts: set[str]) -> bool:
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"}:
        return False
    hostname = _normalized_hostname(url)
    if not hostname:
        return False
    if _is_local_hostname(hostname) or _is_blocked_ip_literal(hostname):
        return False
    return _host_matches_allowed(hostname, allowed_hosts)


def is_safe_resolved_article_url(url: str) -> bool:
    hostname = _normalized_hostname(url)
    if not hostname or _is_blocked_ip_literal(hostname):
        return False
    return not _hostname_resolves_to_blocked_ip(hostname)


@dataclass(frozen=True)
class PipelineResult:
    items_seen: int = 0
    items_saved: int = 0
    analyses_saved: int = 0
    failures: int = 0
    sources_skipped: int = 0


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
        sources_skipped = 0

        with self.session_factory() as session:
            for source in sources:
                if not source.active:
                    sources_skipped += 1
                    continue
                if not source.rss_urls:
                    sources_skipped += 1
                    continue
                source_allowed_hosts = allowed_article_hosts(source)
                for rss_url in source.rss_urls:
                    try:
                        feed_xml = self.news_client.fetch_text(str(rss_url))
                        entries = _parse_source_entries(source, feed_xml, source_url=str(rss_url))
                    except Exception:
                        failures += 1
                        continue

                    for entry in entries:
                        items_seen += 1
                        try:
                            if not is_allowed_article_url(entry.url, source_allowed_hosts):
                                raise ValueError("blocked article URL")
                            html = self.news_client.fetch_text(entry.url, allowed_hosts=source_allowed_hosts)
                            normalized_url = normalize_source_url(entry.url)
                            article = extract_article_text(normalized_url, html)
                            with session.begin_nested():
                                text = article.clean_text.strip()
                                if not text:
                                    raise ValueError("article extraction produced no text")
                                item = NormalizedItem(
                                    source_type=SourceType.NEWS,
                                    source_name=source.name,
                                    source_url=normalized_url,
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
                                existing_analysis = get_ai_analysis(
                                    session,
                                    raw.id,
                                    model_name=self.analyzer.model_name,
                                    prompt_version=PROMPT_VERSION,
                                )
                                if existing_analysis:
                                    items_saved += 1
                                    continue
                                analysis = self.analyzer.analyze(item)
                                save_ai_analysis(
                                    session,
                                    raw.id,
                                    analysis,
                                    model_name=self.analyzer.model_name,
                                    prompt_version=PROMPT_VERSION,
                                )
                                items_saved += 1
                                analyses_saved += 1
                        except Exception:
                            failures += 1

            session.commit()

        return PipelineResult(
            items_seen=items_seen,
            items_saved=items_saved,
            analyses_saved=analyses_saved,
            failures=failures,
            sources_skipped=sources_skipped,
        )


def _parse_source_entries(source: NewspaperSource, content: str, *, source_url: str) -> list[FeedEntry]:
    entries = parse_feed_entries(source, content)
    if entries:
        return entries
    return parse_listing_entries(source, content, base_url=source_url)
