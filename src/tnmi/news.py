from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from time import struct_time
from urllib.parse import urljoin, urlsplit

import feedparser
import trafilatura

from tnmi.contracts import NewspaperSource


@dataclass(frozen=True)
class FeedEntry:
    source_name: str
    url: str
    title: str
    published_at: datetime | None


@dataclass(frozen=True)
class ExtractedArticle:
    url: str
    title: str | None
    clean_text: str
    raw_text: str
    metadata: dict[str, str]
    extraction_succeeded: bool


class _TextFallbackParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = data.strip()
        if text:
            self._chunks.append(text)

    def get_text(self) -> str:
        return " ".join(self._chunks)


class _ListingLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._active_href: str | None = None
        self._text_chunks: list[str] = []
        self.links: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href:
            self._active_href = href
            self._text_chunks = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._active_href:
            return
        title = " ".join(chunk.strip() for chunk in self._text_chunks if chunk.strip()).strip()
        self.links.append((self._active_href, title))
        self._active_href = None
        self._text_chunks = []

    def handle_data(self, data: str) -> None:
        if self._active_href:
            self._text_chunks.append(data)


def _is_url_like(value: str | None) -> bool:
    return bool(value and value.startswith(("http://", "https://")))


def _entry_value(entry: object, key: str) -> object:
    if isinstance(entry, dict):
        return dict.get(entry, key)
    if hasattr(entry, "get"):
        value = entry.get(key)
        if value:
            return value
    return getattr(entry, key, None)


def _entry_url(entry: object) -> str | None:
    link = _entry_value(entry, "link")
    if isinstance(link, str) and _is_url_like(link):
        return link
    for key in ("id", "guid"):
        fallback = _entry_value(entry, key)
        if isinstance(fallback, str) and _is_url_like(fallback):
            return fallback
    return None


def _datetime_from_struct_time(value: object) -> datetime | None:
    if not isinstance(value, struct_time):
        return None
    try:
        return datetime.fromtimestamp(calendar.timegm(value), tz=UTC)
    except (ValueError, TypeError, OverflowError):
        return None


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return parsedate_to_datetime(value)
    except (ValueError, TypeError, OverflowError):
        return None


def _entry_published_at(entry: object) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = _datetime_from_struct_time(_entry_value(entry, key))
        if parsed:
            return parsed
    for key in ("published", "updated"):
        parsed = _parse_datetime(_entry_value(entry, key))
        if parsed:
            return parsed
    return None


def _fallback_html_text(html: str) -> str:
    parser = _TextFallbackParser()
    parser.feed(html)
    return parser.get_text()


def parse_feed_entries(source: NewspaperSource, feed_xml: str) -> list[FeedEntry]:
    parsed = feedparser.parse(feed_xml)
    entries: list[FeedEntry] = []
    for entry in parsed.entries:
        url = _entry_url(entry)
        if not url:
            continue
        entries.append(
            FeedEntry(
                source_name=source.name,
                url=url,
                title=getattr(entry, "title", ""),
                published_at=_entry_published_at(entry),
            )
        )
    return entries


def parse_listing_entries(
    source: NewspaperSource,
    html: str,
    *,
    base_url: str,
    limit: int = 20,
) -> list[FeedEntry]:
    parser = _ListingLinkParser()
    parser.feed(html)
    base_hostname = urlsplit(base_url).hostname
    seen_urls: set[str] = set()
    entries: list[FeedEntry] = []
    for href, title in parser.links:
        url = urljoin(base_url, href)
        if not _is_listing_article_candidate(url, title, base_hostname):
            continue
        normalized_title = " ".join(title.split())
        if url in seen_urls:
            continue
        seen_urls.add(url)
        entries.append(
            FeedEntry(
                source_name=source.name,
                url=url,
                title=normalized_title,
                published_at=None,
            )
        )
        if len(entries) >= limit:
            break
    return entries


def _is_listing_article_candidate(url: str, title: str, base_hostname: str | None) -> bool:
    if not _is_url_like(url) or len(title.strip()) < 10:
        return False
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if base_hostname and parsed.hostname != base_hostname:
        return False
    path = parsed.path.strip("/")
    if not path:
        return False
    blocked_paths = {"rss", "epaper", "privacy-policy", "terms"}
    return path not in blocked_paths


def extract_article_text(url: str, html: str) -> ExtractedArticle:
    extracted = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        output_format="txt",
        url=url,
    )
    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = metadata.title if metadata else None
    extraction_succeeded = bool(extracted)
    clean_text = extracted or (metadata.description if metadata and metadata.description else "")
    if not clean_text:
        clean_text = _fallback_html_text(html)
    return ExtractedArticle(
        url=url,
        title=title,
        clean_text=clean_text,
        raw_text=clean_text,
        metadata={
            "author": metadata.author if metadata and metadata.author else "",
            "date": metadata.date if metadata and metadata.date else "",
            "sitename": metadata.sitename if metadata and metadata.sitename else "",
        },
        extraction_succeeded=extraction_succeeded,
    )
