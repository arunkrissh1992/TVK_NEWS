from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from email.utils import parsedate_to_datetime

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


def parse_feed_entries(source: NewspaperSource, feed_xml: str) -> list[FeedEntry]:
    parsed = feedparser.parse(feed_xml)
    entries: list[FeedEntry] = []
    for entry in parsed.entries:
        published_at = None
        if getattr(entry, "published", None):
            published_at = parsedate_to_datetime(entry.published)
        entries.append(
            FeedEntry(
                source_name=source.name,
                url=entry.link,
                title=getattr(entry, "title", ""),
                published_at=published_at,
            )
        )
    return entries


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
    clean_text = extracted or ""
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
    )
