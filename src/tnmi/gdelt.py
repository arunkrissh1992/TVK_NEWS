"""GDELT 2.0 Article API — Phase E cross-reference layer.

GDELT (https://www.gdeltproject.org/) is a free, open dataset run by Google
Jigsaw + Yale. Every news article it can find globally is tagged with people,
locations, themes, and a tone score (sentiment). For our purposes, we use it
to answer one question per Tamil-Nadu story:

    "Is anyone else in the world reporting on this — and if so, what's
     the global tone?"

Useful signals:

  * **Global volume**: if 50 outlets worldwide cover the story, it has
    global weight; if 2 cover it, it's a local affair.
  * **Tone**: GDELT computes a sentiment-style score (-10 negative to
    +10 positive). Compare with our own stance assignment as a
    cross-validation signal.
  * **Themes**: GDELT tags articles with controlled-vocab themes
    (TAX_FNCACT_MINISTER, AFFECT_POSITIVE, KILL, etc.) which lets us
    spot semantic links even when source text languages differ.

API endpoint (no auth, public):

    https://api.gdeltproject.org/api/v2/doc/doc

Documentation:
    https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/

The adapter is intentionally thin — we hit the public REST endpoint, parse
the JSON response, and surface a few aggregate fields the dashboard can
render. We DO NOT batch or cache aggressively here; for production a caller
should add a per-day cache layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests


logger = logging.getLogger(__name__)


GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


@dataclass(frozen=True)
class GdeltMatch:
    title: str
    url: str
    domain: str
    language: str
    seendate: str  # YYYYMMDDHHMMSS
    socialimage: str | None = None


@dataclass(frozen=True)
class GdeltCrossReference:
    query: str
    timespan: str
    article_count: int
    distinct_domains: int
    matches: list[GdeltMatch]
    avg_tone: float | None  # -10..+10 ; None when no tone data
    has_signal: bool        # True if global coverage > a threshold


def search_articles(
    query: str,
    *,
    timespan: str = "24h",
    max_records: int = 25,
    timeout: int = 8,
) -> GdeltCrossReference:
    """Look up recent global coverage matching ``query``.

    ``query`` syntax follows GDELT DOC 2.0 — e.g. ``"Cauvery water" tamilnadu``.
    ``timespan`` accepts ``"24h"``, ``"7d"``, ``"30d"``, etc.

    Returns a GdeltCrossReference (never raises on a non-200 response —
    failures degrade to an empty signal so the dashboard never crashes
    over a GDELT outage).
    """
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "timespan": timespan,
        "maxrecords": str(max(1, min(max_records, 250))),
        "sort": "DateDesc",
    }
    try:
        response = requests.get(GDELT_DOC_URL, params=params, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("GDELT request failed (%s): %s", query, exc)
        return _empty_cross_reference(query, timespan)

    try:
        data = response.json()
    except ValueError:
        logger.warning("GDELT returned non-JSON for query %s", query)
        return _empty_cross_reference(query, timespan)

    articles: list[GdeltMatch] = []
    for row in data.get("articles", []) or []:
        articles.append(
            GdeltMatch(
                title=(row.get("title") or "").strip(),
                url=row.get("url") or "",
                domain=row.get("domain") or "",
                language=row.get("language") or "unknown",
                seendate=row.get("seendate") or "",
                socialimage=row.get("socialimage"),
            )
        )

    return _build_cross_reference(query, timespan, articles)


def _build_cross_reference(
    query: str,
    timespan: str,
    articles: list[GdeltMatch],
) -> GdeltCrossReference:
    distinct_domains = len({a.domain for a in articles if a.domain})
    # The public ArtList endpoint does not return per-article tone; we expose
    # avg_tone as None unless callers swap to ArtListWithToneSummary later.
    return GdeltCrossReference(
        query=query,
        timespan=timespan,
        article_count=len(articles),
        distinct_domains=distinct_domains,
        matches=articles,
        avg_tone=None,
        has_signal=len(articles) >= 3,  # ≥3 global outlets means real signal
    )


def _empty_cross_reference(query: str, timespan: str) -> GdeltCrossReference:
    return GdeltCrossReference(
        query=query,
        timespan=timespan,
        article_count=0,
        distinct_domains=0,
        matches=[],
        avg_tone=None,
        has_signal=False,
    )


def build_query_for_theme(theme_title: str, *, language_hint: str = "en") -> str:
    """Convert a TN-newspaper headline into a GDELT-friendly query.

    GDELT supports basic AND/OR; we keep it simple — strip punctuation,
    keep the first 5–8 informative words, optionally append a
    Tamil-Nadu region filter.
    """
    if not theme_title:
        return "tamil nadu"
    # Strip newspaper-shell markers that pollute the query.
    cleaned = (
        theme_title
        .replace("|", " ")
        .replace(":", " ")
        .replace("·", " ")
    )
    tokens = [t for t in cleaned.split() if len(t) > 2]
    # GDELT works best with English; if the source is Tamil we still pass
    # the original tokens because GDELT can match Tamil-script articles too.
    headline_tokens = tokens[:8]
    base = " ".join(headline_tokens) if headline_tokens else "tamil nadu"
    # Bias to TN-relevant coverage so global noise (e.g. "Cauvery" matching
    # the global river-system corpus) is filtered out a bit.
    if language_hint == "en":
        return f"{base} tamil nadu"
    return base
