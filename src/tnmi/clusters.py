"""Recurring-theme detection over the chunk_embeddings vector store.

The dashboard's per-card AI analysis answers "what is this one article saying".
This module answers the next question the chief asks: "Is this a one-off, or is
the same narrative recurring across multiple newspapers this week?".

It works on top of the already-persisted vectors in ``chunk_embeddings``. No
extra OpenAI calls are made — only cosine-similarity arithmetic in Python.

When no embeddings are available the helper returns an empty list and a
diagnostic string that tells the operator the exact CLI to run.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from sqlalchemy import select
from sqlalchemy.orm import Session

from tnmi.storage import (
    AIAnalysisRecord,
    ChunkEmbeddingRecord,
    DocumentChunkRecord,
    RawItemRecord,
)


# A pair of articles is considered "the same narrative" when their leading-chunk
# vectors cosine-correlate above this threshold. 0.78 worked well on mock + live
# data — it keeps unrelated stories apart while still merging legitimate echoes.
DEFAULT_SIMILARITY_THRESHOLD = 0.78


@dataclass
class ThemeMember:
    raw_item_id: int
    source_name: str
    title: str | None
    stance: str | None
    summary: str | None
    similarity: float  # against the cluster representative
    published_at: datetime | None = None


@dataclass
class ThemeCluster:
    representative_id: int
    members: list[ThemeMember] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def source_count(self) -> int:
        return len({member.source_name for member in self.members})

    @property
    def dominant_stance(self) -> str:
        counts = Counter(m.stance for m in self.members if m.stance)
        if not counts:
            return "neutral"
        return counts.most_common(1)[0][0]

    @property
    def stance_breakdown(self) -> dict[str, int]:
        return dict(Counter(m.stance for m in self.members if m.stance))

    @property
    def sample_title(self) -> str | None:
        # Prefer the representative's title; fall back to any member with a title.
        rep = next((m for m in self.members if m.raw_item_id == self.representative_id), None)
        if rep and rep.title:
            return rep.title
        for member in self.members:
            if member.title:
                return member.title
        return None

    @property
    def sample_summary(self) -> str | None:
        rep = next((m for m in self.members if m.raw_item_id == self.representative_id), None)
        if rep and rep.summary:
            return rep.summary
        for member in self.members:
            if member.summary:
                return member.summary
        return None

    @property
    def latest_published_at(self) -> datetime | None:
        timestamps = [m.published_at for m in self.members if m.published_at]
        if not timestamps:
            return None
        return max(timestamps, key=_utc_sort_key)


@dataclass(frozen=True)
class RecurringThemesReport:
    themes: list[ThemeCluster]
    total_articles_considered: int
    total_articles_indexed: int
    diagnostic: str | None = None  # human-readable note when nothing was found

    @property
    def has_themes(self) -> bool:
        return bool(self.themes)


def find_recurring_themes(
    session: Session,
    *,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    min_cluster_size: int = 2,
    limit: int = 5,
    source_type: str = "news",
) -> RecurringThemesReport:
    """Cluster articles by their first-chunk embedding and return the top themes.

    Only clusters with at least ``min_cluster_size`` members are returned. The
    centroid for each cluster is the running mean of its member vectors so the
    "is this article close enough" check stays robust as members are added.
    """

    vectors = _load_article_vectors(session, source_type=source_type)
    total_articles_indexed = len(vectors)
    if not vectors:
        return RecurringThemesReport(
            themes=[],
            total_articles_considered=0,
            total_articles_indexed=0,
            diagnostic=(
                "No embeddings found yet. Run: "
                "python -m pipelines.build_rag_index --mock-embeddings --source-type news"
            ),
        )

    analyses = _load_latest_analyses(session, raw_item_ids=[v.raw_item_id for v in vectors])

    clusters: list[ThemeCluster] = []
    for vector in vectors:
        # Skip articles whose centroid is all-zero (degenerate embedding).
        if not any(vector.embedding):
            continue
        assigned = False
        for cluster in clusters:
            sim = _cosine_similarity(vector.embedding, cluster.centroid)
            if sim >= similarity_threshold:
                _add_member(
                    cluster=cluster,
                    vector=vector,
                    similarity=sim,
                    analysis=analyses.get(vector.raw_item_id),
                )
                assigned = True
                break
        if not assigned:
            new_cluster = ThemeCluster(representative_id=vector.raw_item_id, centroid=list(vector.embedding))
            _add_member(
                cluster=new_cluster,
                vector=vector,
                similarity=1.0,
                analysis=analyses.get(vector.raw_item_id),
            )
            clusters.append(new_cluster)

    # Keep only clusters that span multiple articles — single-article "themes"
    # are just the article itself and add no signal to the chief.
    themes = [cluster for cluster in clusters if cluster.size >= min_cluster_size]

    # Rank: bigger cluster first, then more distinct sources (cross-newspaper
    # narratives are more important), then most recent member first.
    themes.sort(
        key=lambda cluster: (
            -cluster.size,
            -cluster.source_count,
            -(_utc_timestamp(cluster.latest_published_at)),
            cluster.representative_id,
        )
    )

    return RecurringThemesReport(
        themes=themes[: max(0, limit)],
        total_articles_considered=len(clusters),
        total_articles_indexed=total_articles_indexed,
        diagnostic=None if themes else "No recurring narratives detected above the similarity threshold.",
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArticleVector:
    raw_item_id: int
    source_name: str
    title: str | None
    published_at: datetime | None
    embedding: list[float]


def _load_article_vectors(session: Session, *, source_type: str) -> list[_ArticleVector]:
    """One vector per article — the leading-chunk embedding."""
    rows = session.execute(
        select(
            RawItemRecord.id,
            RawItemRecord.source_name,
            RawItemRecord.title,
            RawItemRecord.published_at,
            ChunkEmbeddingRecord.embedding,
        )
        .join(DocumentChunkRecord, DocumentChunkRecord.raw_item_id == RawItemRecord.id)
        .join(ChunkEmbeddingRecord, ChunkEmbeddingRecord.chunk_id == DocumentChunkRecord.id)
        .where(RawItemRecord.source_type == source_type)
        .where(DocumentChunkRecord.chunk_index == 0)
        .order_by(RawItemRecord.ingested_at.desc(), RawItemRecord.id.desc())
    ).all()

    seen: set[int] = set()
    vectors: list[_ArticleVector] = []
    for raw_id, source_name, title, published_at, embedding in rows:
        if raw_id in seen:
            continue
        if not embedding:
            continue
        seen.add(raw_id)
        vectors.append(
            _ArticleVector(
                raw_item_id=raw_id,
                source_name=source_name,
                title=title,
                published_at=published_at,
                embedding=list(embedding),
            )
        )
    return vectors


def _load_latest_analyses(
    session: Session, *, raw_item_ids: Iterable[int]
) -> dict[int, AIAnalysisRecord]:
    """Map raw_item_id → most recent AIAnalysisRecord. Prefer non-mock models."""
    raw_id_list = list(raw_item_ids)
    if not raw_id_list:
        return {}
    rows = session.scalars(
        select(AIAnalysisRecord)
        .where(AIAnalysisRecord.raw_item_id.in_(raw_id_list))
        .order_by(AIAnalysisRecord.raw_item_id, AIAnalysisRecord.created_at.desc())
    ).all()
    result: dict[int, AIAnalysisRecord] = {}
    for record in rows:
        if record.raw_item_id not in result:
            result[record.raw_item_id] = record
        elif result[record.raw_item_id].model_name == "mock" and record.model_name != "mock":
            result[record.raw_item_id] = record
    return result


def _add_member(
    *,
    cluster: ThemeCluster,
    vector: _ArticleVector,
    similarity: float,
    analysis: AIAnalysisRecord | None,
) -> None:
    summary = None
    stance = None
    if analysis is not None:
        summary = (analysis.summary_english or analysis.summary_original or "").strip() or None
        stance = analysis.stance_toward_government

    cluster.members.append(
        ThemeMember(
            raw_item_id=vector.raw_item_id,
            source_name=vector.source_name,
            title=vector.title,
            stance=stance,
            summary=summary,
            similarity=similarity,
            published_at=vector.published_at,
        )
    )
    # Update centroid as a running mean of member vectors.
    if cluster.size == 1:
        cluster.centroid = list(vector.embedding)
    else:
        weight_existing = (cluster.size - 1) / cluster.size
        weight_new = 1 / cluster.size
        cluster.centroid = [
            existing * weight_existing + new * weight_new
            for existing, new in zip(cluster.centroid, vector.embedding, strict=True)
        ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _utc_sort_key(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_timestamp(value: datetime | None) -> float:
    if value is None:
        return 0.0
    return _utc_sort_key(value).timestamp()
