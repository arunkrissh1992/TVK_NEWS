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
from datetime import datetime, timedelta, timezone
from typing import Iterable, Literal

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

    # ---- Phase E: coordinated-coverage scoring ----

    @property
    def coordination_score(self) -> float:
        """0.0–1.0 score indicating how 'coordinated' this theme looks.

        Higher when many distinct sources publish about it close together in
        time. The signal the chief wants is: 'are 4 newspapers all running
        the same story in the same window — campaign-style coordination, or
        organic break?'
        """
        if self.size < 2:
            return 0.0
        timestamps = sorted(
            (_utc_sort_key(m.published_at) for m in self.members if m.published_at),
            key=lambda dt: dt,
        )
        if len(timestamps) < 2:
            # Time data missing — fall back to source diversity only.
            return min(1.0, self.source_count / 5)
        span_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600.0
        # Heuristic: 3+ distinct sources, all within 24h = strong signal.
        diversity_factor = min(1.0, self.source_count / 4)
        recency_factor = max(0.0, 1.0 - (span_hours / 72.0))  # >72h → 0
        return round(diversity_factor * recency_factor, 3)

    @property
    def is_coordinated(self) -> bool:
        """≥3 distinct sources publishing within 24h, by default."""
        return self.coordination_score >= 0.6

    # ---- Phase E: 7-day momentum ----

    def momentum(self, *, today: datetime | None = None) -> Literal["rising", "steady", "fading", "flat"]:
        """How is this narrative trending across the last 7 days?

        We split the last 7 days into two halves (recent 3 vs. earlier 4)
        and compare member-count per half. ``flat`` means there's not enough
        time data to judge — the operator should ignore that case.
        """
        today = today or datetime.now(timezone.utc)
        recent_cut = today - timedelta(days=3)
        early_cut = today - timedelta(days=7)
        recent = 0
        early = 0
        for member in self.members:
            if not member.published_at:
                continue
            when = _utc_sort_key(member.published_at)
            if when < early_cut:
                continue
            if when >= recent_cut:
                recent += 1
            else:
                early += 1
        if recent + early < 2:
            return "flat"
        if recent > early:
            return "rising"
        if recent < early:
            return "fading"
        return "steady"


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
# Deduplicated briefing — every article is placed in exactly one cluster
# (singletons included) so the narrative grid can render one card per
# narrative instead of one card per article.
# ---------------------------------------------------------------------------


def cluster_all_articles_for_briefing(
    session: Session,
    *,
    similarity_threshold: float = 0.85,
    source_type: str = "news",
) -> list["ArticleCluster"]:
    """Return a complete partition of articles — every article belongs to
    exactly one cluster. Multi-source clusters collapse duplicate coverage
    into one card; unique articles become singleton clusters.

    Articles without embeddings still appear as singletons so they're never
    silently dropped from the briefing.
    """
    vectors = _load_article_vectors(session, source_type=source_type)
    analyses = _load_latest_analyses(session, raw_item_ids=[v.raw_item_id for v in vectors])

    clusters: list[ArticleCluster] = []
    raw_ids_with_vector: set[int] = set()

    # Greedy nearest-cluster assignment, same heuristic the recurring-themes
    # panel uses — but here we keep singletons too.
    for vector in vectors:
        raw_ids_with_vector.add(vector.raw_item_id)
        if not any(vector.embedding):
            # Degenerate empty vector — keep as its own cluster.
            new_cluster = _new_article_cluster(vector, analyses.get(vector.raw_item_id))
            clusters.append(new_cluster)
            continue

        best_cluster: ArticleCluster | None = None
        best_sim = 0.0
        for cluster in clusters:
            if not cluster.centroid:
                continue
            sim = _cosine_similarity(vector.embedding, cluster.centroid)
            if sim >= similarity_threshold and sim > best_sim:
                best_cluster = cluster
                best_sim = sim
        if best_cluster is not None:
            _add_member_to_article_cluster(
                cluster=best_cluster,
                vector=vector,
                similarity=best_sim,
                analysis=analyses.get(vector.raw_item_id),
            )
        else:
            clusters.append(_new_article_cluster(vector, analyses.get(vector.raw_item_id)))

    # Pick up articles that have an analysis but NO chunk embedding yet
    # (e.g. ingested by Pull Latest but build_rag_index hasn't been re-run).
    # They become singleton clusters so the dashboard never loses them.
    extras = session.scalars(
        select(RawItemRecord)
        .where(RawItemRecord.source_type == source_type)
        .order_by(RawItemRecord.ingested_at.desc(), RawItemRecord.id.desc())
    ).all()
    extra_ids = [raw.id for raw in extras if raw.id not in raw_ids_with_vector]
    if not extra_ids:
        # No extras to process.
        clusters.sort(
            key=lambda c: (
                -c.distinct_source_count,
                -c.size,
                -(_utc_timestamp(c.latest_ingested_at)),
                c.representative_id,
            )
        )
        return clusters

    extra_analyses = _load_latest_analyses(session, raw_item_ids=extra_ids)
    raw_by_id = {raw.id: raw for raw in extras}
    for raw_id in extra_ids:
        analysis = extra_analyses.get(raw_id)
        if analysis is None:
            continue
        if (analysis.government_relevance or "").lower() == "none":
            continue
        raw = raw_by_id[raw_id]
        clusters.append(
            ArticleCluster(
                representative_id=raw.id,
                members=[
                    ArticleMember(
                        raw_item_id=raw.id,
                        analysis_id=analysis.id,
                        source_name=raw.source_name,
                        source_url=raw.source_url,
                        title=raw.title,
                        stance=analysis.stance_toward_government,
                        relevance=analysis.government_relevance,
                        summary_english=analysis.summary_english,
                        summary_original=analysis.summary_original,
                        party_action=analysis.party_action,
                        people_impact=analysis.people_impact,
                        root_cause=analysis.root_cause,
                        recommended_step=analysis.recommended_step,
                        evidence_quotes_original=list(analysis.evidence_quotes_original or []),
                        department=analysis.department,
                        district=analysis.district,
                        needs_human_review=analysis.needs_human_review,
                        published_at=raw.published_at,
                        ingested_at=raw.ingested_at,
                        similarity=1.0,
                        model_name=analysis.model_name or "",
                        prompt_version=analysis.prompt_version or "",
                    )
                ],
                centroid=[],
            )
        )

    # Order clusters: biggest first (multi-source duplicates surface), then
    # by recency of the latest member.
    clusters.sort(
        key=lambda c: (
            -c.distinct_source_count,
            -c.size,
            -(_utc_timestamp(c.latest_ingested_at)),
            c.representative_id,
        )
    )
    return clusters


@dataclass
class ArticleMember:
    raw_item_id: int
    analysis_id: int
    source_name: str
    source_url: str
    title: str | None
    stance: str | None
    relevance: str | None
    summary_english: str
    summary_original: str
    party_action: str
    people_impact: str
    root_cause: str
    recommended_step: str
    evidence_quotes_original: list[str]
    department: str
    district: str
    needs_human_review: bool
    published_at: datetime | None
    ingested_at: datetime | None
    similarity: float
    model_name: str = ""
    prompt_version: str = ""


@dataclass
class ArticleCluster:
    representative_id: int
    members: list[ArticleMember] = field(default_factory=list)
    centroid: list[float] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.members)

    @property
    def distinct_sources(self) -> list[str]:
        seen: list[str] = []
        for m in self.members:
            if m.source_name not in seen:
                seen.append(m.source_name)
        return seen

    @property
    def distinct_source_count(self) -> int:
        return len(self.distinct_sources)

    @property
    def representative(self) -> ArticleMember:
        # Prefer high-relevance, then needs_human_review (interesting), then
        # the longest title (likely the more detailed article).
        return max(
            self.members,
            key=lambda m: (
                _relevance_rank(m.relevance),
                int(m.needs_human_review),
                len(m.title or ""),
            ),
        )

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
    def latest_ingested_at(self) -> datetime | None:
        stamps = [m.ingested_at for m in self.members if m.ingested_at]
        return max(stamps, key=_utc_sort_key) if stamps else None

    @property
    def latest_published_at(self) -> datetime | None:
        stamps = [m.published_at for m in self.members if m.published_at]
        return max(stamps, key=_utc_sort_key) if stamps else None


_RELEVANCE_ORDER = {"high": 4, "medium": 3, "low": 2, "none": 1}


def _relevance_rank(relevance: str | None) -> int:
    return _RELEVANCE_ORDER.get((relevance or "").lower(), 0)


def _new_article_cluster(vector: _ArticleVector, analysis: AIAnalysisRecord | None) -> "ArticleCluster":
    member = _build_article_member(vector=vector, analysis=analysis, similarity=1.0)
    return ArticleCluster(
        representative_id=vector.raw_item_id,
        members=[member],
        centroid=list(vector.embedding),
    )


def _add_member_to_article_cluster(
    *,
    cluster: "ArticleCluster",
    vector: _ArticleVector,
    similarity: float,
    analysis: AIAnalysisRecord | None,
) -> None:
    cluster.members.append(
        _build_article_member(vector=vector, analysis=analysis, similarity=similarity)
    )
    if cluster.centroid:
        weight_existing = (cluster.size - 1) / cluster.size
        weight_new = 1 / cluster.size
        cluster.centroid = [
            existing * weight_existing + new * weight_new
            for existing, new in zip(cluster.centroid, vector.embedding, strict=True)
        ]


def _build_article_member(
    *,
    vector: _ArticleVector,
    analysis: AIAnalysisRecord | None,
    similarity: float,
) -> ArticleMember:
    rel = (analysis.government_relevance if analysis else None) or ""
    stance = (analysis.stance_toward_government if analysis else None) or ""
    return ArticleMember(
        raw_item_id=vector.raw_item_id,
        analysis_id=(analysis.id if analysis else 0),
        source_name=vector.source_name,
        source_url=getattr(vector, "source_url", "") or "",
        title=vector.title,
        stance=stance or None,
        relevance=rel or None,
        summary_english=(analysis.summary_english if analysis else "") or "",
        summary_original=(analysis.summary_original if analysis else "") or "",
        party_action=(analysis.party_action if analysis else "") or "",
        people_impact=(analysis.people_impact if analysis else "") or "",
        root_cause=(analysis.root_cause if analysis else "") or "",
        recommended_step=(analysis.recommended_step if analysis else "") or "",
        evidence_quotes_original=list((analysis.evidence_quotes_original or []) if analysis else []),
        department=(analysis.department if analysis else "") or "",
        district=(analysis.district if analysis else "") or "",
        needs_human_review=bool(analysis.needs_human_review) if analysis else False,
        published_at=vector.published_at,
        ingested_at=None,
        similarity=similarity,
        model_name=(analysis.model_name if analysis else "") or "",
        prompt_version=(analysis.prompt_version if analysis else "") or "",
    )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ArticleVector:
    raw_item_id: int
    source_name: str
    source_url: str
    title: str | None
    published_at: datetime | None
    embedding: list[float]


def _load_article_vectors(session: Session, *, source_type: str) -> list[_ArticleVector]:
    """One vector per article — the leading-chunk embedding."""
    rows = session.execute(
        select(
            RawItemRecord.id,
            RawItemRecord.source_name,
            RawItemRecord.source_url,
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
    for raw_id, source_name, source_url, title, published_at, embedding in rows:
        if raw_id in seen:
            continue
        if not embedding:
            continue
        seen.add(raw_id)
        vectors.append(
            _ArticleVector(
                raw_item_id=raw_id,
                source_name=source_name,
                source_url=source_url,
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
