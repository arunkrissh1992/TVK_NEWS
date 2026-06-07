from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from tnmi.contracts import AIAnalysis, DocumentChunk, NormalizedItem, ReviewDecisionCreate


ID_TYPE = BigInteger().with_variant(Integer, "sqlite")
JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


class Base(DeclarativeBase):
    pass


def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class RawItemRecord(Base):
    __tablename__ = "raw_items"
    __table_args__ = (UniqueConstraint("content_hash", name="uq_raw_items_content_hash"),)

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    source_name: Mapped[str] = mapped_column(String(255), index=True)
    source_url: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )
    language: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text_original: Mapped[str] = mapped_column(Text)
    clean_text_original: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict, server_default=text("'{}'"))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)


class AIAnalysisRecord(Base):
    __tablename__ = "ai_analysis"
    __table_args__ = (
        UniqueConstraint("raw_item_id", "model_name", "prompt_version", name="uq_ai_analysis_raw_model_prompt"),
    )

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    raw_item_id: Mapped[int] = mapped_column(ID_TYPE, ForeignKey("raw_items.id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(64))
    government_relevance: Mapped[str] = mapped_column(String(32), index=True)
    stance_toward_government: Mapped[str] = mapped_column(String(32), index=True)
    tvk_relevance: Mapped[str] = mapped_column(String(32), index=True, default="none", server_default="none")
    tvk_portrayal: Mapped[str] = mapped_column(String(32), index=True, default="neutral", server_default="neutral")
    sentiment: Mapped[str] = mapped_column(String(32))
    target: Mapped[str] = mapped_column(Text)
    political_actors: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list, server_default=text("'[]'"))
    department: Mapped[str] = mapped_column(String(128), index=True)
    district: Mapped[str] = mapped_column(String(128), index=True)
    scheme: Mapped[str | None] = mapped_column(String(255), nullable=True)
    topic: Mapped[str] = mapped_column(Text)
    issue_category: Mapped[str] = mapped_column(String(128))
    people_issue: Mapped[bool] = mapped_column(default=False, server_default=text("false"), index=True)
    public_issue: Mapped[str] = mapped_column(Text, default="", server_default="")
    severity: Mapped[str] = mapped_column(String(64))
    summary_original: Mapped[str] = mapped_column(Text)
    summary_english: Mapped[str] = mapped_column(Text)
    party_action: Mapped[str] = mapped_column(Text, default="", server_default="")
    people_impact: Mapped[str] = mapped_column(Text, default="", server_default="")
    root_cause: Mapped[str] = mapped_column(Text, default="", server_default="")
    recommended_step: Mapped[str] = mapped_column(Text, default="", server_default="")
    action_owner: Mapped[str] = mapped_column(String(128), default="", server_default="")
    action_type: Mapped[str] = mapped_column(String(64), default="", server_default="")
    action_priority: Mapped[str] = mapped_column(String(64), default="low", server_default="low")
    risk_if_ignored: Mapped[str] = mapped_column(Text, default="", server_default="")
    talking_points: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list, server_default=text("'[]'"))
    verification_checklist: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list, server_default=text("'[]'"))
    draft_statement_original: Mapped[str] = mapped_column(Text, default="", server_default="")
    draft_statement_english: Mapped[str] = mapped_column(Text, default="", server_default="")
    positive_points: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list, server_default=text("'[]'"))
    negative_points: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list, server_default=text("'[]'"))
    evidence_quotes_original: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list, server_default=text("'[]'"))
    evidence_quotes_english: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list, server_default=text("'[]'"))
    confidence: Mapped[float] = mapped_column(Float)
    needs_human_review: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


class ReviewDecisionRecord(Base):
    __tablename__ = "review_decisions"

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(ID_TYPE, ForeignKey("ai_analysis.id", ondelete="CASCADE"), index=True)
    reviewer_name: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    note: Mapped[str] = mapped_column(Text, default="", server_default="")
    corrected_stance: Mapped[str | None] = mapped_column(String(32), nullable=True)
    corrected_relevance: Mapped[str | None] = mapped_column(String(32), nullable=True)
    corrected_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


class SourceCheckpointRecord(Base):
    __tablename__ = "source_checkpoints"
    __table_args__ = (
        UniqueConstraint("source_type", "source_key", "cursor_name", name="uq_source_checkpoint_key"),
    )

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    source_key: Mapped[str] = mapped_column(String(255), index=True)
    cursor_name: Mapped[str] = mapped_column(String(64))
    cursor_value: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict, server_default=text("'{}'"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (
        UniqueConstraint("raw_item_id", "chunk_version", "chunk_index", name="uq_document_chunk_raw_version_index"),
    )

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    raw_item_id: Mapped[int] = mapped_column(ID_TYPE, ForeignKey("raw_items.id", ondelete="CASCADE"), index=True)
    chunk_version: Mapped[str] = mapped_column(String(64), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    chunk_text: Mapped[str] = mapped_column(Text)
    token_estimate: Mapped[int] = mapped_column(Integer)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict, server_default=text("'{}'"))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


class ChunkEmbeddingRecord(Base):
    __tablename__ = "chunk_embeddings"
    __table_args__ = (
        UniqueConstraint("chunk_id", "provider_name", "model_name", name="uq_chunk_embedding_provider_model"),
    )

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    chunk_id: Mapped[int] = mapped_column(
        ID_TYPE,
        ForeignKey("document_chunks.id", ondelete="CASCADE"),
        index=True,
    )
    provider_name: Mapped[str] = mapped_column(String(128), index=True)
    model_name: Mapped[str] = mapped_column(String(128), index=True)
    embedding_dimension: Mapped[int] = mapped_column(Integer)
    embedding: Mapped[list[float]] = mapped_column(JSON_TYPE, default=list, server_default=text("'[]'"))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    engine = create_engine(database_url, future=True)
    if engine.dialect.name == "sqlite":
        event.listen(engine, "connect", _enable_sqlite_foreign_keys)
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


def init_db(session_factory: sessionmaker[Session]) -> None:
    engine = session_factory.kw["bind"]
    Base.metadata.create_all(engine)
    _apply_lightweight_migrations(engine)


def _apply_lightweight_migrations(engine: Any) -> None:
    # Forward-only column additions for tables that already exist in older
    # SQLite demo databases. Adding nullable columns with defaults is safe.
    expected_columns = {
        "ai_analysis": (
            ("party_action", "TEXT NOT NULL DEFAULT ''", "TEXT NOT NULL DEFAULT ''"),
            ("people_impact", "TEXT NOT NULL DEFAULT ''", "TEXT NOT NULL DEFAULT ''"),
            ("root_cause", "TEXT NOT NULL DEFAULT ''", "TEXT NOT NULL DEFAULT ''"),
            ("recommended_step", "TEXT NOT NULL DEFAULT ''", "TEXT NOT NULL DEFAULT ''"),
            ("tvk_relevance", "VARCHAR(32) NOT NULL DEFAULT 'none'", "VARCHAR(32) NOT NULL DEFAULT 'none'"),
            ("tvk_portrayal", "VARCHAR(32) NOT NULL DEFAULT 'neutral'", "VARCHAR(32) NOT NULL DEFAULT 'neutral'"),
            ("political_actors", "JSON NOT NULL DEFAULT '[]'", "JSONB NOT NULL DEFAULT '[]'::jsonb"),
            ("people_issue", "BOOLEAN NOT NULL DEFAULT 0", "BOOLEAN NOT NULL DEFAULT false"),
            ("public_issue", "TEXT NOT NULL DEFAULT ''", "TEXT NOT NULL DEFAULT ''"),
            ("action_owner", "VARCHAR(128) NOT NULL DEFAULT ''", "VARCHAR(128) NOT NULL DEFAULT ''"),
            ("action_type", "VARCHAR(64) NOT NULL DEFAULT ''", "VARCHAR(64) NOT NULL DEFAULT ''"),
            ("action_priority", "VARCHAR(64) NOT NULL DEFAULT 'low'", "VARCHAR(64) NOT NULL DEFAULT 'low'"),
            ("risk_if_ignored", "TEXT NOT NULL DEFAULT ''", "TEXT NOT NULL DEFAULT ''"),
            ("talking_points", "JSON NOT NULL DEFAULT '[]'", "JSONB NOT NULL DEFAULT '[]'::jsonb"),
            ("verification_checklist", "JSON NOT NULL DEFAULT '[]'", "JSONB NOT NULL DEFAULT '[]'::jsonb"),
            ("draft_statement_original", "TEXT NOT NULL DEFAULT ''", "TEXT NOT NULL DEFAULT ''"),
            ("draft_statement_english", "TEXT NOT NULL DEFAULT ''", "TEXT NOT NULL DEFAULT ''"),
        ),
    }
    with engine.begin() as connection:
        for table_name, columns in expected_columns.items():
            existing_rows = connection.execute(text(f"PRAGMA table_info('{table_name}')")).all() \
                if engine.dialect.name == "sqlite" else []
            existing_names = {row[1] for row in existing_rows} if engine.dialect.name == "sqlite" else set()
            for column_name, sqlite_column_ddl, postgres_column_ddl in columns:
                if engine.dialect.name == "sqlite":
                    if column_name in existing_names:
                        continue
                    connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {sqlite_column_ddl}"))
                else:
                    connection.execute(
                        text(
                            f"ALTER TABLE {table_name} "
                            f"ADD COLUMN IF NOT EXISTS {column_name} {postgres_column_ddl}"
                        )
                    )


def compute_content_hash(item: NormalizedItem) -> str:
    return hashlib.sha256(item.content_hash_input().encode("utf-8")).hexdigest()


def compute_chunk_hash(chunk: DocumentChunk) -> str:
    return hashlib.sha256(chunk.content_hash_input().encode("utf-8")).hexdigest()


def save_raw_item(session: Session, item: NormalizedItem) -> RawItemRecord:
    content_hash = compute_content_hash(item)
    existing = session.scalar(select(RawItemRecord).where(RawItemRecord.content_hash == content_hash))
    if existing:
        return existing

    record = RawItemRecord(
        source_type=item.source_type.value,
        source_name=item.source_name,
        source_url=item.source_url,
        published_at=item.published_at,
        language=item.language,
        title=item.title,
        raw_text_original=item.raw_text_original,
        clean_text_original=item.clean_text_original,
        metadata_json=item.metadata,
        content_hash=content_hash,
    )
    try:
        with session.begin_nested():
            session.add(record)
            session.flush()
    except IntegrityError:
        existing = session.scalar(select(RawItemRecord).where(RawItemRecord.content_hash == content_hash))
        if existing:
            return existing
        raise

    return record


def save_document_chunk(session: Session, chunk: DocumentChunk) -> DocumentChunkRecord:
    existing = session.scalar(
        select(DocumentChunkRecord).where(
            DocumentChunkRecord.raw_item_id == chunk.raw_item_id,
            DocumentChunkRecord.chunk_version == chunk.chunk_version,
            DocumentChunkRecord.chunk_index == chunk.chunk_index,
        )
    )
    if existing:
        return existing

    record = DocumentChunkRecord(
        raw_item_id=chunk.raw_item_id,
        chunk_version=chunk.chunk_version,
        chunk_index=chunk.chunk_index,
        chunk_text=chunk.chunk_text,
        token_estimate=chunk.token_estimate,
        metadata_json=chunk.metadata,
        content_hash=compute_chunk_hash(chunk),
    )
    try:
        with session.begin_nested():
            session.add(record)
            session.flush()
    except IntegrityError:
        existing = session.scalar(
            select(DocumentChunkRecord).where(
                DocumentChunkRecord.raw_item_id == chunk.raw_item_id,
                DocumentChunkRecord.chunk_version == chunk.chunk_version,
                DocumentChunkRecord.chunk_index == chunk.chunk_index,
            )
        )
        if existing:
            return existing
        raise
    return record


def save_document_chunks(session: Session, chunks: list[DocumentChunk]) -> list[DocumentChunkRecord]:
    return [save_document_chunk(session, chunk) for chunk in chunks]


def get_document_chunks(
    session: Session,
    raw_item_id: int,
    *,
    chunk_version: str | None = None,
) -> list[DocumentChunkRecord]:
    statement = select(DocumentChunkRecord).where(DocumentChunkRecord.raw_item_id == raw_item_id)
    if chunk_version is not None:
        statement = statement.where(DocumentChunkRecord.chunk_version == chunk_version)
    return list(session.scalars(statement.order_by(DocumentChunkRecord.chunk_index.asc())))


def save_chunk_embedding(
    session: Session,
    *,
    chunk_id: int,
    provider_name: str,
    model_name: str,
    embedding: list[float],
) -> ChunkEmbeddingRecord:
    existing = get_chunk_embedding(
        session,
        chunk_id,
        provider_name=provider_name,
        model_name=model_name,
    )
    if existing:
        return existing

    record = ChunkEmbeddingRecord(
        chunk_id=chunk_id,
        provider_name=provider_name,
        model_name=model_name,
        embedding_dimension=len(embedding),
        embedding=embedding,
    )
    try:
        with session.begin_nested():
            session.add(record)
            session.flush()
    except IntegrityError:
        existing = get_chunk_embedding(
            session,
            chunk_id,
            provider_name=provider_name,
            model_name=model_name,
        )
        if existing:
            return existing
        raise
    return record


def get_chunk_embedding(
    session: Session,
    chunk_id: int,
    *,
    provider_name: str,
    model_name: str,
) -> ChunkEmbeddingRecord | None:
    return session.scalar(
        select(ChunkEmbeddingRecord).where(
            ChunkEmbeddingRecord.chunk_id == chunk_id,
            ChunkEmbeddingRecord.provider_name == provider_name,
            ChunkEmbeddingRecord.model_name == model_name,
        )
    )


def save_ai_analysis(
    session: Session,
    raw_item_id: int,
    analysis: AIAnalysis,
    *,
    model_name: str,
    prompt_version: str,
) -> AIAnalysisRecord:
    existing = session.scalar(
        select(AIAnalysisRecord).where(
            AIAnalysisRecord.raw_item_id == raw_item_id,
            AIAnalysisRecord.model_name == model_name,
            AIAnalysisRecord.prompt_version == prompt_version,
        )
    )
    if existing:
        return existing

    record = AIAnalysisRecord(
        raw_item_id=raw_item_id,
        model_name=model_name,
        prompt_version=prompt_version,
        government_relevance=analysis.government_relevance.value,
        stance_toward_government=analysis.stance_toward_government.value,
        tvk_relevance=(analysis.tvk_relevance or analysis.government_relevance).value,
        tvk_portrayal=(analysis.tvk_portrayal or analysis.stance_toward_government).value,
        sentiment=analysis.sentiment.value,
        target=analysis.target,
        political_actors=analysis.political_actors,
        department=analysis.department,
        district=analysis.district,
        scheme=analysis.scheme,
        topic=analysis.topic,
        issue_category=analysis.issue_category,
        people_issue=bool(analysis.people_issue),
        public_issue=analysis.public_issue,
        severity=analysis.severity.value,
        summary_original=analysis.summary_original,
        summary_english=analysis.summary_english,
        party_action=analysis.party_action,
        people_impact=analysis.people_impact,
        root_cause=analysis.root_cause,
        recommended_step=analysis.recommended_step,
        action_owner=analysis.action_owner,
        action_type=analysis.action_type,
        action_priority=(analysis.action_priority or analysis.severity).value,
        risk_if_ignored=analysis.risk_if_ignored,
        talking_points=analysis.talking_points,
        verification_checklist=analysis.verification_checklist,
        draft_statement_original=analysis.draft_statement_original,
        draft_statement_english=analysis.draft_statement_english,
        positive_points=analysis.positive_points,
        negative_points=analysis.negative_points,
        evidence_quotes_original=analysis.evidence_quotes_original,
        evidence_quotes_english=analysis.evidence_quotes_english,
        confidence=analysis.confidence,
        needs_human_review=analysis.needs_human_review,
    )
    try:
        with session.begin_nested():
            session.add(record)
            session.flush()
    except IntegrityError:
        existing = session.scalar(
            select(AIAnalysisRecord).where(
                AIAnalysisRecord.raw_item_id == raw_item_id,
                AIAnalysisRecord.model_name == model_name,
                AIAnalysisRecord.prompt_version == prompt_version,
            )
        )
        if existing:
            return existing
        raise

    return record


def get_ai_analysis(
    session: Session,
    raw_item_id: int,
    *,
    model_name: str,
    prompt_version: str,
) -> AIAnalysisRecord | None:
    return session.scalar(
        select(AIAnalysisRecord).where(
            AIAnalysisRecord.raw_item_id == raw_item_id,
            AIAnalysisRecord.model_name == model_name,
            AIAnalysisRecord.prompt_version == prompt_version,
        )
    )


def save_review_decision(session: Session, decision: ReviewDecisionCreate) -> ReviewDecisionRecord:
    record = ReviewDecisionRecord(
        analysis_id=decision.analysis_id,
        reviewer_name=decision.reviewer_name.strip(),
        status=decision.status.value,
        note=decision.note.strip(),
        corrected_stance=decision.corrected_stance.value if decision.corrected_stance else None,
        corrected_relevance=decision.corrected_relevance.value if decision.corrected_relevance else None,
        corrected_summary=decision.corrected_summary.strip() if decision.corrected_summary else None,
    )
    session.add(record)
    session.flush()
    return record


def get_latest_review_decision(session: Session, analysis_id: int) -> ReviewDecisionRecord | None:
    return session.scalar(
        select(ReviewDecisionRecord)
        .where(ReviewDecisionRecord.analysis_id == analysis_id)
        .order_by(ReviewDecisionRecord.created_at.desc(), ReviewDecisionRecord.id.desc())
        .limit(1)
    )


def get_source_checkpoint(
    session: Session,
    *,
    source_type: str,
    source_key: str,
    cursor_name: str,
) -> SourceCheckpointRecord | None:
    return session.scalar(
        select(SourceCheckpointRecord).where(
            SourceCheckpointRecord.source_type == source_type,
            SourceCheckpointRecord.source_key == source_key,
            SourceCheckpointRecord.cursor_name == cursor_name,
        )
    )


def save_source_checkpoint(
    session: Session,
    *,
    source_type: str,
    source_key: str,
    cursor_name: str,
    cursor_value: str,
    metadata: dict[str, Any] | None = None,
) -> SourceCheckpointRecord:
    existing = get_source_checkpoint(
        session,
        source_type=source_type,
        source_key=source_key,
        cursor_name=cursor_name,
    )
    if existing:
        existing.cursor_value = cursor_value
        existing.metadata_json = metadata or {}
        existing.updated_at = datetime.now(timezone.utc)
        session.flush()
        return existing

    record = SourceCheckpointRecord(
        source_type=source_type,
        source_key=source_key,
        cursor_name=cursor_name,
        cursor_value=cursor_value,
        metadata_json=metadata or {},
    )
    session.add(record)
    session.flush()
    return record
