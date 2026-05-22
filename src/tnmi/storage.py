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
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from tnmi.contracts import AIAnalysis, NormalizedItem


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
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON_TYPE, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)


class AIAnalysisRecord(Base):
    __tablename__ = "ai_analysis"

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    raw_item_id: Mapped[int] = mapped_column(ID_TYPE, ForeignKey("raw_items.id", ondelete="CASCADE"), index=True)
    model_name: Mapped[str] = mapped_column(String(128))
    prompt_version: Mapped[str] = mapped_column(String(64))
    government_relevance: Mapped[str] = mapped_column(String(32), index=True)
    stance_toward_government: Mapped[str] = mapped_column(String(32), index=True)
    sentiment: Mapped[str] = mapped_column(String(32))
    target: Mapped[str] = mapped_column(Text)
    department: Mapped[str] = mapped_column(String(128), index=True)
    district: Mapped[str] = mapped_column(String(128), index=True)
    scheme: Mapped[str | None] = mapped_column(String(255), nullable=True)
    topic: Mapped[str] = mapped_column(Text)
    issue_category: Mapped[str] = mapped_column(String(128))
    severity: Mapped[str] = mapped_column(String(64))
    summary_original: Mapped[str] = mapped_column(Text)
    summary_english: Mapped[str] = mapped_column(Text)
    positive_points: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list)
    negative_points: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list)
    evidence_quotes_original: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list)
    evidence_quotes_english: Mapped[list[str]] = mapped_column(JSON_TYPE, default=list)
    confidence: Mapped[float] = mapped_column(Float)
    needs_human_review: Mapped[bool]
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
    Base.metadata.create_all(session_factory.kw["bind"])


def compute_content_hash(item: NormalizedItem) -> str:
    return hashlib.sha256(item.content_hash_input().encode("utf-8")).hexdigest()


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


def save_ai_analysis(
    session: Session,
    raw_item_id: int,
    analysis: AIAnalysis,
    *,
    model_name: str,
    prompt_version: str,
) -> AIAnalysisRecord:
    record = AIAnalysisRecord(
        raw_item_id=raw_item_id,
        model_name=model_name,
        prompt_version=prompt_version,
        government_relevance=analysis.government_relevance.value,
        stance_toward_government=analysis.stance_toward_government.value,
        sentiment=analysis.sentiment.value,
        target=analysis.target,
        department=analysis.department,
        district=analysis.district,
        scheme=analysis.scheme,
        topic=analysis.topic,
        issue_category=analysis.issue_category,
        severity=analysis.severity.value,
        summary_original=analysis.summary_original,
        summary_english=analysis.summary_english,
        positive_points=analysis.positive_points,
        negative_points=analysis.negative_points,
        evidence_quotes_original=analysis.evidence_quotes_original,
        evidence_quotes_english=analysis.evidence_quotes_english,
        confidence=analysis.confidence,
        needs_human_review=analysis.needs_human_review,
    )
    session.add(record)
    session.flush()
    return record
