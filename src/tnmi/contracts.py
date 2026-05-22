from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator


class SourceType(StrEnum):
    NEWS = "news"
    X = "x"
    INSTAGRAM = "instagram"
    VIDEO = "video"
    YOUTUBE = "youtube"
    PRESS_RELEASE = "press_release"
    MANUAL_UPLOAD = "manual_upload"
    PROVIDER_EXPORT = "provider_export"


class GovernmentRelevance(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


class Stance(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    MIXED = "mixed"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewStatus(StrEnum):
    APPROVED = "approved"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"
    CORRECTED = "corrected"


class NewspaperSource(BaseModel):
    name: str
    source_type: SourceType = SourceType.NEWS
    language_hint: str = "ta"
    priority: int = Field(default=5, ge=1, le=10)
    active: bool = True
    rss_urls: list[HttpUrl] = Field(default_factory=list)
    sitemap_urls: list[HttpUrl] = Field(default_factory=list)
    section_urls: list[HttpUrl] = Field(default_factory=list)
    legal_notes: str = "Public newspaper source; respect robots, rate limits, and terms."


class XHandleSource(BaseModel):
    handle: str = Field(min_length=1, max_length=15, pattern=r"^[A-Za-z0-9_]{1,15}$")
    display_name: str | None = None
    source_type: SourceType = SourceType.X
    language_hint: str = "ta-en-mixed"
    priority: int = Field(default=5, ge=1, le=10)
    active: bool = True
    legal_notes: str = "Public X source; use official X API access only."

    @field_validator("handle", mode="before")
    @classmethod
    def strip_at_prefix(cls, value: object) -> object:
        if isinstance(value, str):
            return value.removeprefix("@")
        return value

    @property
    def source_name(self) -> str:
        return f"@{self.handle}"


class NormalizedItem(BaseModel):
    source_type: SourceType
    source_name: str
    # Original source locator; may be an HTTP URL or a provider/manual upload locator.
    source_url: str
    published_at: datetime | None = None
    language: str
    title: str | None = None
    raw_text_original: str
    clean_text_original: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def content_hash_input(self) -> str:
        return json.dumps(
            {
                "source_type": self.source_type.value,
                "source_url": self.source_url,
                "title": self.title or "",
                "clean_text_original": self.clean_text_original,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )


class AIAnalysis(BaseModel):
    government_relevance: GovernmentRelevance
    stance_toward_government: Stance
    sentiment: Sentiment
    target: str
    department: str
    district: str
    scheme: str | None = None
    topic: str
    issue_category: str
    severity: Severity
    summary_original: str
    summary_english: str
    positive_points: list[str] = Field(default_factory=list)
    negative_points: list[str] = Field(default_factory=list)
    evidence_quotes_original: list[str] = Field(default_factory=list)
    evidence_quotes_english: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool


class XPost(BaseModel):
    id: str = Field(min_length=1)
    handle: str = Field(min_length=1, max_length=15)
    text: str
    created_at: datetime | None = None
    lang: str | None = None
    public_metrics: dict[str, int] = Field(default_factory=dict)
    url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReviewDecisionCreate(BaseModel):
    analysis_id: int
    reviewer_name: str = Field(min_length=1, max_length=128)
    status: ReviewStatus
    note: str = Field(default="", max_length=4000)
    corrected_stance: Stance | None = None
    corrected_relevance: GovernmentRelevance | None = None
    corrected_summary: str | None = Field(default=None, max_length=4000)
