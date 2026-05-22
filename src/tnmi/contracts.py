from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl


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


class NormalizedItem(BaseModel):
    source_type: SourceType
    source_name: str
    source_url: str
    published_at: datetime | None = None
    language: str
    title: str | None = None
    raw_text_original: str
    clean_text_original: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    def content_hash_input(self) -> str:
        return "|".join(
            [
                self.source_type.value,
                self.source_url,
                self.title or "",
                self.clean_text_original,
            ]
        )


class AIAnalysis(BaseModel):
    government_relevance: GovernmentRelevance
    stance_toward_government: Stance
    sentiment: str
    target: str
    department: str
    district: str
    scheme: str | None = None
    topic: str
    issue_category: str
    severity: str
    summary_original: str
    summary_english: str
    positive_points: list[str] = Field(default_factory=list)
    negative_points: list[str] = Field(default_factory=list)
    evidence_quotes_original: list[str] = Field(default_factory=list)
    evidence_quotes_english: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool
