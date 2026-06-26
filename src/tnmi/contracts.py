from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator


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


class LabelTier(StrEnum):
    """Data-quality tier for a training label.

    BRONZE — raw single-model output, unverified.
    SILVER — agreed by an independent (teacher) model or high-confidence; usable
             as bulk training signal but never as the yardstick.
    GOLD   — human-verified. The only tier that gates model promotion, and the
             only tier the frozen eval test set is drawn from.
    """

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"


class LabelProvenance(StrEnum):
    """Where a label came from — kept on every row so training can weight,
    downweight, or exclude a source, and so collapse is auditable."""

    AI = "ai"  # raw student-model output (bronze)
    AI_HIGH_CONF = "ai_high_conf"  # student agreed with itself at high confidence (silver)
    TEACHER_MODEL = "teacher_model"  # an independent stronger model agreed (silver)
    HUMAN = "human"  # human reviewer verified or corrected (gold)
    ENRICHED = "enriched"  # verifiable structured join (geo/entity/demographic)


# The classifier targets we collect labels for and train/evaluate against.
LABEL_FIELDS: tuple[str, ...] = (
    "government_relevance",
    "tvk_relevance",
    "stance_toward_government",
    "tvk_portrayal",
    "people_issue",
    "issue_category",
    "severity",
)


class EntityType(StrEnum):
    """Kinds of canonical political objects the knowledge vault tracks."""

    PERSON = "person"
    PARTY = "party"
    OFFICE = "office"  # role-word mentions ("Chief Minister", "Minister", "MLA")
    ORGANIZATION = "org"
    SOURCE = "source"
    DEPARTMENT = "department"
    DISTRICT = "district"
    CONSTITUENCY = "constituency"
    SCHEME = "scheme"


class EntityStatus(StrEnum):
    ACTIVE = "active"
    CANDIDATE = "candidate"  # auto-created from an unknown surface; needs human confirmation
    RETIRED = "retired"


class EntitySeed(BaseModel):
    """One curated entity from configs/entities.seed.yaml.

    The seed file is config, not code — it encodes the deployment's political
    reality (who holds which office, which parties matter) so operators can
    edit the roster without touching Python.
    """

    entity_type: EntityType
    slug: str = Field(min_length=1, max_length=160)
    canonical_name: str = Field(min_length=1, max_length=255)
    name_ta: str = ""
    role: str = ""
    party: str = ""
    district: str = ""
    portfolio: str = ""
    is_tvk: bool = False
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class NewspaperSource(BaseModel):
    name: str
    source_type: SourceType = SourceType.NEWS
    language_hint: str = "ta"
    priority: int = Field(default=5, ge=1, le=10)
    active: bool = True
    rss_urls: list[HttpUrl] = Field(default_factory=list)
    sitemap_urls: list[HttpUrl] = Field(default_factory=list)
    section_urls: list[HttpUrl] = Field(default_factory=list)
    # TN coverage metadata (Phase B). "statewide" means the paper covers all
    # 38 districts. A specific district name (e.g. "Madurai") means the paper
    # is the local edition for that district.
    coverage_scope: str = "statewide"
    district_focus: str | None = None
    # Optional cross-reference to Media Cloud's source registry — lets us
    # reconcile imports later without duplicating sources.
    mediacloud_media_id: int | None = None
    legal_notes: str = "Public newspaper source; respect robots, rate limits, and terms."


class YouTubeChannelSource(BaseModel):
    """A monitored YouTube channel — typically a Tamil-language news outlet.

    The ingestion pipeline uses the YouTube Data API to discover the newest
    videos for each channel, downloads the audio via yt-dlp, transcribes
    with faster-whisper (Tamil), then runs the transcript through the same
    AIAnalysis pipeline as newspaper articles.
    """

    name: str
    channel_id: str = Field(min_length=1, max_length=64)
    source_type: SourceType = SourceType.YOUTUBE
    language_hint: str = "ta"
    priority: int = Field(default=5, ge=1, le=10)
    active: bool = True
    coverage_scope: str = "statewide"
    district_focus: str | None = None
    legal_notes: str = "Public YouTube channel; transcripts produced under fair-use research."

    @property
    def source_name(self) -> str:
        return self.name

    @property
    def channel_url(self) -> str:
        return f"https://www.youtube.com/channel/{self.channel_id}"


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
    tvk_relevance: GovernmentRelevance | None = None
    tvk_portrayal: Stance | None = None
    sentiment: Sentiment
    target: str
    political_actors: list[str] = Field(default_factory=list)
    department: str
    district: str
    scheme: str | None = None
    topic: str
    issue_category: str
    people_issue: bool | None = None
    public_issue: str = ""
    severity: Severity
    summary_original: str
    summary_english: str
    # Chief's briefing lens — short, crisp, action-oriented. Empty string when
    # the article does not warrant that angle (e.g. a story unrelated to TVK
    # members has no party_action line).
    party_action: str = ""
    people_impact: str = ""
    root_cause: str = ""
    recommended_step: str = ""
    action_owner: str = ""
    action_type: str = ""
    action_priority: Severity | None = None
    # Action playbook — populated for negative / people-issue items so the
    # leadership office gets a ready-to-act brief, not just a one-line step.
    # All empty for positive, neutral, or out-of-scope rows.
    risk_if_ignored: str = ""
    talking_points: list[str] = Field(default_factory=list)
    verification_checklist: list[str] = Field(default_factory=list)
    draft_statement_original: str = ""
    draft_statement_english: str = ""
    positive_points: list[str] = Field(default_factory=list)
    negative_points: list[str] = Field(default_factory=list)
    evidence_quotes_original: list[str] = Field(default_factory=list)
    evidence_quotes_english: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    needs_human_review: bool

    @model_validator(mode="after")
    def backfill_tvk_intelligence_fields(self) -> "AIAnalysis":
        if self.tvk_relevance is None:
            self.tvk_relevance = self.government_relevance
        if self.tvk_portrayal is None:
            self.tvk_portrayal = self.stance_toward_government
        if self.people_issue is None:
            self.people_issue = bool(
                self.people_impact.strip()
                or self.public_issue.strip()
                or self.stance_toward_government in {Stance.NEGATIVE, Stance.MIXED}
                or self.needs_human_review
            )
        if self.action_priority is None:
            self.action_priority = self.severity
        return self


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


class DocumentChunk(BaseModel):
    raw_item_id: int
    source_type: SourceType
    source_name: str
    source_url: str
    language: str
    title: str | None = None
    published_at: datetime | None = None
    chunk_version: str
    chunk_index: int = Field(ge=0)
    chunk_text: str = Field(min_length=1)
    token_estimate: int = Field(ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def content_hash_input(self) -> str:
        return json.dumps(
            {
                "raw_item_id": self.raw_item_id,
                "chunk_version": self.chunk_version,
                "chunk_index": self.chunk_index,
                "chunk_text": self.chunk_text,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
