# Daily Newspaper Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first production-shaped vertical slice: daily ingestion, extraction, Tamil-first AI analysis, storage, and reporting for configured newspaper sources.

**Architecture:** Implement a Python-first modular service inside the private repo. The first slice uses domain contracts, source configuration, SQLAlchemy storage, feedparser/trafilatura adapters, a mockable AI provider interface, a command-line pipeline runner, an Airflow-compatible DAG wrapper, and a small FastAPI API for inspection.

**Tech Stack:** Python 3.11+, Pydantic, SQLAlchemy, FastAPI, feedparser, trafilatura, PyYAML, OpenAI SDK, pytest, Docker Compose, PostgreSQL, Redis, OpenSearch, MinIO.

---

## Scope

This plan implements Milestone 1 from the production spec:

- Daily newspaper source registry.
- RSS/sitemap/manual article ingestion.
- Article extraction.
- Language detection for Tamil, English, and mixed content.
- Normalized item storage.
- AI classification interface with mock and OpenAI providers.
- Daily newspaper report generation.
- Basic API endpoints for sources, items, analyses, and reports.
- Local Docker Compose for production-shaped development.

It does not implement X, Instagram/video, OCR, transcription, dashboard UI, email delivery, or Kubernetes deployment. Those get separate plans.

## File Structure

Create these files:

- `pyproject.toml`: Python project metadata, dependencies, pytest config.
- `.gitignore`: excludes env files, caches, generated reports, object storage.
- `.env.example`: documented local environment values.
- `README.md`: local setup and Milestone 1 commands.
- `configs/sources.newspapers.yaml`: initial newspaper registry.
- `db/schema.sql`: production Postgres schema for Milestone 1 tables.
- `src/tnmi/__init__.py`: package marker.
- `src/tnmi/contracts.py`: Pydantic models and enums shared by pipeline/API.
- `src/tnmi/config.py`: settings and newspaper config loader.
- `src/tnmi/language.py`: Tamil/English/mixed language detection.
- `src/tnmi/storage.py`: SQLAlchemy engine, ORM models, repositories.
- `src/tnmi/news.py`: feed parsing and article extraction adapters.
- `src/tnmi/ai.py`: AI provider protocol, mock provider, OpenAI provider.
- `src/tnmi/reports.py`: daily newspaper report builder.
- `src/tnmi/pipeline.py`: orchestrates daily newspaper ingestion and analysis.
- `apps/api/main.py`: FastAPI app for inspecting Milestone 1 data.
- `pipelines/run_daily_news.py`: CLI entrypoint.
- `pipelines/dags/daily_news_intelligence.py`: Airflow DAG wrapper.
- `docker-compose.yml`: Postgres, Redis, OpenSearch, MinIO.
- `tests/fixtures/sample_feed.xml`: deterministic RSS fixture.
- `tests/fixtures/sample_article.html`: deterministic article fixture.
- `tests/test_contracts.py`: contract validation tests.
- `tests/test_config.py`: source config tests.
- `tests/test_language.py`: language detection tests.
- `tests/test_news.py`: feed/extraction tests.
- `tests/test_ai.py`: AI schema/provider tests.
- `tests/test_storage.py`: storage and dedupe tests.
- `tests/test_pipeline.py`: end-to-end pipeline test using fakes.
- `tests/test_reports.py`: report generation test.
- `tests/test_api.py`: API route tests.

## Task 1: Project Foundation

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `README.md`
- Create: `src/tnmi/__init__.py`
- Test: `tests/test_contracts.py`

- [ ] **Step 1: Write the initial import smoke test**

Create `tests/test_contracts.py`:

```python
def test_package_imports():
    import tnmi

    assert tnmi.__version__ == "0.1.0"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_contracts.py -v
```

Expected: FAIL because `tnmi` does not exist yet.

- [ ] **Step 3: Add project metadata**

Create `pyproject.toml`:

```toml
[project]
name = "tn-media-intelligence"
version = "0.1.0"
description = "Tamil-first public media intelligence platform"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.30.0",
  "pydantic>=2.8.0",
  "pydantic-settings>=2.4.0",
  "sqlalchemy>=2.0.0",
  "psycopg[binary]>=3.2.0",
  "feedparser>=6.0.0",
  "trafilatura>=1.12.0",
  "PyYAML>=6.0.0",
  "openai>=1.0.0",
  "python-dotenv>=1.0.0",
  "jinja2>=3.1.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
  "pytest-cov>=5.0.0",
  "httpx>=0.27.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
```

Create `.gitignore`:

```gitignore
.env
.venv/
__pycache__/
.pytest_cache/
.coverage
htmlcov/
reports/generated/
object-store/
*.pyc
```

Create `.env.example`:

```bash
DATABASE_URL=postgresql+psycopg://mediaintel:mediaintel@localhost:5432/mediaintel
OPENAI_API_KEY=
OPENAI_MODEL_ITEM_CLASSIFIER=gpt-5.4-mini
OPENAI_MODEL_REPORT=gpt-5.5
NEWS_SOURCE_CONFIG=configs/sources.newspapers.yaml
REPORT_OUTPUT_DIR=reports/generated
```

Create `src/tnmi/__init__.py`:

```python
__version__ = "0.1.0"
```

Create `README.md`:

```markdown
# Tamil Nadu Public Media Intelligence

Tamil-first public media intelligence platform for newspaper, social, and video analysis.

## Milestone 1

Daily newspaper ingestion and AI analysis.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

## Run Daily News Pipeline

```powershell
python pipelines/run_daily_news.py --date 2026-05-21
```
```

- [ ] **Step 4: Run smoke test**

Run:

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_contracts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add pyproject.toml .gitignore .env.example README.md src/tnmi/__init__.py tests/test_contracts.py
git commit -m "chore: initialize newspaper pipeline project"
```

## Task 2: Domain Contracts

**Files:**
- Modify: `src/tnmi/contracts.py`
- Modify: `tests/test_contracts.py`

- [ ] **Step 1: Add failing contract tests**

Replace `tests/test_contracts.py` with:

```python
from datetime import datetime, timezone

from tnmi import __version__
from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
    SourceType,
    Stance,
)


def test_package_imports():
    assert __version__ == "0.1.0"


def test_normalized_item_accepts_tamil_news_article():
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example Tamil Daily",
        source_url="https://example.com/article",
        published_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        language="ta",
        title="தமிழக அரசு புதிய திட்டம் அறிவிப்பு",
        raw_text_original="தமிழக அரசு இன்று புதிய திட்டத்தை அறிவித்தது.",
        clean_text_original="தமிழக அரசு இன்று புதிய திட்டத்தை அறிவித்தது.",
        metadata={"section": "politics"},
    )

    assert item.source_type == SourceType.NEWS
    assert item.content_hash_input().startswith("news|https://example.com/article|")


def test_ai_analysis_schema_has_evidence_and_review_flag():
    analysis = AIAnalysis(
        government_relevance=GovernmentRelevance.HIGH,
        stance_toward_government=Stance.POSITIVE,
        sentiment="positive",
        target="Tamil Nadu Government",
        department="welfare",
        district="unknown",
        scheme=None,
        topic="new scheme",
        issue_category="welfare",
        severity="low",
        summary_original="அரசு திட்டம் குறித்து சாதகமான செய்தி.",
        summary_english="Positive coverage about a government scheme.",
        positive_points=["Scheme announcement was described favorably."],
        negative_points=[],
        evidence_quotes_original=["புதிய திட்டத்தை அறிவித்தது"],
        evidence_quotes_english=["announced a new scheme"],
        confidence=0.86,
        needs_human_review=False,
    )

    assert analysis.confidence == 0.86
    assert analysis.needs_human_review is False
```

- [ ] **Step 2: Run contract tests to verify failure**

Run:

```powershell
python -m pytest tests/test_contracts.py -v
```

Expected: FAIL because `tnmi.contracts` does not exist.

- [ ] **Step 3: Implement contracts**

Create `src/tnmi/contracts.py`:

```python
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
```

- [ ] **Step 4: Run contract tests**

Run:

```powershell
python -m pytest tests/test_contracts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/tnmi/contracts.py tests/test_contracts.py
git commit -m "feat: add milestone one domain contracts"
```

## Task 3: Newspaper Source Configuration

**Files:**
- Create: `configs/sources.newspapers.yaml`
- Create: `src/tnmi/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

from tnmi.config import load_newspaper_sources


def test_load_newspaper_sources_from_yaml(tmp_path: Path):
    config = tmp_path / "sources.yaml"
    config.write_text(
        """
newspapers:
  - name: Example Tamil Daily
    language_hint: ta
    priority: 1
    active: true
    rss_urls:
      - https://example.com/rss
    sitemap_urls: []
    section_urls: []
""",
        encoding="utf-8",
    )

    sources = load_newspaper_sources(config)

    assert len(sources) == 1
    assert sources[0].name == "Example Tamil Daily"
    assert str(sources[0].rss_urls[0]) == "https://example.com/rss"
```

- [ ] **Step 2: Run config test to verify failure**

Run:

```powershell
python -m pytest tests/test_config.py -v
```

Expected: FAIL because `tnmi.config` does not exist.

- [ ] **Step 3: Implement config loader**

Create `src/tnmi/config.py`:

```python
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from tnmi.contracts import NewspaperSource


class Settings(BaseSettings):
    database_url: str = "sqlite:///./mediaintel.db"
    openai_api_key: str | None = None
    openai_model_item_classifier: str = "gpt-5.4-mini"
    openai_model_report: str = "gpt-5.5"
    news_source_config: Path = Path("configs/sources.newspapers.yaml")
    report_output_dir: Path = Path("reports/generated")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def load_newspaper_sources(path: str | Path) -> list[NewspaperSource]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [NewspaperSource.model_validate(item) for item in data.get("newspapers", [])]
```

Create `configs/sources.newspapers.yaml`:

```yaml
newspapers:
  - name: The Hindu Tamil Thisai
    language_hint: ta
    priority: 1
    active: true
    rss_urls:
      - https://www.hindutamil.in/rss
    sitemap_urls: []
    section_urls: []
    legal_notes: Public RSS source; verify production access and terms before launch.

  - name: Dinamalar
    language_hint: ta
    priority: 1
    active: true
    rss_urls: []
    sitemap_urls: []
    section_urls:
      - https://www.dinamalar.com/
    legal_notes: Public website source; configure approved RSS/sitemap if available.

  - name: Dinamani
    language_hint: ta
    priority: 1
    active: true
    rss_urls: []
    sitemap_urls: []
    section_urls:
      - https://www.dinamani.com/
    legal_notes: Public website source; configure approved RSS/sitemap if available.
```

- [ ] **Step 4: Run config tests**

Run:

```powershell
python -m pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add configs/sources.newspapers.yaml src/tnmi/config.py tests/test_config.py
git commit -m "feat: add newspaper source configuration"
```

## Task 4: Tamil-First Language Detection

**Files:**
- Create: `src/tnmi/language.py`
- Create: `tests/test_language.py`

- [ ] **Step 1: Write failing language tests**

Create `tests/test_language.py`:

```python
from tnmi.language import detect_language


def test_detects_tamil():
    assert detect_language("தமிழக அரசு இன்று புதிய திட்டம் அறிவித்தது") == "ta"


def test_detects_english():
    assert detect_language("The Tamil Nadu government announced a new scheme") == "en"


def test_detects_mixed_tamil_english():
    assert detect_language("தமிழக அரசு new scheme announce செய்தது") == "ta-en-mixed"


def test_detects_unknown_for_empty_text():
    assert detect_language("   ") == "unknown"
```

- [ ] **Step 2: Run language tests to verify failure**

Run:

```powershell
python -m pytest tests/test_language.py -v
```

Expected: FAIL because `tnmi.language` does not exist.

- [ ] **Step 3: Implement language detector**

Create `src/tnmi/language.py`:

```python
from __future__ import annotations

import re

TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "unknown"

    tamil_count = len(TAMIL_RE.findall(stripped))
    latin_count = len(LATIN_RE.findall(stripped))

    if tamil_count and latin_count:
        return "ta-en-mixed"
    if tamil_count:
        return "ta"
    if latin_count:
        return "en"
    return "unknown"
```

- [ ] **Step 4: Run language tests**

Run:

```powershell
python -m pytest tests/test_language.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/tnmi/language.py tests/test_language.py
git commit -m "feat: add tamil-first language detection"
```

## Task 5: Storage Schema and Repositories

**Files:**
- Create: `db/schema.sql`
- Create: `src/tnmi/storage.py`
- Create: `tests/test_storage.py`

- [ ] **Step 1: Write failing storage tests**

Create `tests/test_storage.py`:

```python
from datetime import datetime, timezone

from tnmi.contracts import AIAnalysis, GovernmentRelevance, NormalizedItem, SourceType, Stance
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item


def test_save_raw_item_is_idempotent_by_content_hash(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/a",
        published_at=datetime(2026, 5, 21, tzinfo=timezone.utc),
        language="ta",
        title="Title",
        raw_text_original="தமிழக அரசு செய்தி",
        clean_text_original="தமிழக அரசு செய்தி",
    )

    with session_factory() as session:
        first = save_raw_item(session, item)
        second = save_raw_item(session, item)
        session.commit()

    assert first.id == second.id


def test_save_ai_analysis_for_raw_item(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/a",
        published_at=None,
        language="en",
        title="Title",
        raw_text_original="Government announced a scheme",
        clean_text_original="Government announced a scheme",
    )
    analysis = AIAnalysis(
        government_relevance=GovernmentRelevance.HIGH,
        stance_toward_government=Stance.POSITIVE,
        sentiment="positive",
        target="Tamil Nadu Government",
        department="welfare",
        district="unknown",
        scheme=None,
        topic="scheme",
        issue_category="welfare",
        severity="low",
        summary_original="Positive item.",
        summary_english="Positive item.",
        positive_points=["positive"],
        negative_points=[],
        evidence_quotes_original=["announced a scheme"],
        evidence_quotes_english=["announced a scheme"],
        confidence=0.9,
        needs_human_review=False,
    )

    with session_factory() as session:
        raw = save_raw_item(session, item)
        saved = save_ai_analysis(session, raw.id, analysis, model_name="mock", prompt_version="v1")
        session.commit()

    assert saved.raw_item_id == raw.id
```

- [ ] **Step 2: Run storage tests to verify failure**

Run:

```powershell
python -m pytest tests/test_storage.py -v
```

Expected: FAIL because `tnmi.storage` does not exist.

- [ ] **Step 3: Implement storage layer**

Create `src/tnmi/storage.py` with SQLAlchemy ORM models for `raw_items` and `ai_analysis`, plus repository functions:

```python
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, String, Text, UniqueConstraint, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from tnmi.contracts import AIAnalysis, NormalizedItem


class Base(DeclarativeBase):
    pass


class RawItemRecord(Base):
    __tablename__ = "raw_items"
    __table_args__ = (UniqueConstraint("content_hash", name="uq_raw_items_content_hash"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    source_type: Mapped[str] = mapped_column(String(64), index=True)
    source_name: Mapped[str] = mapped_column(String(255), index=True)
    source_url: Mapped[str] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    language: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_text_original: Mapped[str] = mapped_column(Text)
    clean_text_original: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)


class AIAnalysisRecord(Base):
    __tablename__ = "ai_analysis"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    raw_item_id: Mapped[int] = mapped_column(index=True)
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
    positive_points: Mapped[list[str]] = mapped_column(JSON, default=list)
    negative_points: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence_quotes_original: Mapped[list[str]] = mapped_column(JSON, default=list)
    evidence_quotes_english: Mapped[list[str]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float)
    needs_human_review: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


def create_session_factory(database_url: str) -> sessionmaker[Session]:
    engine = create_engine(database_url, future=True)
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
    session.add(record)
    session.flush()
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
        sentiment=analysis.sentiment,
        target=analysis.target,
        department=analysis.department,
        district=analysis.district,
        scheme=analysis.scheme,
        topic=analysis.topic,
        issue_category=analysis.issue_category,
        severity=analysis.severity,
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
```

Create `db/schema.sql` matching these tables for PostgreSQL. Use `JSONB` for JSON columns and `TIMESTAMPTZ` for timestamp columns.

- [ ] **Step 4: Run storage tests**

Run:

```powershell
python -m pytest tests/test_storage.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add db/schema.sql src/tnmi/storage.py tests/test_storage.py
git commit -m "feat: add storage schema and repositories"
```

## Task 6: News Feed and Article Extraction

**Files:**
- Create: `src/tnmi/news.py`
- Create: `tests/fixtures/sample_feed.xml`
- Create: `tests/fixtures/sample_article.html`
- Create: `tests/test_news.py`

- [ ] **Step 1: Write fixtures and failing news tests**

Create `tests/fixtures/sample_feed.xml`:

```xml
<?xml version="1.0" encoding="UTF-8" ?>
<rss version="2.0">
  <channel>
    <title>Example Tamil Daily</title>
    <item>
      <title>தமிழக அரசு புதிய திட்டம்</title>
      <link>https://example.com/news/tamil-nadu-scheme</link>
      <pubDate>Thu, 21 May 2026 06:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>
```

Create `tests/fixtures/sample_article.html`:

```html
<!doctype html>
<html>
  <head><title>தமிழக அரசு புதிய திட்டம்</title></head>
  <body>
    <article>
      <h1>தமிழக அரசு புதிய திட்டம்</h1>
      <p>தமிழக அரசு இன்று மக்களுக்கு புதிய நலத்திட்டத்தை அறிவித்தது.</p>
    </article>
  </body>
</html>
```

Create `tests/test_news.py`:

```python
from pathlib import Path

from tnmi.contracts import NewspaperSource
from tnmi.news import extract_article_text, parse_feed_entries


def test_parse_feed_entries_reads_rss_fixture():
    feed_xml = Path("tests/fixtures/sample_feed.xml").read_text(encoding="utf-8")
    source = NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])

    entries = parse_feed_entries(source, feed_xml)

    assert len(entries) == 1
    assert entries[0].url == "https://example.com/news/tamil-nadu-scheme"
    assert "தமிழக அரசு" in entries[0].title


def test_extract_article_text_from_html_fixture():
    html = Path("tests/fixtures/sample_article.html").read_text(encoding="utf-8")

    article = extract_article_text("https://example.com/news/tamil-nadu-scheme", html)

    assert article.title == "தமிழக அரசு புதிய திட்டம்"
    assert "நலத்திட்டத்தை" in article.clean_text
```

- [ ] **Step 2: Run news tests to verify failure**

Run:

```powershell
python -m pytest tests/test_news.py -v
```

Expected: FAIL because `tnmi.news` does not exist.

- [ ] **Step 3: Implement news module**

Create `src/tnmi/news.py`:

```python
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
```

- [ ] **Step 4: Run news tests**

Run:

```powershell
python -m pytest tests/test_news.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/tnmi/news.py tests/fixtures/sample_feed.xml tests/fixtures/sample_article.html tests/test_news.py
git commit -m "feat: add newspaper feed and article extraction"
```

## Task 7: AI Provider Abstraction

**Files:**
- Create: `src/tnmi/ai.py`
- Create: `tests/test_ai.py`

- [ ] **Step 1: Write failing AI tests**

Create `tests/test_ai.py`:

```python
from tnmi.ai import MockAIAnalyzer
from tnmi.contracts import GovernmentRelevance, NormalizedItem, SourceType, Stance


def test_mock_ai_analyzer_returns_positive_for_scheme_news():
    item = NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example",
        source_url="https://example.com/a",
        language="ta",
        title="தமிழக அரசு புதிய திட்டம்",
        raw_text_original="தமிழக அரசு இன்று புதிய நலத்திட்டத்தை அறிவித்தது.",
        clean_text_original="தமிழக அரசு இன்று புதிய நலத்திட்டத்தை அறிவித்தது.",
    )

    analysis = MockAIAnalyzer().analyze(item)

    assert analysis.government_relevance == GovernmentRelevance.HIGH
    assert analysis.stance_toward_government == Stance.POSITIVE
    assert analysis.confidence >= 0.5
```

- [ ] **Step 2: Run AI tests to verify failure**

Run:

```powershell
python -m pytest tests/test_ai.py -v
```

Expected: FAIL because `tnmi.ai` does not exist.

- [ ] **Step 3: Implement AI provider interface and mock**

Create `src/tnmi/ai.py`:

```python
from __future__ import annotations

import json
from typing import Protocol

from openai import OpenAI

from tnmi.contracts import AIAnalysis, GovernmentRelevance, NormalizedItem, Stance


PROMPT_VERSION = "newspaper-stance-v1"


class AIAnalyzer(Protocol):
    model_name: str

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        ...


class MockAIAnalyzer:
    model_name = "mock"

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        text = f"{item.title or ''}\n{item.clean_text_original}".lower()
        relevance = GovernmentRelevance.HIGH if "அரசு" in text or "government" in text else GovernmentRelevance.NONE
        stance = Stance.POSITIVE if "திட்ட" in text or "scheme" in text else Stance.NEUTRAL
        return AIAnalysis(
            government_relevance=relevance,
            stance_toward_government=stance,
            sentiment="positive" if stance == Stance.POSITIVE else "neutral",
            target="Tamil Nadu Government" if relevance != GovernmentRelevance.NONE else "none",
            department="unknown",
            district="unknown",
            scheme=None,
            topic=item.title or "news item",
            issue_category="welfare" if stance == Stance.POSITIVE else "unknown",
            severity="low",
            summary_original=(item.clean_text_original[:180] or item.title or "").strip(),
            summary_english="Mock analysis summary.",
            positive_points=["Mentions a government scheme."] if stance == Stance.POSITIVE else [],
            negative_points=[],
            evidence_quotes_original=[item.clean_text_original[:120]],
            evidence_quotes_english=["Mock evidence translation."],
            confidence=0.75,
            needs_human_review=False,
        )


class OpenAIAnalyzer:
    def __init__(self, api_key: str, model_name: str = "gpt-5.4-mini") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        prompt = build_classification_prompt(item)
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )
        payload = json.loads(response.output_text)
        return AIAnalysis.model_validate(payload)


def build_classification_prompt(item: NormalizedItem) -> str:
    return f"""
You are analyzing public media content about the Tamil Nadu Government.

Analyze Tamil, English, Tanglish, and mixed-script content. Preserve original meaning.
Do not over-translate slogans, sarcasm, allegations, or local political phrases.
Classify stance toward the Tamil Nadu Government only from evidence in the text.
If the content is not about the Tamil Nadu Government, government_relevance must be "none".
If a claim is an allegation, needs_human_review must be true.

Return JSON only with this schema:
{{
  "government_relevance": "high|medium|low|none",
  "stance_toward_government": "positive|negative|neutral|mixed",
  "sentiment": "positive|negative|neutral",
  "target": "...",
  "department": "...",
  "district": "...",
  "scheme": null,
  "topic": "...",
  "issue_category": "...",
  "severity": "low|medium|high|critical",
  "summary_original": "...",
  "summary_english": "...",
  "positive_points": [],
  "negative_points": [],
  "evidence_quotes_original": [],
  "evidence_quotes_english": [],
  "confidence": 0.0,
  "needs_human_review": true
}}

Source: {item.source_name}
URL: {item.source_url}
Language: {item.language}
Title: {item.title or ""}
Text:
{item.clean_text_original}
""".strip()
```

- [ ] **Step 4: Run AI tests**

Run:

```powershell
python -m pytest tests/test_ai.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/tnmi/ai.py tests/test_ai.py
git commit -m "feat: add AI analyzer abstraction"
```

## Task 8: Daily Newspaper Pipeline Orchestrator

**Files:**
- Create: `src/tnmi/pipeline.py`
- Create: `tests/test_pipeline.py`

- [ ] **Step 1: Write failing pipeline test**

Create `tests/test_pipeline.py`:

```python
from pathlib import Path

from tnmi.ai import MockAIAnalyzer
from tnmi.contracts import NewspaperSource
from tnmi.pipeline import DailyNewsPipeline, InMemoryNewsClient
from tnmi.storage import create_session_factory, init_db


def test_daily_news_pipeline_processes_feed_and_article(tmp_path):
    feed_xml = Path("tests/fixtures/sample_feed.xml").read_text(encoding="utf-8")
    article_html = Path("tests/fixtures/sample_article.html").read_text(encoding="utf-8")
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(session_factory)
    client = InMemoryNewsClient(
        feeds={"https://example.com/rss": feed_xml},
        articles={"https://example.com/news/tamil-nadu-scheme": article_html},
    )
    pipeline = DailyNewsPipeline(
        session_factory=session_factory,
        news_client=client,
        analyzer=MockAIAnalyzer(),
    )

    result = pipeline.run([NewspaperSource(name="Example Tamil Daily", rss_urls=["https://example.com/rss"])])

    assert result.items_seen == 1
    assert result.items_saved == 1
    assert result.analyses_saved == 1
```

- [ ] **Step 2: Run pipeline test to verify failure**

Run:

```powershell
python -m pytest tests/test_pipeline.py -v
```

Expected: FAIL because `tnmi.pipeline` does not exist.

- [ ] **Step 3: Implement pipeline orchestrator**

Create `src/tnmi/pipeline.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests
from sqlalchemy.orm import Session, sessionmaker

from tnmi.ai import AIAnalyzer, PROMPT_VERSION
from tnmi.contracts import NewspaperSource, NormalizedItem, SourceType
from tnmi.language import detect_language
from tnmi.news import extract_article_text, parse_feed_entries
from tnmi.storage import save_ai_analysis, save_raw_item


class NewsClient(Protocol):
    def fetch_text(self, url: str) -> str:
        ...


class RequestsNewsClient:
    def fetch_text(self, url: str) -> str:
        response = requests.get(url, timeout=30, headers={"User-Agent": "tn-media-intelligence/0.1"})
        response.raise_for_status()
        return response.text


class InMemoryNewsClient:
    def __init__(self, *, feeds: dict[str, str], articles: dict[str, str]) -> None:
        self.feeds = feeds
        self.articles = articles

    def fetch_text(self, url: str) -> str:
        if url in self.feeds:
            return self.feeds[url]
        return self.articles[url]


@dataclass(frozen=True)
class PipelineResult:
    items_seen: int = 0
    items_saved: int = 0
    analyses_saved: int = 0
    failures: int = 0


class DailyNewsPipeline:
    def __init__(
        self,
        *,
        session_factory: sessionmaker[Session],
        news_client: NewsClient,
        analyzer: AIAnalyzer,
    ) -> None:
        self.session_factory = session_factory
        self.news_client = news_client
        self.analyzer = analyzer

    def run(self, sources: list[NewspaperSource]) -> PipelineResult:
        items_seen = 0
        items_saved = 0
        analyses_saved = 0
        failures = 0

        with self.session_factory() as session:
            for source in sources:
                if not source.active:
                    continue
                for rss_url in source.rss_urls:
                    try:
                        feed_xml = self.news_client.fetch_text(str(rss_url))
                        entries = parse_feed_entries(source, feed_xml)
                    except Exception:
                        failures += 1
                        continue

                    for entry in entries:
                        items_seen += 1
                        try:
                            html = self.news_client.fetch_text(entry.url)
                            article = extract_article_text(entry.url, html)
                            text = article.clean_text.strip()
                            if not text:
                                failures += 1
                                continue
                            item = NormalizedItem(
                                source_type=SourceType.NEWS,
                                source_name=source.name,
                                source_url=entry.url,
                                published_at=entry.published_at,
                                language=detect_language(text),
                                title=article.title or entry.title,
                                raw_text_original=article.raw_text,
                                clean_text_original=text,
                                metadata=article.metadata,
                            )
                            raw = save_raw_item(session, item)
                            items_saved += 1
                            analysis = self.analyzer.analyze(item)
                            save_ai_analysis(
                                session,
                                raw.id,
                                analysis,
                                model_name=self.analyzer.model_name,
                                prompt_version=PROMPT_VERSION,
                            )
                            analyses_saved += 1
                        except Exception:
                            failures += 1

            session.commit()

        return PipelineResult(
            items_seen=items_seen,
            items_saved=items_saved,
            analyses_saved=analyses_saved,
            failures=failures,
        )
```

- [ ] **Step 4: Run pipeline test**

Run:

```powershell
python -m pytest tests/test_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/tnmi/pipeline.py tests/test_pipeline.py
git commit -m "feat: add daily newspaper pipeline"
```

## Task 9: Daily Newspaper Report

**Files:**
- Create: `src/tnmi/reports.py`
- Create: `tests/test_reports.py`

- [ ] **Step 1: Write failing report test**

Create `tests/test_reports.py`:

```python
from datetime import date

from tnmi.reports import render_daily_news_markdown


def test_render_daily_news_markdown_includes_stance_counts():
    markdown = render_daily_news_markdown(
        report_date=date(2026, 5, 21),
        stance_counts={"positive": 2, "negative": 1, "neutral": 3, "mixed": 1},
        top_items=[
            {
                "source_name": "Example Tamil Daily",
                "title": "தமிழக அரசு புதிய திட்டம்",
                "stance": "positive",
                "summary": "சாதகமான செய்தி.",
                "url": "https://example.com/a",
            }
        ],
    )

    assert "# Daily Newspaper Intelligence Report - 2026-05-21" in markdown
    assert "- Positive: 2" in markdown
    assert "தமிழக அரசு புதிய திட்டம்" in markdown
```

- [ ] **Step 2: Run report test to verify failure**

Run:

```powershell
python -m pytest tests/test_reports.py -v
```

Expected: FAIL because `tnmi.reports` does not exist.

- [ ] **Step 3: Implement report renderer**

Create `src/tnmi/reports.py`:

```python
from __future__ import annotations

from datetime import date
from pathlib import Path


def render_daily_news_markdown(
    *,
    report_date: date,
    stance_counts: dict[str, int],
    top_items: list[dict[str, str]],
) -> str:
    lines = [
        f"# Daily Newspaper Intelligence Report - {report_date.isoformat()}",
        "",
        "## Stance Split",
        "",
        f"- Positive: {stance_counts.get('positive', 0)}",
        f"- Negative: {stance_counts.get('negative', 0)}",
        f"- Neutral: {stance_counts.get('neutral', 0)}",
        f"- Mixed: {stance_counts.get('mixed', 0)}",
        "",
        "## Top Items",
        "",
    ]
    for item in top_items:
        lines.extend(
            [
                f"### {item['title']}",
                "",
                f"- Source: {item['source_name']}",
                f"- Stance: {item['stance']}",
                f"- Summary: {item['summary']}",
                f"- URL: {item['url']}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def write_report(markdown: str, output_dir: Path, filename: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    path.write_text(markdown, encoding="utf-8")
    return path
```

- [ ] **Step 4: Run report tests**

Run:

```powershell
python -m pytest tests/test_reports.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src/tnmi/reports.py tests/test_reports.py
git commit -m "feat: add daily newspaper report renderer"
```

## Task 10: CLI and Airflow-Compatible Entry Points

**Files:**
- Create: `pipelines/run_daily_news.py`
- Create: `pipelines/dags/daily_news_intelligence.py`

- [ ] **Step 1: Implement CLI runner**

Create `pipelines/run_daily_news.py`:

```python
from __future__ import annotations

import argparse
from datetime import date

from tnmi.ai import MockAIAnalyzer, OpenAIAnalyzer
from tnmi.config import Settings, load_newspaper_sources
from tnmi.pipeline import DailyNewsPipeline, RequestsNewsClient
from tnmi.storage import create_session_factory, init_db


def build_analyzer(settings: Settings):
    if settings.openai_api_key:
        return OpenAIAnalyzer(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model_item_classifier,
        )
    return MockAIAnalyzer()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    settings = Settings()
    sources = load_newspaper_sources(settings.news_source_config)
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    pipeline = DailyNewsPipeline(
        session_factory=session_factory,
        news_client=RequestsNewsClient(),
        analyzer=build_analyzer(settings),
    )
    result = pipeline.run(sources)
    print(f"date={args.date} items_seen={result.items_seen} items_saved={result.items_saved} analyses_saved={result.analyses_saved} failures={result.failures}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Implement Airflow wrapper**

Create `pipelines/dags/daily_news_intelligence.py`:

```python
from __future__ import annotations

from datetime import datetime

try:
    from airflow.decorators import dag, task
except ImportError:
    dag = None
    task = None


if dag and task:

    @dag(
        dag_id="daily_news_intelligence",
        start_date=datetime(2026, 5, 1),
        schedule="0 6 * * *",
        catchup=False,
        tags=["media-intelligence", "news"],
    )
    def daily_news_intelligence():
        @task
        def run_daily_news_pipeline():
            from pipelines.run_daily_news import main

            main()

        run_daily_news_pipeline()

    daily_news_intelligence()
```

- [ ] **Step 3: Run CLI help**

Run:

```powershell
python pipelines/run_daily_news.py --help
```

Expected: command prints `--date` option and exits with code 0.

- [ ] **Step 4: Commit**

```powershell
git add pipelines/run_daily_news.py pipelines/dags/daily_news_intelligence.py
git commit -m "feat: add newspaper pipeline entrypoints"
```

## Task 11: Basic API

**Files:**
- Create: `apps/api/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: Write failing API tests**

Create `tests/test_api.py`:

```python
from fastapi.testclient import TestClient

from apps.api.main import app


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint():
    client = TestClient(app)

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["version"] == "0.1.0"
```

- [ ] **Step 2: Run API tests to verify failure**

Run:

```powershell
python -m pytest tests/test_api.py -v
```

Expected: FAIL because `apps.api.main` does not exist.

- [ ] **Step 3: Implement FastAPI app**

Create `apps/api/main.py`:

```python
from __future__ import annotations

from fastapi import FastAPI

from tnmi import __version__

app = FastAPI(title="Tamil Nadu Media Intelligence API", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": __version__}
```

- [ ] **Step 4: Run API tests**

Run:

```powershell
python -m pytest tests/test_api.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add apps/api/main.py tests/test_api.py
git commit -m "feat: add basic inspection API"
```

## Task 12: Local Production-Shaped Infrastructure

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Add Docker Compose**

Create `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: mediaintel
      POSTGRES_USER: mediaintel
      POSTGRES_PASSWORD: mediaintel
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

  redis:
    image: redis:7
    ports:
      - "6379:6379"

  opensearch:
    image: opensearchproject/opensearch:latest
    environment:
      discovery.type: single-node
      plugins.security.disabled: "true"
      OPENSEARCH_INITIAL_ADMIN_PASSWORD: "ChangeMeStrong123!"
      DISABLE_INSTALL_DEMO_CONFIG: "true"
    ports:
      - "9200:9200"
    volumes:
      - opensearch-data:/usr/share/opensearch/data

  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: mediaintel
      MINIO_ROOT_PASSWORD: mediaintel123
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio-data:/data

volumes:
  postgres-data:
  opensearch-data:
  minio-data:
```

- [ ] **Step 2: Validate Compose syntax**

Run:

```powershell
docker compose config
```

Expected: Compose renders service configuration without errors.

- [ ] **Step 3: Commit**

```powershell
git add docker-compose.yml
git commit -m "chore: add local infrastructure compose"
```

## Task 13: Full Verification

**Files:**
- Modify only if failures reveal small fixes in files above.

- [ ] **Step 1: Run all tests**

Run:

```powershell
python -m pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run API locally**

Run:

```powershell
python -m uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Expected: server starts and `/health` returns `{"status":"ok"}`.

- [ ] **Step 3: Run newspaper CLI with fixtures only if configured for real URLs is not desired**

Run:

```powershell
python pipelines/run_daily_news.py --date 2026-05-21
```

Expected with live config: command attempts configured sources and prints a result line. If external sources fail, failures are counted and the process exits normally.

- [ ] **Step 4: Commit verification fixes**

If any small fixes were needed:

```powershell
git add .
git commit -m "fix: stabilize newspaper pipeline verification"
```

## Self-Review

Spec coverage:

- Newspaper source registry: Task 3.
- Article ingestion/extraction: Task 6 and Task 8.
- Tamil-first language handling: Task 4.
- AI classification abstraction: Task 7.
- Storage and dedupe: Task 5.
- Daily report: Task 9.
- API inspection: Task 11.
- Production-shaped local infra: Task 12.
- Repeatable verification: Task 13.

Known gaps intentionally left for later milestone plans:

- X/Twitter ingestion.
- Instagram/video ingestion.
- OCR/transcription.
- Dashboard UI.
- OpenSearch indexing.
- turbovec semantic indexing.
- PDF/email delivery.
- Kubernetes production deployment.

Red flag scan:

- No unresolved implementation gaps are intended.

Type consistency:

- `NormalizedItem`, `AIAnalysis`, and enum names are introduced in Task 2 and reused consistently in later tasks.
- Storage functions introduced in Task 5 are used by the pipeline in Task 8.
- `MockAIAnalyzer` and `OpenAIAnalyzer` introduced in Task 7 are used by the CLI in Task 10.
