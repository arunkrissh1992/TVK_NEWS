# Tamil Nadu Public Media Intelligence Platform - Production Design

Date: 2026-05-21
Status: Approved direction, pending user review of written spec
Workspace: `TVK_NEWS`

## 1. Product Summary

Build a confidential, production-grade, Tamil-first public media intelligence platform for the Tamil Nadu Government. The platform ingests public media from newspapers, X/Twitter, Instagram/video sources through approved access, extracts text/audio/video evidence, classifies how content portrays the government, and produces evidence-backed daily reports.

The product is public media intelligence, not citizen surveillance. It analyzes public articles, posts, captions, transcripts, and video evidence. It must not collect private messages, bypass platform access controls, maintain citizen blacklists, or use unauthorized scraping as a production dependency.

## 2. Strategic Architecture Principle

Design for a long-lived intelligence platform, not a throwaway monitoring script.

The system should support:

- Tamil, English, and Tamil-English mixed content.
- Text, audio, images, and video-derived evidence.
- Daily scheduled reports first, real-time alerts later.
- Cloud, private VPC, and on-prem deployment patterns.
- Model/provider changes without rewriting ingestion and reporting.
- Evidence preservation and auditability for every AI-generated conclusion.
- Gradual source expansion from newspapers to X, Instagram, video, YouTube, press releases, and future public data providers.

## 3. Goals

### 3.1 Business Goals

- Produce a daily intelligence report on how public media portrays the Tamil Nadu Government.
- Identify positive narratives, negative narratives, mixed coverage, neutral coverage, and emerging issues.
- Attribute findings to evidence: source link, original Tamil/English quote, transcript, OCR text, or media frame.
- Help departments understand issues by district, department, scheme, topic, source, and urgency.
- Provide human review before escalation of allegations and sensitive claims.

### 3.2 Engineering Goals

- Build one private product repo with modular services.
- Preserve raw evidence before AI processing.
- Normalize all source types into one common data model.
- Use AI through provider abstractions.
- Make every pipeline idempotent, retryable, observable, and auditable.
- Keep source connectors independent so platform access changes do not break core analysis.

### 3.3 First Implementation Goal

Milestone 1 is daily newspaper processing:

- Configure 15 newspapers.
- Ingest RSS/sitemap/manual article URLs.
- Extract clean article text.
- Detect language.
- Run Tamil-first government relevance and stance classification.
- Store analysis.
- Generate a daily newspaper-only report.

## 4. Non-Goals

- No private-message collection.
- No unauthorized bypassing of X, Instagram, or newspaper access controls.
- No facial recognition or identity profiling.
- No citizen blacklists.
- No fully automated punitive action from AI analysis.
- No claim that AI classification is ground truth without evidence and confidence.

## 5. Product Modules

### 5.1 Source Registry

Stores all configured sources and their access method.

Source types:

- `news`
- `x`
- `instagram`
- `video`
- `youtube`
- `press_release`
- `manual_upload`
- `provider_export`

Each source stores:

- name
- source type
- handle, feed URL, sitemap URL, or provider ID
- language expectation
- priority
- active status
- access method
- legal/access notes
- rate limit policy
- last cursor/checkpoint

### 5.2 Ingestion Layer

Responsible only for fetching public/source-authorized content and creating raw records.

Ingestion connectors:

- News RSS connector.
- News sitemap connector.
- Manual URL connector.
- X official API connector.
- Instagram approved API/provider/export connector.
- Manual media upload connector for demo and review workflows.

Ingestion must be:

- idempotent
- cursor-aware
- rate-limit aware
- retryable
- source-attributed
- legally bounded

### 5.3 Evidence Lake

Stores raw source material before cleaning or summarization.

Examples:

- raw HTML
- original article text
- original captions
- transcripts
- OCR outputs
- sampled video frames
- thumbnails
- source JSON payloads
- generated report PDFs

Storage target:

- S3-compatible object storage in production.
- MinIO in local/dev environments.

### 5.4 Extraction Layer

Transforms raw inputs into normalized text and metadata.

News extraction:

- `feedparser` for RSS/Atom.
- `trafilatura` for clean article text, title, author/date metadata, and page extraction.

Media extraction:

- audio extraction from video using ffmpeg.
- transcription through OpenAI speech-to-text or self-hosted `faster-whisper`.
- frame sampling through OpenCV.
- OCR through PaddleOCR using Tamil and English passes.

Language processing:

- language detection: Tamil, English, mixed, unknown.
- preserve original text.
- optional English translation for reporting.
- optional Tamil summary for local teams.

### 5.5 AI Enrichment Layer

Adds structured analysis to every normalized item.

Primary item-level tasks:

- government relevance
- stance toward Tamil Nadu Government
- sentiment
- target entity
- department
- district
- scheme/topic
- issue category
- summary in original language
- English summary
- positive points
- negative points
- evidence quotes
- confidence
- human review flag

Important distinction:

Sentiment and stance are different. A sad flood article may have negative sentiment but positive government stance if it praises relief work.

Recommended models:

- High-volume item classification: `gpt-5.4-mini`.
- Final report reasoning and synthesis: `gpt-5.5`.
- Transcription: `gpt-4o-transcribe`, `gpt-4o-mini-transcribe`, `gpt-4o-transcribe-diarize`, or self-hosted `faster-whisper`.
- OCR: PaddleOCR with Tamil and English passes.
- Translation fallback: IndicTrans2 or model-based translation.

Model access must be abstracted through internal interfaces so future models can be added without changing pipeline logic.

### 5.6 Taxonomy Layer

Controlled dictionaries used by AI and rules.

Taxonomies:

- Tamil Nadu departments.
- Districts.
- Government schemes.
- Ministers and senior offices.
- Local bodies.
- Topic categories.
- Issue severity levels.
- Source trust/priority tiers.
- Tamil, English, and transliterated aliases.

Examples:

- `முதல்வர்`, `CM`, `Chief Minister`.
- `சென்னை`, `Chennai`, `Madras`.
- `மின்சாரம்`, `electricity`, `TANGEDCO`.

### 5.7 Search and Retrieval

Use hybrid retrieval.

PostgreSQL:

- source of truth.
- transactional records.
- normalized item and analysis tables.

OpenSearch:

- keyword search.
- dashboard filters.
- full-text retrieval.
- aggregations by source, district, department, date, stance.

Turbovec:

- local compressed semantic vector index.
- narrative similarity.
- duplicate story grouping.
- similar allegation detection.
- private RAG over historical items and reports.

Turbovec must not replace PostgreSQL or OpenSearch. It should be integrated behind a retrieval interface so it can be swapped with another vector backend later.

### 5.8 Human Review Queue

Human review is required for:

- allegations.
- low-confidence AI results.
- high-virality negative narratives.
- claims involving corruption, law and order, communal/caste sensitivity, public safety, or individual accusations.
- conflicting model outputs.

Reviewers can:

- approve AI analysis.
- edit stance/department/district/category.
- add reviewer notes.
- mark false positive.
- mark escalation needed.
- redact unnecessary personal data.

All review actions must be audited.

### 5.9 Reporting Layer

Report types:

- daily executive report.
- daily department report.
- district report.
- source-wise report.
- issue escalation memo.
- weekly narrative trend report.

Daily report sections:

1. Executive summary.
2. Overall positive/negative/mixed/neutral split.
3. Top positive narratives.
4. Top negative narratives.
5. Emerging issues.
6. Department-wise breakdown.
7. District-wise breakdown.
8. Source comparison: newspapers vs X vs Instagram/video.
9. Viral/high-engagement items.
10. Sensitive allegations needing human review.
11. Evidence links and original snippets.
12. English translations where needed.
13. Data coverage gaps.
14. Model confidence and limitations.

Report output:

- PDF.
- secure dashboard.
- email delivery to approved recipients.
- API endpoint for internal systems.

### 5.10 Dashboard

Dashboard users:

- administrator
- analyst
- reviewer
- department viewer
- report recipient

Core views:

- source registry
- ingestion health
- raw item search
- AI analysis results
- review queue
- report builder
- source coverage dashboard
- district/department analytics
- narrative clusters
- audit logs

Dashboard style should be operational and dense, not marketing-like.

## 6. Data Model

### 6.1 Core Tables

#### `sources`

- `id`
- `source_type`
- `name`
- `handle_or_url`
- `access_method`
- `language_hint`
- `priority`
- `active`
- `last_cursor`
- `legal_notes`
- `created_at`
- `updated_at`

#### `raw_items`

- `id`
- `source_id`
- `source_type`
- `source_url`
- `external_id`
- `published_at`
- `ingested_at`
- `title`
- `raw_text_original`
- `clean_text_original`
- `language`
- `metadata`
- `raw_object_url`
- `content_hash`
- `created_at`

#### `media_assets`

- `id`
- `raw_item_id`
- `asset_type`
- `object_url`
- `transcript_original`
- `ocr_text_original`
- `sampled_frame_urls`
- `duration_seconds`
- `metadata`
- `created_at`

#### `ai_analysis`

- `id`
- `raw_item_id`
- `analysis_version`
- `model_provider`
- `model_name`
- `prompt_version`
- `government_relevance`
- `stance_toward_government`
- `sentiment`
- `target`
- `department`
- `district`
- `scheme`
- `topic`
- `issue_category`
- `severity`
- `summary_original`
- `summary_english`
- `positive_points`
- `negative_points`
- `evidence_quotes_original`
- `evidence_quotes_english`
- `confidence`
- `needs_human_review`
- `created_at`

#### `review_items`

- `id`
- `raw_item_id`
- `ai_analysis_id`
- `status`
- `assigned_to`
- `reviewer_decision`
- `reviewer_notes`
- `escalation_level`
- `created_at`
- `reviewed_at`

#### `daily_reports`

- `id`
- `report_date`
- `report_type`
- `title`
- `status`
- `summary_json`
- `pdf_object_url`
- `created_by`
- `created_at`

#### `audit_logs`

- `id`
- `actor_id`
- `action`
- `entity_type`
- `entity_id`
- `before_json`
- `after_json`
- `ip_address`
- `created_at`

#### `vector_index_records`

- `id`
- `raw_item_id`
- `embedding_model`
- `embedding_dim`
- `index_backend`
- `index_key`
- `created_at`

## 7. Common Normalized Item Contract

Every source becomes a normalized item:

```json
{
  "source_type": "news | x | instagram | video",
  "source_name": "The Hindu Tamil | Dinamalar | @handle",
  "source_url": "original public URL",
  "published_at": "2026-05-21T08:30:00+05:30",
  "language": "ta | en | ta-en-mixed | unknown",
  "title": "...",
  "raw_text_original": "...",
  "clean_text_original": "...",
  "english_translation": "...optional...",
  "media": {
    "audio_transcript_original": "...",
    "ocr_text_original": "...",
    "sampled_frames": ["s3://bucket/frame1.jpg"]
  },
  "engagement": {
    "likes": 0,
    "shares": 0,
    "comments": 0,
    "views": 0
  }
}
```

AI analysis contract:

```json
{
  "government_relevance": "high | medium | low | none",
  "stance_toward_government": "positive | negative | neutral | mixed",
  "sentiment": "positive | negative | neutral",
  "target": "Tamil Nadu Government | CM | department | scheme | local body",
  "department": "health | education | police | transport | electricity | revenue | unknown",
  "district": "Chennai | Madurai | Coimbatore | unknown",
  "issue_category": "welfare | corruption allegation | infrastructure | law and order | economy | unknown",
  "summary_original": "One-line factual summary in source language when possible",
  "summary_english": "English summary for reporting",
  "positive_points": [],
  "negative_points": [],
  "evidence_quotes_original": [],
  "evidence_quotes_english": [],
  "confidence": 0.82,
  "needs_human_review": true
}
```

## 8. AI Prompt Principles

All classification prompts must include:

- Analyze Tamil, English, Tanglish, and mixed-script content.
- Preserve original meaning.
- Do not over-translate slogans, sarcasm, allegations, or local political phrases.
- Classify stance toward the Tamil Nadu Government only from evidence.
- Do not infer beyond the content.
- If a claim is an allegation, mark `needs_human_review = true`.
- If content is not about the Tamil Nadu Government, use `government_relevance = none`.
- Return strict JSON matching the schema.

## 9. Source-Specific Design

### 9.1 Newspapers

Initial source count:

- 15 top newspapers.

Access methods:

- RSS feeds.
- sitemaps.
- curated section URLs.
- manual URL upload.

Tools:

- `feedparser` for RSS/Atom.
- `trafilatura` for extraction.

Pipeline:

1. Load active news sources.
2. Fetch RSS/sitemap entries.
3. Deduplicate by URL and content hash.
4. Store raw HTML.
5. Extract clean article text.
6. Detect language.
7. Normalize item.
8. Run AI classification.
9. Index in OpenSearch.
10. Add embedding to turbovec when enabled.
11. Include in daily report.

### 9.2 X/Twitter

Access method:

- official X API only.

Production requirements:

- handle registry.
- rate limit tracking.
- cursor storage.
- retry queue.
- source priority tiers.
- quota usage dashboard.

Scale target:

- 10,000 handles in production, subject to X API contract/quota.

### 9.3 Instagram and Video

Production access:

- Meta Content Library/API if eligible and approved.
- licensed data provider.
- government-approved data agreement.
- approved manual uploads for demo and internal review.

Do not rely on unofficial scraping for production.

Video pipeline:

1. Store authorized video object.
2. Extract audio.
3. Transcribe speech.
4. Sample frames.
5. OCR frames.
6. Combine caption, transcript, OCR, metadata.
7. Classify through AI.

### 9.4 Future Sources

The architecture must allow:

- YouTube channels.
- press releases.
- TV broadcast transcripts.
- public grievance dashboards.
- public government response pages.
- fact-checking feeds.
- public blogs and web forums where legally permitted.

## 10. Security and Confidentiality

Required controls:

- private repository.
- private deployment network.
- RBAC authentication.
- per-role permissions.
- encrypted database storage.
- encrypted object storage.
- encrypted backups.
- secrets manager or Vault.
- audit logs for all access and edits.
- least-privilege API keys.
- IP allowlisting for admin dashboard.
- periodic access review.
- data retention policy.
- secure deletion for expired objects.

OpenAI/API data controls:

- Use API data controls appropriate for government use.
- Seek Zero Data Retention or Modified Abuse Monitoring eligibility where required.
- Do not use API features that violate project retention rules.
- Store prompts/responses internally only where required for audit and allowed by policy.

## 11. Privacy and Responsible Use

Rules:

- Analyze public media narratives, not private citizens.
- Avoid unnecessary publication of personal data in reports.
- Use Presidio or equivalent redaction for phone numbers, addresses, personal IDs, and unrelated private details.
- Keep original evidence in restricted storage.
- Show sensitive evidence only to authorized reviewers.
- Require human review for allegations and sensitive claims.
- Do not produce automated action recommendations against individuals.

## 12. Observability and Operations

Every pipeline run must track:

- run ID.
- source count.
- attempted fetches.
- successful items.
- failed items.
- extraction failures.
- AI failures.
- retry count.
- API quota usage.
- latency.
- cost estimate.

Monitoring:

- Airflow DAG status.
- queue depth.
- worker health.
- source freshness.
- report generation status.
- OpenSearch indexing status.
- object storage failures.
- AI API error rates.

Alerts:

- daily report failed.
- high-priority source not ingested.
- API quota exhaustion.
- spike in extraction failures.
- database/search unavailable.
- high negative narrative spike after review threshold.

## 13. Deployment Design

### 13.1 Local Development

- Docker Compose.
- PostgreSQL.
- Redis.
- OpenSearch.
- MinIO.
- FastAPI.
- worker process.
- optional Airflow.

### 13.2 Staging

- same components as production.
- realistic source subset.
- test AI keys.
- synthetic and approved sample data.
- security and load testing.

### 13.3 Production

- Kubernetes or equivalent orchestrator.
- private VPC or on-prem.
- managed PostgreSQL or HA PostgreSQL.
- OpenSearch cluster.
- S3-compatible object storage.
- Airflow deployment.
- autoscaled workers.
- secrets manager.
- centralized logging.
- backup and disaster recovery.

## 14. Repository Structure

```text
tn-media-intelligence/
  apps/
    api/
    dashboard/
    report_generator/
  pipelines/
    dags/
    tasks/
    worker.py
  services/
    ai/
    extraction/
    media/
    news/
    privacy/
    retrieval/
    social/
    storage/
    taxonomy/
  packages/
    common/
    contracts/
  db/
    migrations/
    schema.sql
    seeds/
  configs/
    sources.newspapers.yaml
    sources.x_handles.yaml
    sources.instagram.yaml
    taxonomy.departments.yaml
    taxonomy.districts.yaml
    taxonomy.schemes.yaml
    report_template.yaml
  infra/
    docker/
    k8s/
    terraform/
  tests/
    unit/
    integration/
    fixtures/
  docs/
    architecture/
    runbooks/
    security/
    superpowers/
      specs/
```

## 15. Milestones

### Milestone 1: Newspaper Daily Pipeline

Goal:

Production-shaped pipeline for 15 newspapers.

Deliverables:

- repo skeleton.
- database schema.
- source registry.
- news ingestion connector.
- article extraction service.
- item normalization.
- AI classification service.
- daily newspaper report.
- basic API endpoints for sources/items/reports.
- local Docker Compose.
- tests for extraction, dedupe, classification schema, and reporting.

Acceptance criteria:

- Can ingest configured newspaper sources.
- Can extract article text and metadata.
- Can store raw and clean content.
- Can classify Tamil/English content into relevance and stance.
- Can produce a daily report with evidence links.
- Pipeline is repeatable without duplicate items.

### Milestone 2: Review Dashboard

Deliverables:

- secure login.
- item list/search.
- analysis detail view.
- human review queue.
- report preview.
- audit logs.

### Milestone 3: X/Twitter Pipeline

Deliverables:

- official X API integration.
- handle registry.
- rate limit and quota tracking.
- post normalization.
- stance analysis.
- source and topic analytics.

### Milestone 4: Instagram/Video Pipeline

Deliverables:

- approved provider/API/manual-upload connector.
- video/audio pipeline.
- transcription.
- Tamil/English OCR.
- multimodal evidence classification.

### Milestone 5: Advanced Intelligence

Deliverables:

- OpenSearch dashboards.
- turbovec semantic similarity.
- duplicate/narrative clustering.
- emerging issue detection.
- alert rules.
- weekly trend reports.

### Milestone 6: Production Hardening

Deliverables:

- Kubernetes deployment.
- backup and restore.
- security review.
- access controls.
- data retention automation.
- load tests.
- operations runbooks.

## 16. Testing Strategy

Test levels:

- unit tests for connectors, parsers, schemas, taxonomy matching.
- integration tests for DB, OpenSearch, object storage, AI mock provider.
- pipeline tests for Airflow/Celery tasks.
- golden dataset tests for Tamil stance classification.
- report snapshot tests.
- security tests for RBAC and audit logging.

Tamil evaluation set:

- collect 500 to 1,000 real public samples over time.
- manually label relevance, stance, department, district, issue category.
- use it to measure model and prompt quality.
- maintain examples for sarcasm, transliteration, mixed language, allegations, and praise.

## 17. Key Risks and Mitigations

### Data access risk

Instagram and X access may require paid or approved access.

Mitigation:

- isolate source connectors.
- support provider exports.
- avoid unauthorized scraping.
- build newspaper pipeline first.

### Tamil nuance risk

Models may miss local context, slang, sarcasm, or political phrasing.

Mitigation:

- Tamil-first prompts.
- taxonomy aliases.
- human review.
- evaluation set.
- prompt/model versioning.

### AI hallucination risk

Model may infer beyond evidence.

Mitigation:

- strict JSON.
- evidence quote requirement.
- confidence scoring.
- human review flag.
- report only evidence-backed claims.

### Scale risk

10,000 X handles and 10,000 Instagram/video sources require quotas and compute.

Mitigation:

- priority tiers.
- queues.
- rate limits.
- batch processing.
- autoscaling workers.
- cost dashboards.

### Confidentiality risk

Sensitive reports and raw evidence must stay protected.

Mitigation:

- private network.
- encryption.
- RBAC.
- audit logs.
- secrets manager.
- data retention.

## 18. First Build Plan After Spec Approval

After this spec is approved, create the implementation plan for Milestone 1 only:

1. Initialize private repo structure.
2. Add backend package layout.
3. Add database schema and migrations.
4. Add source config for newspapers.
5. Implement news ingestion.
6. Implement extraction with feedparser/trafilatura.
7. Implement normalized item storage.
8. Implement AI classification interface with mock and OpenAI provider.
9. Implement daily report generation.
10. Add Docker Compose.
11. Add tests and sample fixtures.

## 19. Reference Technologies

- feedparser: https://github.com/kurtmckee/feedparser
- trafilatura: https://github.com/adbar/trafilatura
- tweepy: https://github.com/tweepy/tweepy
- faster-whisper: https://github.com/SYSTRAN/faster-whisper
- OpenCV: https://github.com/opencv/opencv
- PaddleOCR: https://github.com/PaddlePaddle/PaddleOCR
- IndicTrans2: https://github.com/AI4Bharat/IndicTrans2
- Microsoft Presidio: https://github.com/microsoft/presidio
- OpenSearch: https://github.com/opensearch-project/OpenSearch
- turbovec: https://github.com/RyanCodrai/turbovec
- OpenAI data controls: https://developers.openai.com/api/docs/guides/your-data
- OpenAI speech-to-text: https://developers.openai.com/api/docs/guides/speech-to-text
- OpenAI model docs: https://developers.openai.com/api/docs/models
