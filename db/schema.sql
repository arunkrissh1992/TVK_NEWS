CREATE TABLE raw_items (
    id BIGSERIAL PRIMARY KEY,
    source_type VARCHAR(64) NOT NULL,
    source_name VARCHAR(255) NOT NULL,
    source_url TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    language VARCHAR(32) NOT NULL,
    title TEXT,
    raw_text_original TEXT NOT NULL,
    clean_text_original TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash VARCHAR(64) NOT NULL,
    CONSTRAINT uq_raw_items_content_hash UNIQUE (content_hash)
);

CREATE INDEX ix_raw_items_source_type ON raw_items (source_type);
CREATE INDEX ix_raw_items_source_name ON raw_items (source_name);
CREATE INDEX ix_raw_items_language ON raw_items (language);
CREATE INDEX ix_raw_items_content_hash ON raw_items (content_hash);

CREATE TABLE ai_analysis (
    id BIGSERIAL PRIMARY KEY,
    raw_item_id BIGINT NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
    model_name VARCHAR(128) NOT NULL,
    prompt_version VARCHAR(64) NOT NULL,
    government_relevance VARCHAR(32) NOT NULL,
    stance_toward_government VARCHAR(32) NOT NULL,
    tvk_relevance VARCHAR(32) NOT NULL DEFAULT 'none',
    tvk_portrayal VARCHAR(32) NOT NULL DEFAULT 'neutral',
    sentiment VARCHAR(32) NOT NULL,
    target TEXT NOT NULL,
    political_actors JSONB NOT NULL DEFAULT '[]'::jsonb,
    department VARCHAR(128) NOT NULL,
    district VARCHAR(128) NOT NULL,
    scheme VARCHAR(255),
    topic TEXT NOT NULL,
    issue_category VARCHAR(128) NOT NULL,
    people_issue BOOLEAN NOT NULL DEFAULT false,
    public_issue TEXT NOT NULL DEFAULT '',
    severity VARCHAR(64) NOT NULL,
    summary_original TEXT NOT NULL,
    summary_english TEXT NOT NULL,
    party_action TEXT NOT NULL DEFAULT '',
    people_impact TEXT NOT NULL DEFAULT '',
    root_cause TEXT NOT NULL DEFAULT '',
    recommended_step TEXT NOT NULL DEFAULT '',
    action_owner VARCHAR(128) NOT NULL DEFAULT '',
    action_type VARCHAR(64) NOT NULL DEFAULT '',
    action_priority VARCHAR(64) NOT NULL DEFAULT 'low',
    risk_if_ignored TEXT NOT NULL DEFAULT '',
    talking_points JSONB NOT NULL DEFAULT '[]'::jsonb,
    verification_checklist JSONB NOT NULL DEFAULT '[]'::jsonb,
    draft_statement_original TEXT NOT NULL DEFAULT '',
    draft_statement_english TEXT NOT NULL DEFAULT '',
    positive_points JSONB NOT NULL DEFAULT '[]'::jsonb,
    negative_points JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_quotes_original JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_quotes_english JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence DOUBLE PRECISION NOT NULL,
    needs_human_review BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_ai_analysis_raw_model_prompt UNIQUE (raw_item_id, model_name, prompt_version)
);

CREATE INDEX ix_ai_analysis_raw_item_id ON ai_analysis (raw_item_id);
CREATE INDEX ix_ai_analysis_government_relevance ON ai_analysis (government_relevance);
CREATE INDEX ix_ai_analysis_stance_toward_government ON ai_analysis (stance_toward_government);
CREATE INDEX ix_ai_analysis_tvk_relevance ON ai_analysis (tvk_relevance);
CREATE INDEX ix_ai_analysis_tvk_portrayal ON ai_analysis (tvk_portrayal);
CREATE INDEX ix_ai_analysis_people_issue ON ai_analysis (people_issue);
CREATE INDEX ix_ai_analysis_department ON ai_analysis (department);
CREATE INDEX ix_ai_analysis_district ON ai_analysis (district);

CREATE TABLE review_decisions (
    id BIGSERIAL PRIMARY KEY,
    analysis_id BIGINT NOT NULL REFERENCES ai_analysis(id) ON DELETE CASCADE,
    reviewer_name VARCHAR(128) NOT NULL,
    status VARCHAR(32) NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    corrected_stance VARCHAR(32),
    corrected_relevance VARCHAR(32),
    corrected_summary TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_review_decisions_analysis_id ON review_decisions (analysis_id);
CREATE INDEX ix_review_decisions_reviewer_name ON review_decisions (reviewer_name);
CREATE INDEX ix_review_decisions_status ON review_decisions (status);

-- Bronze/Silver/Gold training labels — the spine of the learning flywheel.
CREATE TABLE labeled_examples (
    id BIGSERIAL PRIMARY KEY,
    raw_item_id BIGINT NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
    analysis_id BIGINT REFERENCES ai_analysis(id) ON DELETE SET NULL,
    field VARCHAR(64) NOT NULL,
    value VARCHAR(64) NOT NULL,
    tier VARCHAR(16) NOT NULL,
    provenance VARCHAR(32) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    validator VARCHAR(128) NOT NULL DEFAULT '',
    split_bucket INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_labeled_example_item_field_tier UNIQUE (raw_item_id, field, tier)
);

CREATE INDEX ix_labeled_examples_raw_item_id ON labeled_examples (raw_item_id);
CREATE INDEX ix_labeled_examples_field ON labeled_examples (field);
CREATE INDEX ix_labeled_examples_tier ON labeled_examples (tier);
CREATE INDEX ix_labeled_examples_provenance ON labeled_examples (provenance);
CREATE INDEX ix_labeled_examples_split_bucket ON labeled_examples (split_bucket);

-- Model versions + the promotion gate (is_live flips only when a candidate
-- beats the incumbent on the gold test set).
CREATE TABLE model_registry (
    id BIGSERIAL PRIMARY KEY,
    model_name VARCHAR(128) NOT NULL,
    version VARCHAR(64) NOT NULL,
    kind VARCHAR(32) NOT NULL DEFAULT 'classifier',
    metrics_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    primary_metric DOUBLE PRECISION NOT NULL DEFAULT 0,
    eval_examples INTEGER NOT NULL DEFAULT 0,
    is_live BOOLEAN NOT NULL DEFAULT false,
    artifact_uri TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_model_registry_name_version UNIQUE (model_name, version)
);

CREATE INDEX ix_model_registry_model_name ON model_registry (model_name);
CREATE INDEX ix_model_registry_is_live ON model_registry (is_live);

-- Canonical political objects (people, parties, offices, districts, sources…)
-- — the nodes of the knowledge vault. Free-text analysis surfaces resolve onto
-- these via entity_aliases; unknown surfaces become status='candidate' rows.
CREATE TABLE entities (
    id BIGSERIAL PRIMARY KEY,
    entity_type VARCHAR(32) NOT NULL,
    slug VARCHAR(160) NOT NULL,
    canonical_name VARCHAR(255) NOT NULL,
    name_ta VARCHAR(255) NOT NULL DEFAULT '',
    role VARCHAR(64) NOT NULL DEFAULT '',
    party VARCHAR(64) NOT NULL DEFAULT '',
    district VARCHAR(128) NOT NULL DEFAULT '',
    portfolio VARCHAR(128) NOT NULL DEFAULT '',
    is_tvk BOOLEAN NOT NULL DEFAULT false,
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_entities_slug UNIQUE (slug)
);

CREATE INDEX ix_entities_entity_type ON entities (entity_type);
CREATE INDEX ix_entities_status ON entities (status);

CREATE TABLE entity_aliases (
    id BIGSERIAL PRIMARY KEY,
    entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    alias VARCHAR(255) NOT NULL,
    normalized VARCHAR(255) NOT NULL,
    lang VARCHAR(8) NOT NULL DEFAULT '',
    CONSTRAINT uq_entity_alias UNIQUE (entity_id, alias)
);

CREATE INDEX ix_entity_aliases_entity_id ON entity_aliases (entity_id);
CREATE INDEX ix_entity_aliases_normalized ON entity_aliases (normalized);

-- One resolved mention: this item references this entity. surface keeps the
-- original free text for audit; exactly one analysis's view per item (the
-- dashboard's non-mock-then-latest winner).
CREATE TABLE item_entities (
    id BIGSERIAL PRIMARY KEY,
    raw_item_id BIGINT NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
    analysis_id BIGINT REFERENCES ai_analysis(id) ON DELETE SET NULL,
    entity_id BIGINT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    mention_field VARCHAR(32) NOT NULL,
    surface VARCHAR(255) NOT NULL,
    portrayal VARCHAR(32) NOT NULL DEFAULT '',
    action_summary TEXT NOT NULL DEFAULT '',
    resolver_version VARCHAR(32) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    CONSTRAINT uq_item_entity UNIQUE (raw_item_id, entity_id, mention_field, surface)
);

CREATE INDEX ix_item_entities_raw_item_id ON item_entities (raw_item_id);
CREATE INDEX ix_item_entities_analysis_id ON item_entities (analysis_id);
CREATE INDEX ix_item_entities_entity_id ON item_entities (entity_id);
CREATE INDEX ix_item_entities_mention_field ON item_entities (mention_field);

CREATE TABLE source_checkpoints (
    id BIGSERIAL PRIMARY KEY,
    source_type VARCHAR(64) NOT NULL,
    source_key VARCHAR(255) NOT NULL,
    cursor_name VARCHAR(64) NOT NULL,
    cursor_value TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_source_checkpoint_key UNIQUE (source_type, source_key, cursor_name)
);

CREATE INDEX ix_source_checkpoints_source_type ON source_checkpoints (source_type);
CREATE INDEX ix_source_checkpoints_source_key ON source_checkpoints (source_key);

CREATE TABLE document_chunks (
    id BIGSERIAL PRIMARY KEY,
    raw_item_id BIGINT NOT NULL REFERENCES raw_items(id) ON DELETE CASCADE,
    chunk_version VARCHAR(64) NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    content_hash VARCHAR(64) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_document_chunk_raw_version_index UNIQUE (raw_item_id, chunk_version, chunk_index)
);

CREATE INDEX ix_document_chunks_raw_item_id ON document_chunks (raw_item_id);
CREATE INDEX ix_document_chunks_chunk_version ON document_chunks (chunk_version);
CREATE INDEX ix_document_chunks_content_hash ON document_chunks (content_hash);

CREATE TABLE chunk_embeddings (
    id BIGSERIAL PRIMARY KEY,
    chunk_id BIGINT NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    provider_name VARCHAR(128) NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    embedding JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_chunk_embedding_provider_model UNIQUE (chunk_id, provider_name, model_name)
);

CREATE INDEX ix_chunk_embeddings_chunk_id ON chunk_embeddings (chunk_id);
CREATE INDEX ix_chunk_embeddings_provider_name ON chunk_embeddings (provider_name);
CREATE INDEX ix_chunk_embeddings_model_name ON chunk_embeddings (model_name);
