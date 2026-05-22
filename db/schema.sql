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
    raw_item_id BIGINT NOT NULL,
    model_name VARCHAR(128) NOT NULL,
    prompt_version VARCHAR(64) NOT NULL,
    government_relevance VARCHAR(32) NOT NULL,
    stance_toward_government VARCHAR(32) NOT NULL,
    sentiment VARCHAR(32) NOT NULL,
    target TEXT NOT NULL,
    department VARCHAR(128) NOT NULL,
    district VARCHAR(128) NOT NULL,
    scheme VARCHAR(255),
    topic TEXT NOT NULL,
    issue_category VARCHAR(128) NOT NULL,
    severity VARCHAR(64) NOT NULL,
    summary_original TEXT NOT NULL,
    summary_english TEXT NOT NULL,
    positive_points JSONB NOT NULL DEFAULT '[]'::jsonb,
    negative_points JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_quotes_original JSONB NOT NULL DEFAULT '[]'::jsonb,
    evidence_quotes_english JSONB NOT NULL DEFAULT '[]'::jsonb,
    confidence DOUBLE PRECISION NOT NULL,
    needs_human_review BOOLEAN NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX ix_ai_analysis_raw_item_id ON ai_analysis (raw_item_id);
CREATE INDEX ix_ai_analysis_government_relevance ON ai_analysis (government_relevance);
CREATE INDEX ix_ai_analysis_stance_toward_government ON ai_analysis (stance_toward_government);
CREATE INDEX ix_ai_analysis_department ON ai_analysis (department);
CREATE INDEX ix_ai_analysis_district ON ai_analysis (district);
