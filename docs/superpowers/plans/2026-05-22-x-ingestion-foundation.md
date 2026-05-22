# X Ingestion Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a compliant, cursor-aware X/Twitter ingestion foundation using the official API path only.

**Architecture:** Add X source contracts and config loading, a reusable source checkpoint table, and a source-specific `DailyXPipeline` that normalizes posts into the existing `raw_items` and `ai_analysis` tables. The real client is isolated behind an `XClient` protocol so tests use an in-memory client and production can use Tweepy with `X_BEARER_TOKEN`.

**Tech Stack:** Python 3.11+, Pydantic, SQLAlchemy, PyYAML, Tweepy optional runtime client, FastAPI, pytest.

---

## File Structure

- Modify `src/tnmi/contracts.py`: add `XHandleSource` and `XPost`.
- Modify `src/tnmi/config.py`: add `x_source_config`, `x_bearer_token`, and `load_x_handle_sources`.
- Create `configs/sources.x_handles.yaml`: inactive pilot handle examples pending approved access.
- Modify `src/tnmi/storage.py`: add `SourceCheckpointRecord` and checkpoint helpers.
- Modify `db/schema.sql`: add `source_checkpoints`.
- Create `src/tnmi/x_ingestion.py`: X client protocol, in-memory client, Tweepy client, normalization, and pipeline.
- Create `pipelines/run_x_recent.py`: controlled X ingestion CLI.
- Modify `apps/api/main.py`: add `GET /sources/x`.
- Modify `pyproject.toml`: add `tweepy`.
- Modify `.env.example`: add X settings.
- Modify `README.md`: document X pilot command and official-access rule.
- Create/modify tests:
  - `tests/test_config.py`
  - `tests/test_storage.py`
  - `tests/test_x_ingestion.py`
  - `tests/test_entrypoints.py`
  - `tests/test_api.py`

---

### Task 1: X Contracts and Config

**Files:**
- Modify: `src/tnmi/contracts.py`
- Modify: `src/tnmi/config.py`
- Create: `configs/sources.x_handles.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Add failing config tests**

Add tests for valid handle loading and invalid handle rejection.

- [ ] **Step 2: Implement contracts/config**

Add `XHandleSource`, `XPost`, `Settings.x_source_config`, `Settings.x_bearer_token`, and `load_x_handle_sources`.

- [ ] **Step 3: Add sample X config**

Create inactive pilot handles so no live ingestion starts accidentally.

- [ ] **Step 4: Run tests**

Run `python -m pytest tests/test_config.py -q`.

- [ ] **Step 5: Commit**

Commit with `feat: add x source configuration`.

---

### Task 2: Source Checkpoint Storage

**Files:**
- Modify: `src/tnmi/storage.py`
- Modify: `db/schema.sql`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Add failing checkpoint tests**

Cover create/update/read and PostgreSQL DDL uniqueness.

- [ ] **Step 2: Implement checkpoint model/helpers**

Add `SourceCheckpointRecord`, `get_source_checkpoint`, and `save_source_checkpoint`.

- [ ] **Step 3: Run storage tests**

Run `python -m pytest tests/test_storage.py -q`.

- [ ] **Step 4: Commit**

Commit with `feat: add source checkpoint storage`.

---

### Task 3: X Ingestion Service

**Files:**
- Create: `src/tnmi/x_ingestion.py`
- Create: `tests/test_x_ingestion.py`

- [ ] **Step 1: Add failing X ingestion tests**

Cover post normalization, metadata preservation, pipeline save/analyze, rerun behavior, and `since_id` use.

- [ ] **Step 2: Implement X ingestion module**

Add `XClient`, `InMemoryXClient`, `TweepyXClient`, `normalize_x_post`, `XIngestionResult`, and `DailyXPipeline`.

- [ ] **Step 3: Run X ingestion tests**

Run `python -m pytest tests/test_x_ingestion.py -q`.

- [ ] **Step 4: Commit**

Commit with `feat: add x ingestion pipeline`.

---

### Task 4: CLI, API, and Docs

**Files:**
- Create: `pipelines/run_x_recent.py`
- Modify: `apps/api/main.py`
- Modify: `tests/test_entrypoints.py`
- Modify: `tests/test_api.py`
- Modify: `pyproject.toml`
- Modify: `.env.example`
- Modify: `README.md`

- [ ] **Step 1: Add failing CLI/API tests**

Cover `--help`, missing token validation, mock mode execution with monkeypatched fakes, and `/sources/x`.

- [ ] **Step 2: Implement CLI/API/docs**

Add controlled CLI, API route, dependency/env docs, and README usage.

- [ ] **Step 3: Run endpoint tests**

Run `python -m pytest tests/test_entrypoints.py tests/test_api.py -q`.

- [ ] **Step 4: Commit**

Commit with `feat: expose x ingestion entrypoints`.

---

### Task 5: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Full tests**

Run `python -m pytest -q`.

- [ ] **Step 2: Compose validation**

Run `docker compose config`.

- [ ] **Step 3: Diff whitespace**

Run `git diff --check master...HEAD`.

- [ ] **Step 4: Branch status**

Run `git status --short --branch`.

- [ ] **Step 5: Finish branch**

Use `superpowers:finishing-a-development-branch`.

---

## Self-Review

- Spec coverage: Covers X config, official API abstraction, checkpointing, normalization, AI analysis, CLI, API, and tests.
- Placeholder scan: No placeholder tasks remain.
- Type consistency: `XHandleSource`, `XPost`, `SourceCheckpointRecord`, `DailyXPipeline`, and entrypoint names are used consistently.
