# X Ingestion Foundation Design

Date: 2026-05-22
Status: Approved for Milestone 3 implementation
Workspace: `TVK_NEWS`

## Product Goal

Add the first official social-media ingestion layer for public X/Twitter posts. This milestone does not try to process 10,000 handles at production volume yet; it builds the compliant, testable foundation that can scale once the X API contract, quota, and handle priority list are approved.

## Scope

Milestone 3 covers X/Twitter only.

Included:

- Approved handle configuration.
- Official X API client abstraction.
- Fake/in-memory client for deterministic tests.
- Recent-post ingestion by handle.
- Cursor/checkpoint storage using `since_id`.
- Normalization into the existing `NormalizedItem` contract.
- AI analysis through the existing analyzer abstraction.
- Idempotent storage in `raw_items` and `ai_analysis`.
- CLI entrypoint for controlled pilot runs.
- API endpoint for configured X handles.

Excluded:

- Unofficial scraping.
- Full 10,000-handle production scheduler.
- Instagram/video processing.
- OpenSearch indexing.
- Semantic clustering or turbovec retrieval.

## Compliance Position

Production X ingestion must use official X API access only. The connector must not scrape web pages, bypass access controls, or rely on undocumented endpoints.

The implementation will use Tweepy as the Python SDK wrapper. Current Tweepy documentation exposes `Client.search_recent_tweets(...)` with `query`, `since_id`, `next_token`, `tweet_fields`, and `max_results`; recent search returns 10 to 100 results per request. That means the product must support batching, cursors, rate-limit recovery, and source prioritization before attempting large handle counts.

## Architecture

The design mirrors the newspaper pipeline but avoids mixing source-specific logic into `DailyNewsPipeline`.

```text
configs/sources.x_handles.yaml
        |
        v
load_x_handle_sources()
        |
        v
DailyXPipeline
        |
        |-- XClient protocol
        |     |-- TweepyXClient for official API
        |     |-- InMemoryXClient for tests
        |
        |-- normalize_x_post()
        |
        |-- save_raw_item()
        |-- get_ai_analysis()
        |-- analyzer.analyze()
        |-- save_ai_analysis()
        |-- save_source_checkpoint()
```

The reusable `source_checkpoints` table stores cursor state for any source type, not only X. Later Instagram provider exports, YouTube channels, and RSS source health can use the same table.

## Data Contracts

### `XHandleSource`

Fields:

- `handle`: X username without `@`.
- `display_name`: optional human-readable name.
- `source_type`: always `x`.
- `language_hint`: default `ta-en-mixed`.
- `priority`: 1-10.
- `active`: boolean.
- `legal_notes`: public official API source note.

The public source name should be `@handle`.

### `XPost`

Fields:

- `id`: X post ID string.
- `handle`: source handle.
- `text`: original post text.
- `created_at`: optional timestamp.
- `lang`: optional platform language code.
- `public_metrics`: likes, replies, reposts, quotes, impressions when available.
- `url`: canonical `https://x.com/{handle}/status/{id}`.
- `metadata`: raw provider metadata safe to store.

### `SourceCheckpoint`

Fields:

- `source_type`
- `source_key`
- `cursor_name`
- `cursor_value`
- `metadata_json`
- `updated_at`

For X, use:

- `source_type = "x"`
- `source_key = "@handle"`
- `cursor_name = "since_id"`
- `cursor_value = newest processed post ID`

## Pipeline Behavior

For each active handle:

1. Load latest `since_id` checkpoint.
2. Query recent posts using the official X client.
3. Exclude retweets in the query where supported.
4. Normalize each post into `NormalizedItem`.
5. Save raw item idempotently by content hash.
6. Reuse existing AI analysis when present.
7. Analyze new raw items through the configured analyzer.
8. Save the AI result.
9. Update the `since_id` checkpoint to the highest post ID seen only after successful processing.
10. Count skipped inactive handles, failures, saved items, and analyses.

The first pilot should support limiting handles and posts per handle from the CLI so testing does not burn quota.

## CLI

Add:

```powershell
python pipelines/run_x_recent.py --limit-handles 100 --max-results 50
```

Behavior:

- Initializes the database.
- Loads `configs/sources.x_handles.yaml`.
- Requires `X_BEARER_TOKEN` when using the real Tweepy client.
- Uses `MockAIAnalyzer` only when `--mock-ai` is provided.
- Prints a compact summary: handles seen, handles skipped, posts seen, items saved, analyses saved, failures.

## API

Add:

- `GET /sources/x`

Returns configured X handle sources. This is intentionally read-only in this milestone.

## Error Handling

- Inactive handles are skipped and counted.
- Missing credentials fail fast in the real CLI path.
- Per-handle client failures increment failures and continue to the next handle.
- Individual post failures increment failures but do not stop the whole run.
- Checkpoints update only after the handle’s posts are processed.

## Testing Strategy

Unit and pipeline tests should not require live X credentials.

Tests must cover:

- YAML config loading.
- `@handle` normalization and invalid handle rejection.
- X post to `NormalizedItem` conversion.
- Public metrics stored in metadata.
- Checkpoint save/read/update.
- Pipeline saves and analyzes posts.
- Rerun does not duplicate rows or re-call analyzer for existing analysis.
- Pipeline uses stored `since_id`.
- API returns configured X sources.
- CLI validates missing real credentials and supports mock mode.

## Production Notes

This milestone prepares the architecture for 10,000 handles but does not promise that volume without API procurement. Production scaling needs:

- handle priority tiers.
- quota dashboards.
- backoff and retry policy based on X API response headers.
- distributed queue workers.
- source freshness monitoring.
- cost and usage reporting.

## References

- Tweepy Client documentation: https://docs.tweepy.org/en/latest/client.html
- X API documentation: https://docs.x.com/x-api/introduction
