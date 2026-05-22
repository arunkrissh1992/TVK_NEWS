# RAG and AI Provider Architecture Design

Date: 2026-05-22
Status: Approved direction, initial implementation started
Workspace: `TVK_NEWS`

## Goal

Add a production-shaped retrieval layer so newspaper articles, social posts, transcripts, and OCR text can be indexed as source-attributed chunks, retrieved semantically, and passed to AI for evidence-backed analysis and report synthesis.

## Core Decision

RAG is a support layer, not the primary database and not the first step for every analysis.

The platform stores canonical records in PostgreSQL, keeps keyword/search analytics in OpenSearch, and uses a vector index for semantic retrieval. `turbovec` is useful as an optional local/VPC vector backend because it keeps vectors inside our infrastructure and supports filtered search, but it must sit behind an internal `VectorIndex` interface. This lets us swap to OpenSearch vector search, pgvector, FAISS, or a managed/private vector store later.

## Data Flow

```text
RSS / X / provider / video
  -> raw_items
  -> extraction/transcript/OCR
  -> normalized item text
  -> document_chunks
  -> chunk_embeddings
  -> VectorIndex adapter
  -> retrieved evidence context
  -> AI analyzer / daily report writer
```

## Storage Model

`document_chunks` stores the text chunks derived from a raw item. Each chunk is versioned by `chunk_version` so chunking strategy can evolve without corrupting prior analysis.

`chunk_embeddings` stores provider/model-specific vectors for chunks. Embeddings are idempotent by chunk, provider, and model. Production deployments may keep full vectors in this table for audit and rebuilds while also writing them into the active vector backend.

## AI Provider Boundary

The product should not import a coding-agent CLI as its AI engine. The inspected Claude-style reference repository is a leaked/source-map snapshot for security research, not a clean reusable dependency for a confidential government product. It can inform architecture ideas such as provider routing, permissions, and tool registries, but it should not be copied or integrated.

The product-owned AI boundary should remain simple:

```text
AIAnalyzer
  -> OpenAIAnalyzer
  -> AnthropicAnalyzer
  -> AzureOpenAIAnalyzer
  -> LocalLLMAnalyzer
  -> MockAIAnalyzer for tests only

EmbeddingProvider
  -> OpenAIEmbeddingProvider
  -> LocalEmbeddingProvider
  -> HashEmbeddingProvider for tests only
```

Provider limits must not be bypassed in code. The correct production answer is queueing, backoff, batching, approved keys, budget controls, and contracted rate limits.

## Retrieval Rules

Every retrieval result must preserve:

- raw item id
- chunk id
- source type
- source name
- source URL
- title
- language
- published date when available
- chunk version
- model/provider used for embedding

AI report prompts must cite retrieved chunks and original source URLs. If a claim is sensitive, an allegation, or low-confidence, it must be sent to human review rather than treated as established fact.

## First Implementation Slice

The first slice adds:

- `DocumentChunk` contract.
- `document_chunks` and `chunk_embeddings` tables.
- deterministic local embedding provider for tests and demos.
- OpenAI embedding provider boundary.
- in-memory vector index for tests.
- optional `TurbovecVectorIndex` adapter imported only when selected.
- `RAGIndexer` service that chunks a raw item, stores chunks, embeds missing chunks, and indexes vectors idempotently.

The next slice should connect this to the daily newspaper pipeline as an optional post-analysis indexing step, then expose retrieval-backed report synthesis.
