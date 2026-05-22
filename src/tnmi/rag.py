from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from tnmi.contracts import DocumentChunk, NormalizedItem, SourceType
from tnmi.embeddings import EmbeddingProvider
from tnmi.storage import (
    RawItemRecord,
    get_chunk_embedding,
    get_document_chunks,
    save_chunk_embedding,
    save_document_chunks,
)
from tnmi.vector_index import VectorIndex


CHUNK_VERSION = "article-chunk-v1"


@dataclass(frozen=True)
class RAGIndexingResult:
    chunks_seen: int
    chunks_created: int
    embeddings_created: int
    vectors_indexed: int


class RAGIndexer:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_index: VectorIndex,
        chunk_version: str = CHUNK_VERSION,
        max_chars: int = 1200,
        overlap_chars: int = 200,
    ) -> None:
        if embedding_provider.dimension != vector_index.dimension:
            raise ValueError("embedding provider and vector index dimensions must match")
        self.embedding_provider = embedding_provider
        self.vector_index = vector_index
        self.chunk_version = chunk_version
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def index_raw_item(self, session: Session, raw: RawItemRecord) -> RAGIndexingResult:
        item = _normalized_item_from_record(raw)
        chunks = chunk_normalized_item(
            item,
            raw_item_id=raw.id,
            chunk_version=self.chunk_version,
            max_chars=self.max_chars,
            overlap_chars=self.overlap_chars,
        )
        existing_chunk_keys = {
            (record.chunk_version, record.chunk_index)
            for record in get_document_chunks(session, raw.id, chunk_version=self.chunk_version)
        }
        records = save_document_chunks(session, chunks)
        chunks_created = sum(
            1 for chunk in chunks if (chunk.chunk_version, chunk.chunk_index) not in existing_chunk_keys
        )
        embeddings_created = 0
        ids_to_index: list[int] = []
        vectors_to_index: list[list[float]] = []

        for record in records:
            existing_embedding = get_chunk_embedding(
                session,
                record.id,
                provider_name=self.embedding_provider.provider_name,
                model_name=self.embedding_provider.model_name,
            )
            if existing_embedding:
                ids_to_index.append(record.id)
                vectors_to_index.append(existing_embedding.embedding)
                continue

            vector = self.embedding_provider.embed_texts([record.chunk_text])[0]
            save_chunk_embedding(
                session,
                chunk_id=record.id,
                provider_name=self.embedding_provider.provider_name,
                model_name=self.embedding_provider.model_name,
                embedding=vector,
            )
            embeddings_created += 1
            ids_to_index.append(record.id)
            vectors_to_index.append(vector)

        self.vector_index.upsert_many(ids=ids_to_index, vectors=vectors_to_index)
        return RAGIndexingResult(
            chunks_seen=len(chunks),
            chunks_created=chunks_created,
            embeddings_created=embeddings_created,
            vectors_indexed=len(ids_to_index),
        )


def chunk_normalized_item(
    item: NormalizedItem,
    *,
    raw_item_id: int,
    chunk_version: str = CHUNK_VERSION,
    max_chars: int = 1200,
    overlap_chars: int = 200,
) -> list[DocumentChunk]:
    text_chunks = split_text_for_rag(item.clean_text_original, max_chars=max_chars, overlap_chars=overlap_chars)
    return [
        DocumentChunk(
            raw_item_id=raw_item_id,
            source_type=item.source_type,
            source_name=item.source_name,
            source_url=item.source_url,
            language=item.language,
            title=item.title,
            published_at=item.published_at,
            chunk_version=chunk_version,
            chunk_index=index,
            chunk_text=chunk,
            token_estimate=max(1, len(chunk) // 4),
            metadata={"source_type": item.source_type.value},
        )
        for index, chunk in enumerate(text_chunks)
    ]


def split_text_for_rag(text: str, *, max_chars: int, overlap_chars: int) -> list[str]:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be non-negative and smaller than max_chars")

    normalized = " ".join(text.split())
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [normalized]

    words = normalized.split(" ")
    chunks: list[str] = []
    start_word = 0
    while start_word < len(words):
        end_word = start_word
        current_words: list[str] = []
        while end_word < len(words):
            candidate = [*current_words, words[end_word]]
            if len(" ".join(candidate)) > max_chars and current_words:
                break
            current_words = candidate
            end_word += 1
        chunk = " ".join(current_words).strip()
        if chunk:
            chunks.append(chunk)
        if end_word == len(words):
            break
        overlap_word_count = _overlap_word_count(current_words, overlap_chars)
        start_word = max(start_word + 1, end_word - overlap_word_count)
    return chunks


def _overlap_word_count(words: list[str], overlap_chars: int) -> int:
    if overlap_chars == 0:
        return 0
    selected = 0
    selected_chars = 0
    for word in reversed(words):
        added_chars = len(word) if selected == 0 else len(word) + 1
        if selected > 0 and selected_chars + added_chars > overlap_chars:
            break
        selected += 1
        selected_chars += added_chars
    return selected


def _normalized_item_from_record(raw: RawItemRecord) -> NormalizedItem:
    return NormalizedItem(
        source_type=SourceType(raw.source_type),
        source_name=raw.source_name,
        source_url=raw.source_url,
        published_at=raw.published_at,
        language=raw.language,
        title=raw.title,
        raw_text_original=raw.raw_text_original,
        clean_text_original=raw.clean_text_original,
        metadata=raw.metadata_json,
    )
