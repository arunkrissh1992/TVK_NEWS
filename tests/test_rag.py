from sqlalchemy import func, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from tests.test_storage import make_item
from tnmi.embeddings import HashEmbeddingProvider
from tnmi.rag import RAGIndexer, chunk_normalized_item
from tnmi.storage import (
    ChunkEmbeddingRecord,
    DocumentChunkRecord,
    create_session_factory,
    get_chunk_embedding,
    get_document_chunks,
    init_db,
    save_raw_item,
)
from tnmi.vector_index import InMemoryVectorIndex


def test_chunk_normalized_item_creates_ordered_source_attributed_chunks():
    item = make_item().model_copy(
        update={
            "source_name": "Example Tamil Daily",
            "source_url": "https://example.com/news/1",
            "title": "Tamil Nadu scheme expands",
            "clean_text_original": "Alpha beta gamma delta. Epsilon zeta eta theta. Iota kappa lambda mu.",
        }
    )

    chunks = chunk_normalized_item(item, raw_item_id=42, max_chars=36, overlap_chars=8)

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert chunks[0].raw_item_id == 42
    assert chunks[0].source_name == "Example Tamil Daily"
    assert chunks[0].source_url == "https://example.com/news/1"
    assert chunks[0].title == "Tamil Nadu scheme expands"
    assert all(len(chunk.chunk_text) <= 36 for chunk in chunks)
    assert chunks[1].chunk_text.startswith("zeta ")
    assert chunks[2].chunk_text.endswith("lambda mu.")


def test_rag_indexer_stores_chunks_embeddings_and_vectors_idempotently(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'rag.db'}")
    init_db(session_factory)
    embeddings = HashEmbeddingProvider(dimension=8)
    vector_index = InMemoryVectorIndex(dimension=8)

    with session_factory() as session:
        raw = save_raw_item(
            session,
            make_item().model_copy(
                update={
                    "title": "Water supply issue in Chennai",
                    "clean_text_original": (
                        "Water supply was interrupted in Chennai. "
                        "The government response was reviewed. "
                        "Residents asked for clear restoration timelines."
                    ),
                    "raw_text_original": (
                        "Water supply was interrupted in Chennai. "
                        "The government response was reviewed. "
                        "Residents asked for clear restoration timelines."
                    ),
                }
            ),
        )
        indexer = RAGIndexer(embedding_provider=embeddings, vector_index=vector_index, max_chars=48, overlap_chars=10)

        first = indexer.index_raw_item(session, raw)
        second = indexer.index_raw_item(session, raw)
        chunk_count = session.scalar(select(func.count()).select_from(DocumentChunkRecord))
        embedding_count = session.scalar(select(func.count()).select_from(ChunkEmbeddingRecord))
        chunks = get_document_chunks(session, raw.id)
        first_embedding = get_chunk_embedding(
            session,
            chunks[0].id,
            provider_name=embeddings.provider_name,
            model_name=embeddings.model_name,
        )
        session.commit()

    assert first.chunks_seen >= 3
    assert first.chunks_created == first.chunks_seen
    assert first.embeddings_created == first.chunks_seen
    assert first.vectors_indexed == first.chunks_seen
    assert second.chunks_created == 0
    assert second.embeddings_created == 0
    assert second.vectors_indexed == first.chunks_seen
    assert chunk_count == first.chunks_seen
    assert embedding_count == first.chunks_seen
    assert first_embedding is not None
    assert first_embedding.embedding_dimension == 8


def test_in_memory_vector_index_search_respects_allowlist():
    index = InMemoryVectorIndex(dimension=3)
    index.upsert_many(
        ids=[10, 20, 30],
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.9, 0.1, 0.0],
        ],
    )

    unrestricted = index.search([1.0, 0.0, 0.0], k=2)
    restricted = index.search([1.0, 0.0, 0.0], k=2, allowlist={20, 30})

    assert [result.id for result in unrestricted] == [10, 30]
    assert [result.id for result in restricted] == [30, 20]


def test_postgresql_ddl_includes_rag_tables():
    document_chunks_ddl = str(CreateTable(DocumentChunkRecord.__table__).compile(dialect=postgresql.dialect()))
    chunk_embeddings_ddl = str(CreateTable(ChunkEmbeddingRecord.__table__).compile(dialect=postgresql.dialect()))

    assert "document_chunks" in document_chunks_ddl
    assert "CONSTRAINT uq_document_chunk_raw_version_index UNIQUE (raw_item_id, chunk_version, chunk_index)" in document_chunks_ddl
    assert "chunk_embeddings" in chunk_embeddings_ddl
    assert "CONSTRAINT uq_chunk_embedding_provider_model UNIQUE (chunk_id, provider_name, model_name)" in chunk_embeddings_ddl
