from tests.test_storage import make_item
from tnmi.storage import ChunkEmbeddingRecord, DocumentChunkRecord, create_session_factory, init_db, save_raw_item


def test_build_rag_index_stores_chunks_and_embeddings_with_mock_provider(monkeypatch, tmp_path, capsys):
    import pipelines.build_rag_index as build_rag_index

    db_path = tmp_path / "rag-entrypoint.db"
    session_factory = create_session_factory(f"sqlite:///{db_path}")
    init_db(session_factory)
    with session_factory() as session:
        save_raw_item(
            session,
            make_item().model_copy(
                update={
                    "clean_text_original": "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda.",
                    "raw_text_original": "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda.",
                }
            ),
        )
        session.commit()

    class FakeSettings:
        database_url = f"sqlite:///{db_path}"
        openai_api_key = None
        openai_embedding_model = "fake-embedding"
        openai_embedding_dimension = 8
        rag_chunk_max_chars = 32
        rag_chunk_overlap_chars = 8

    monkeypatch.setattr(build_rag_index, "Settings", FakeSettings)

    build_rag_index.main(["--mock-embeddings"])

    output = capsys.readouterr().out
    assert "items_seen=1" in output
    assert "embeddings_created=" in output
    with session_factory() as session:
        assert session.query(DocumentChunkRecord).count() >= 2
        assert session.query(ChunkEmbeddingRecord).count() >= 2


def test_build_rag_index_requires_openai_key_without_mock(monkeypatch):
    import pipelines.build_rag_index as build_rag_index

    class FakeSettings:
        database_url = "sqlite:///unused.db"
        openai_api_key = None
        openai_embedding_model = "fake-embedding"
        openai_embedding_dimension = 8
        rag_chunk_max_chars = 32
        rag_chunk_overlap_chars = 8

    monkeypatch.setattr(build_rag_index, "Settings", FakeSettings)

    try:
        build_rag_index.main([])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing OPENAI_API_KEY should exit")
