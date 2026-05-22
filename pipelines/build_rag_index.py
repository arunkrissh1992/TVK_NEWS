from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tnmi.config import Settings
from tnmi.embeddings import HashEmbeddingProvider, OpenAIEmbeddingProvider
from tnmi.rag import RAGIndexer
from tnmi.storage import RawItemRecord, create_session_factory, init_db
from tnmi.vector_index import InMemoryVectorIndex, TurbovecVectorIndex, VectorIndex


def build_embedding_provider(settings: Settings, *, mock_embeddings: bool):
    if mock_embeddings:
        return HashEmbeddingProvider(dimension=settings.openai_embedding_dimension)
    if settings.openai_api_key:
        return OpenAIEmbeddingProvider(
            api_key=settings.openai_api_key,
            model_name=settings.openai_embedding_model,
            dimension=settings.openai_embedding_dimension,
        )
    raise RuntimeError("OPENAI_API_KEY is required unless --mock-embeddings is provided")


def build_vector_index(*, backend: str, dimension: int) -> VectorIndex:
    if backend == "memory":
        return InMemoryVectorIndex(dimension=dimension)
    if backend == "turbovec":
        return TurbovecVectorIndex(dimension=dimension)
    raise ValueError(f"unsupported vector backend: {backend}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-embeddings", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source-type", default="news")
    parser.add_argument("--vector-backend", choices=["memory", "turbovec"], default="memory")
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        embedding_provider = build_embedding_provider(settings, mock_embeddings=args.mock_embeddings)
        vector_index = build_vector_index(backend=args.vector_backend, dimension=embedding_provider.dimension)
    except (RuntimeError, ValueError, ImportError) as exc:
        parser.error(str(exc))

    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    items_seen = 0
    chunks_seen = 0
    chunks_created = 0
    embeddings_created = 0
    vectors_indexed = 0

    with session_factory() as session:
        statement = (
            select(RawItemRecord)
            .where(RawItemRecord.source_type == args.source_type)
            .order_by(RawItemRecord.ingested_at.desc(), RawItemRecord.id.desc())
        )
        if args.limit is not None:
            statement = statement.limit(max(0, args.limit))
        indexer = RAGIndexer(
            embedding_provider=embedding_provider,
            vector_index=vector_index,
            max_chars=settings.rag_chunk_max_chars,
            overlap_chars=settings.rag_chunk_overlap_chars,
        )
        for raw in session.scalars(statement):
            result = indexer.index_raw_item(session, raw)
            items_seen += 1
            chunks_seen += result.chunks_seen
            chunks_created += result.chunks_created
            embeddings_created += result.embeddings_created
            vectors_indexed += result.vectors_indexed
        session.commit()

    print(
        f"items_seen={items_seen} chunks_seen={chunks_seen} chunks_created={chunks_created} "
        f"embeddings_created={embeddings_created} vectors_indexed={vectors_indexed} "
        f"embedding_provider={embedding_provider.provider_name} embedding_model={embedding_provider.model_name} "
        f"vector_backend={args.vector_backend}"
    )


if __name__ == "__main__":
    main()
