from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class VectorSearchResult:
    id: int
    score: float


class VectorIndex(Protocol):
    dimension: int

    def upsert_many(self, *, ids: list[int], vectors: list[list[float]]) -> None:
        ...

    def search(
        self,
        query_vector: list[float],
        *,
        k: int,
        allowlist: set[int] | None = None,
    ) -> list[VectorSearchResult]:
        ...


class InMemoryVectorIndex:
    def __init__(self, *, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self._vectors: dict[int, list[float]] = {}

    def upsert_many(self, *, ids: list[int], vectors: list[list[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        for vector_id, vector in zip(ids, vectors, strict=True):
            self._validate_vector(vector)
            self._vectors[vector_id] = vector

    def search(
        self,
        query_vector: list[float],
        *,
        k: int,
        allowlist: set[int] | None = None,
    ) -> list[VectorSearchResult]:
        if k <= 0:
            return []
        self._validate_vector(query_vector)
        candidates = self._vectors.items()
        if allowlist is not None:
            candidates = ((vector_id, vector) for vector_id, vector in candidates if vector_id in allowlist)
        results = [
            VectorSearchResult(id=vector_id, score=_cosine_similarity(query_vector, vector))
            for vector_id, vector in candidates
        ]
        return sorted(results, key=lambda result: (-result.score, result.id))[:k]

    def _validate_vector(self, vector: list[float]) -> None:
        if len(vector) != self.dimension:
            raise ValueError(f"expected vector dimension {self.dimension}, got {len(vector)}")


class TurbovecVectorIndex:
    """Optional local vector backend. Imports turbovec only when selected."""

    def __init__(self, *, dimension: int, bit_width: int = 4) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        import numpy as np
        from turbovec import IdMapIndex

        self.dimension = dimension
        self._np = np
        self._index = IdMapIndex(dim=dimension, bit_width=bit_width)

    def upsert_many(self, *, ids: list[int], vectors: list[list[float]]) -> None:
        if not ids:
            return
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have the same length")
        matrix = self._np.array(vectors, dtype=self._np.float32)
        stable_ids = self._np.array(ids, dtype=self._np.uint64)
        self._index.add_with_ids(matrix, stable_ids)

    def search(
        self,
        query_vector: list[float],
        *,
        k: int,
        allowlist: set[int] | None = None,
    ) -> list[VectorSearchResult]:
        if k <= 0:
            return []
        query = self._np.array([query_vector], dtype=self._np.float32)
        allowed = None
        if allowlist is not None:
            allowed = self._np.array(sorted(allowlist), dtype=self._np.uint64)
        scores, ids = self._index.search(query, k=k, allowlist=allowed)
        return [
            VectorSearchResult(id=int(vector_id), score=float(score))
            for score, vector_id in zip(scores[0], ids[0], strict=True)
        ]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)
