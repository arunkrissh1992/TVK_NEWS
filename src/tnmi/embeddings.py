from __future__ import annotations

import hashlib
import math
from typing import Protocol

from openai import OpenAI


class EmbeddingProvider(Protocol):
    provider_name: str
    model_name: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class HashEmbeddingProvider:
    """Deterministic local embeddings for tests and demos, not semantic search."""

    provider_name = "local"
    model_name = "hash-embedding-v1"

    def __init__(self, *, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [_unit_vector(_hash_to_vector(text, self.dimension)) for text in texts]


class OpenAIEmbeddingProvider:
    provider_name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        dimension: int = 1536,
    ) -> None:
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.model_name, input=texts)
        vectors = [list(item.embedding) for item in response.data]
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    f"embedding dimension mismatch: expected {self.dimension}, got {len(vector)}"
                )
        return vectors


def _hash_to_vector(text: str, dimension: int) -> list[float]:
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        digest = hashlib.sha256(f"{counter}:{text}".encode("utf-8")).digest()
        for byte in digest:
            values.append((byte / 127.5) - 1.0)
            if len(values) == dimension:
                break
        counter += 1
    return values


def _unit_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
