from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

from openai import OpenAI


_WORD_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
_DEFAULT_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "or", "in", "on", "for", "with",
    "is", "are", "was", "were", "be", "been", "as", "at", "by", "this", "that",
    "from", "it", "its", "has", "have", "had", "but", "not", "their", "they",
    "we", "i", "you", "he", "she", "his", "her", "our", "your", "if", "than",
    "then", "so", "such", "into", "out", "up", "down", "over", "under", "also",
    "will", "would", "could", "should", "may", "can", "do", "does", "did",
}


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


class LocalBagOfWordsEmbeddingProvider:
    """Bag-of-words local embeddings that DO produce similar vectors for similar
    content. Suitable for demo theme-clustering without an OpenAI key.

    Each Unicode word is hashed into a single bucket; the resulting bucket
    counts are L2-normalised. Two articles sharing many words will land at high
    cosine similarity. Not as good as a real sentence-transformer model, but
    good enough to demonstrate Recurring-Theme detection on the demo data.
    """

    provider_name = "local"
    model_name = "bag-of-words-v1"

    def __init__(self, *, dimension: int = 1024, stopwords: set[str] | None = None) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self.dimension = dimension
        self.stopwords = stopwords if stopwords is not None else _DEFAULT_STOPWORDS

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        for token in _WORD_TOKEN_RE.findall((text or "").lower()):
            if token in self.stopwords or len(token) < 2:
                continue
            bucket = int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big") % self.dimension
            vector[bucket] += 1.0
        return _unit_vector(vector)


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
