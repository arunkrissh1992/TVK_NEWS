from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from tnmi.contracts import NewspaperSource, XHandleSource


class Settings(BaseSettings):
    database_url: str = "sqlite:///./mediaintel.db"
    openai_api_key: str | None = None
    openai_model_item_classifier: str = "gpt-5-mini"
    openai_model_report: str = "gpt-5.2"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimension: int = 1536
    news_source_config: Path = Path("configs/sources.newspapers.yaml")
    x_source_config: Path = Path("configs/sources.x_handles.yaml")
    x_bearer_token: str | None = None
    report_output_dir: Path = Path("reports/generated")
    operator_api_token: str | None = None
    rag_chunk_max_chars: int = 1200
    rag_chunk_overlap_chars: int = 200
    # Ollama analyser config — local CPU model by default. Set to a
    # -cloud tag (e.g. "gemma3:27b-cloud") after `ollama signin` for
    # the cloud-GPU path: 20× faster, larger/better model.
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma2:2b"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def load_newspaper_sources(path: str | Path) -> list[NewspaperSource]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [NewspaperSource.model_validate(item) for item in data.get("newspapers", [])]


def load_x_handle_sources(path: str | Path) -> list[XHandleSource]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [XHandleSource.model_validate(item) for item in data.get("x_handles", [])]
