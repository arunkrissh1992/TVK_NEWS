from __future__ import annotations

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from tnmi.contracts import NewspaperSource


class Settings(BaseSettings):
    database_url: str = "sqlite:///./mediaintel.db"
    openai_api_key: str | None = None
    openai_model_item_classifier: str = "gpt-5.4-mini"
    openai_model_report: str = "gpt-5.5"
    news_source_config: Path = Path("configs/sources.newspapers.yaml")
    report_output_dir: Path = Path("reports/generated")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


def load_newspaper_sources(path: str | Path) -> list[NewspaperSource]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [NewspaperSource.model_validate(item) for item in data.get("newspapers", [])]
