from pathlib import Path

from tnmi.config import Settings, load_newspaper_sources


def test_load_newspaper_sources_from_yaml(tmp_path: Path):
    config = tmp_path / "sources.yaml"
    config.write_text(
        """
newspapers:
  - name: Example Tamil Daily
    language_hint: ta
    priority: 1
    active: true
    rss_urls:
      - https://example.com/rss
    sitemap_urls: []
    section_urls: []
""",
        encoding="utf-8",
    )

    sources = load_newspaper_sources(config)

    assert len(sources) == 1
    assert sources[0].name == "Example Tamil Daily"
    assert str(sources[0].rss_urls[0]) == "https://example.com/rss"


def test_settings_ignore_unrelated_dotenv_keys(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
DATABASE_URL=sqlite:///./custom.db
UNRELATED_KEY=ignored
""",
        encoding="utf-8",
    )

    settings = Settings(_env_file=env_file)

    assert settings.database_url == "sqlite:///./custom.db"
