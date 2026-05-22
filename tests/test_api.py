from fastapi.testclient import TestClient

from apps.api import main as api_main
from apps.api.main import app
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item
from tests.test_storage import make_analysis, make_item


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version_endpoint():
    client = TestClient(app)

    response = client.get("/version")

    assert response.status_code == 200
    assert response.json()["version"] == "0.1.0"


def test_inspection_endpoints_return_configured_sources_items_analyses_and_reports(monkeypatch, tmp_path):
    config_path = tmp_path / "sources.yaml"
    config_path.write_text(
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
    legal_notes: Test source.
""",
        encoding="utf-8",
    )
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    (report_dir / "daily-news-2026-05-21.md").write_text("# Report\n", encoding="utf-8")
    (report_dir / "notes.txt").write_text("ignored\n", encoding="utf-8")

    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'api.db'}")
    init_db(session_factory)
    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        session.commit()

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api.db'}"
        news_source_config = config_path
        report_output_dir = report_dir

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    client = TestClient(app)

    sources = client.get("/sources")
    items = client.get("/items")
    analyses = client.get("/analyses")
    reports = client.get("/reports")

    assert sources.status_code == 200
    assert sources.json()[0]["name"] == "Example Tamil Daily"
    assert items.status_code == 200
    assert items.json()[0]["source_url"] == "https://example.com/a"
    assert analyses.status_code == 200
    assert analyses.json()[0]["raw_item_id"] == raw.id
    assert analyses.json()[0]["summary"] == "Positive item."
    assert reports.status_code == 200
    assert reports.json() == [{"filename": "daily-news-2026-05-21.md"}]
