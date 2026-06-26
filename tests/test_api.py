import json

from fastapi.testclient import TestClient

from apps.api import main as api_main
from apps.api.main import app
from tnmi.contracts import Severity, Stance
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


def test_x_sources_endpoint_returns_configured_handles(monkeypatch, tmp_path):
    x_config = tmp_path / "x.yaml"
    x_config.write_text(
        """
x_handles:
  - handle: ExampleTNNews
    display_name: Example TN News
    active: false
""",
        encoding="utf-8",
    )

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api.db'}"
        news_source_config = tmp_path / "missing-news.yaml"
        x_source_config = x_config
        report_output_dir = tmp_path / "reports"
        operator_api_token = None

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    client = TestClient(app)

    response = client.get("/sources/x")

    assert response.status_code == 200
    assert response.json()[0]["handle"] == "ExampleTNNews"
    assert response.json()[0]["source_name"] == "@ExampleTNNews"


def test_dashboard_json_and_review_queue_endpoints(monkeypatch, tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'api-dashboard.db'}")
    init_db(session_factory)
    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        analysis = save_ai_analysis(
            session,
            raw.id,
            make_analysis().model_copy(update={"needs_human_review": True}),
            model_name="mock",
            prompt_version="v1",
        )
        session.commit()

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api-dashboard.db'}"
        news_source_config = tmp_path / "missing.yaml"
        report_output_dir = tmp_path / "reports"
        operator_api_token = None

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    client = TestClient(app)

    summary = client.get("/dashboard/summary")
    queue = client.get("/review/queue")

    assert summary.status_code == 200
    assert summary.json()["needs_human_review"] == 1
    assert queue.status_code == 200
    assert queue.json()[0]["analysis_id"] == analysis.id


def test_dashboard_alerts_endpoint_surfaces_urgent_negative_item(monkeypatch, tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'api-alerts.db'}")
    init_db(session_factory)
    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        save_ai_analysis(
            session,
            raw.id,
            make_analysis().model_copy(
                update={
                    "stance_toward_government": Stance.NEGATIVE,
                    "tvk_portrayal": Stance.NEGATIVE,
                    "severity": Severity.CRITICAL,
                    "action_priority": Severity.CRITICAL,
                    "needs_human_review": True,
                    "summary_english": "Urgent negative item.",
                    "risk_if_ignored": "Narrative hardens against TVK.",
                }
            ),
            model_name="mock",
            prompt_version="v1",
        )
        session.commit()

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api-alerts.db'}"
        news_source_config = tmp_path / "missing.yaml"
        report_output_dir = tmp_path / "reports"
        operator_api_token = None

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    client = TestClient(app)

    response = client.get("/dashboard/alerts")

    assert response.status_code == 200
    payload = response.json()
    # New shape: forward-looking emerging signals + the urgent priority backlog.
    assert "emerging_signals" in payload
    assert isinstance(payload["emerging_signals"], list)
    priority = payload["priority"]
    assert len(priority) == 1
    assert priority[0]["display_category"] == "negative"
    assert priority[0]["action_priority"] == "critical"
    assert priority[0]["needs_human_review"] is True
    assert priority[0]["risk_if_ignored"] == "Narrative hardens against TVK."


def test_review_decision_endpoint_records_latest_review(monkeypatch, tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'api-review.db'}")
    init_db(session_factory)
    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        analysis = save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        session.commit()

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api-review.db'}"
        news_source_config = tmp_path / "missing.yaml"
        report_output_dir = tmp_path / "reports"
        operator_api_token = None

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    client = TestClient(app)

    response = client.post(
        "/review/decisions",
        json={
            "analysis_id": analysis.id,
            "reviewer_name": "analyst-1",
            "status": "approved",
            "note": "Checked against article evidence.",
        },
    )
    latest = client.get(f"/review/decisions/{analysis.id}")

    assert response.status_code == 200
    assert response.json()["status"] == "approved"
    assert latest.status_code == 200
    assert latest.json()["reviewer_name"] == "analyst-1"


def test_dashboard_html_renders_summary_and_review_queue(monkeypatch, tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'api-html.db'}")
    init_db(session_factory)
    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        save_ai_analysis(
            session,
            raw.id,
            make_analysis().model_copy(update={"needs_human_review": True, "summary_english": "Needs review."}),
            model_name="mock",
            prompt_version="v1",
        )
        session.commit()

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api-html.db'}"
        news_source_config = tmp_path / "missing.yaml"
        report_output_dir = tmp_path / "reports"
        operator_api_token = None

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    client = TestClient(app)

    response = client.get("/dashboard")

    assert response.status_code == 200
    assert "TVK Public Media Briefing" in response.text
    assert "Confidential Daily Briefing" in response.text
    assert 'data-filter="all"' in response.text
    assert 'data-filter="positive"' in response.text
    assert 'data-filter="negative"' in response.text
    assert 'data-filter="mixed"' in response.text
    assert 'data-filter="people"' in response.text
    assert "Positive and Negative Portrayal With Evidence" in response.text
    # Ask AI is now a floating assistant (corner launcher + chat panel)
    assert 'id="ai-fab-toggle"' in response.text
    assert 'id="chat-form"' in response.text
    assert "Ask AI" in response.text
    assert "People Issues" in response.text
    assert "Needs review." in response.text
    assert "Evidence &middot;" in response.text
    assert "OpenAI Live" not in response.text
    assert "RAG Chunks" not in response.text


def test_operator_token_blocks_confidential_dashboard_endpoint(monkeypatch, tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'api-auth.db'}")
    init_db(session_factory)

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api-auth.db'}"
        news_source_config = tmp_path / "missing.yaml"
        report_output_dir = tmp_path / "reports"
        operator_api_token = "secret-token"

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    client = TestClient(app)

    blocked = client.get("/dashboard/summary")
    allowed = client.get("/dashboard/summary", headers={"X-TNMI-Operator-Token": "secret-token"})

    assert blocked.status_code == 401
    assert allowed.status_code == 200


def test_chat_ask_endpoint_returns_ai_answer_with_evidence(monkeypatch, tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'api-chat.db'}")
    init_db(session_factory)
    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        session.commit()

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api-chat.db'}"
        news_source_config = tmp_path / "missing.yaml"
        report_output_dir = tmp_path / "reports"
        operator_api_token = None
        ollama_host = "http://localhost:11434"
        ollama_model = "fake"

    class FakeProvider:
        model_name = "fake-ai"

        def answer(self, question, evidence, *, dossier_context=""):  # type: ignore[no-untyped-def]
            assert question == "What is positive?"
            assert evidence[0].source_url == "https://example.com/a"
            return "The stored evidence shows a positive item."

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    monkeypatch.setattr(api_main, "_build_chat_provider", lambda settings: FakeProvider())
    client = TestClient(app)

    response = client.post("/chat/ask", json={"question": "What is positive?"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["used_ai"] is True
    assert payload["model_name"] == "fake-ai"
    assert payload["answer"] == "The stored evidence shows a positive item."
    assert payload["evidence"][0]["source_url"] == "https://example.com/a"


def _stream_events(text):
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def test_chat_stream_endpoint_streams_evidence_then_answer(monkeypatch, tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'api-stream.db'}")
    init_db(session_factory)
    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        session.commit()

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api-stream.db'}"
        news_source_config = tmp_path / "missing.yaml"
        report_output_dir = tmp_path / "reports"
        operator_api_token = None
        ollama_host = "http://localhost:11434"
        ollama_model = "fake"

    class FakeStreamProvider:
        model_name = "fake-ai"

        def stream_answer(self, question, evidence, *, history=None, dossier_context=""):  # type: ignore[no-untyped-def]
            assert question == "What is positive?"
            assert evidence[0].source_url == "https://example.com/a"
            yield "The stored evidence "
            yield "shows a positive item."

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    monkeypatch.setattr(api_main, "_build_chat_provider", lambda settings: FakeStreamProvider())
    client = TestClient(app)

    response = client.post("/chat/stream", json={"question": "What is positive?"})

    assert response.status_code == 200
    events = _stream_events(response.text)
    types = [event["type"] for event in events]
    assert types[0] == "evidence"
    assert types[-1] == "done"
    assert events[0]["evidence"][0]["source_url"] == "https://example.com/a"
    answer = "".join(event["text"] for event in events if event["type"] == "delta")
    assert answer == "The stored evidence shows a positive item."
    assert events[-1]["used_ai"] is True
    assert events[-1]["model_name"] == "fake-ai"


def test_chat_stream_endpoint_falls_back_to_evidence_only_when_ai_unavailable(monkeypatch, tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'api-stream-fb.db'}")
    init_db(session_factory)
    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        session.commit()

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api-stream-fb.db'}"
        news_source_config = tmp_path / "missing.yaml"
        report_output_dir = tmp_path / "reports"
        operator_api_token = None
        ollama_host = "http://localhost:11434"
        ollama_model = "fake"

    def _no_provider(settings):
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    monkeypatch.setattr(api_main, "_build_chat_provider", _no_provider)
    client = TestClient(app)

    response = client.post("/chat/stream", json={"question": "What is positive?"})

    assert response.status_code == 200
    events = _stream_events(response.text)
    assert events[0]["type"] == "evidence"
    assert events[-1]["type"] == "done"
    assert events[-1]["used_ai"] is False
    assert events[-1]["model_name"] == "evidence-only"
    answer = "".join(event["text"] for event in events if event["type"] == "delta")
    assert answer.strip()


def test_settings_page_and_status_mask_openai_secret(monkeypatch, tmp_path):
    secret = "sk-test-do-not-render"

    class FakeSettings:
        database_url = f"sqlite:///{tmp_path / 'api-settings.db'}"
        news_source_config = tmp_path / "missing.yaml"
        report_output_dir = tmp_path / "reports"
        operator_api_token = None
        openai_api_key = secret
        openai_model_item_classifier = "gpt-5-mini"
        openai_model_report = "gpt-5.2"
        openai_embedding_model = "text-embedding-3-small"
        openai_embedding_dimension = 1536

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    client = TestClient(app)

    page = client.get("/settings")
    status = client.get("/settings/status")

    assert page.status_code == 200
    assert status.status_code == 200
    assert "Configured and hidden" in page.text
    assert secret not in page.text
    assert status.json()["openai_configured"] is True
    assert secret not in str(status.json())


def test_multi_tenant_routes_isolate_data_by_api_key(monkeypatch, tmp_path):
    """With multi_tenant on, X-Tenant-Key routes each request to that tenant's
    own database; a missing/invalid key is rejected."""
    from tnmi.storage import save_raw_item
    from tnmi.tenancy import ControlPlane
    from tests.test_storage import make_item

    control_url = f"sqlite:///{tmp_path / 'control.db'}"

    class FakeSettings:
        multi_tenant = True
        control_database_url = control_url
        tenants_dir = tmp_path / "tenants"
        database_url = f"sqlite:///{tmp_path / 'unused.db'}"
        news_source_config = tmp_path / "missing.yaml"
        report_output_dir = tmp_path / "reports"
        operator_api_token = None

    monkeypatch.setattr(api_main, "Settings", FakeSettings)
    api_main._control_plane.cache_clear()
    api_main._cached_factory.cache_clear()

    control = ControlPlane(control_url, tenants_dir=tmp_path / "tenants")
    tvk = control.provision_tenant(name="TVK", slug="tvk", seed_entities=False)
    control.provision_tenant(name="DMK", slug="dmk", seed_entities=False)
    tvk_key, _ = control.issue_api_key(tenant=tvk)
    dmk = control.get_tenant("dmk")
    dmk_key, _ = control.issue_api_key(tenant=dmk)
    with control.session_factory_for(tvk)() as session:
        save_raw_item(session, make_item().model_copy(update={"title": "TVK only"}))
        session.commit()

    client = TestClient(app)
    # TVK key sees TVK's item.
    r_tvk = client.get("/items", headers={"X-Tenant-Key": tvk_key})
    assert r_tvk.status_code == 200
    assert [i["title"] for i in r_tvk.json()] == ["TVK only"]
    # DMK key sees nothing — different database.
    r_dmk = client.get("/items", headers={"X-Tenant-Key": dmk_key})
    assert r_dmk.status_code == 200
    assert r_dmk.json() == []
    # No key is rejected.
    assert client.get("/items").status_code == 401
    assert client.get("/items", headers={"X-Tenant-Key": "tvk_bogus"}).status_code == 401

    api_main._control_plane.cache_clear()
    api_main._cached_factory.cache_clear()
