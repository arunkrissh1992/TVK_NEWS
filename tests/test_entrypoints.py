from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import runpy
import subprocess
import sys
from pathlib import Path

import pytest

from tnmi.pipeline import PipelineResult


import pipelines.run_daily_news as run_daily_news


def test_run_daily_news_help_works_as_direct_script():
    result = subprocess.run(
        [sys.executable, "pipelines/run_daily_news.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--date" in result.stdout


def test_invalid_date_exits_before_pipeline_setup(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("pipeline setup should not run for invalid dates")

    monkeypatch.setattr(run_daily_news, "Settings", fail_if_called)

    with pytest.raises(SystemExit):
        run_daily_news.main(["--date", "2026-99-99"])


def test_main_accepts_empty_argv_and_uses_fakes(monkeypatch, capsys):
    calls: list[object] = []
    sources = [object()]
    session_factory = object()
    news_client = object()
    analyzer = object()

    @dataclass
    class FakeSettings:
        database_url: str = "sqlite:///fake.db"
        news_source_config: str = "fake-sources.yaml"
        openai_api_key: str | None = None
        openai_model_item_classifier: str = "fake-model"

    class FakePipeline:
        def __init__(self, *, session_factory, news_client, analyzer):
            calls.append((session_factory, news_client, analyzer))

        def run(self, passed_sources):
            calls.append(passed_sources)
            return PipelineResult(items_seen=2, items_saved=1, analyses_saved=1, failures=0)

    monkeypatch.setattr(run_daily_news, "Settings", FakeSettings)
    monkeypatch.setattr(run_daily_news, "load_newspaper_sources", lambda path: sources)
    monkeypatch.setattr(run_daily_news, "create_session_factory", lambda url: session_factory)
    monkeypatch.setattr(run_daily_news, "init_db", lambda factory: calls.append(("init_db", factory)))
    monkeypatch.setattr(run_daily_news, "DailyNewsPipeline", FakePipeline)
    monkeypatch.setattr(run_daily_news, "RequestsNewsClient", lambda: news_client)
    monkeypatch.setattr(run_daily_news, "build_analyzer", lambda settings: analyzer)

    run_daily_news.main([])

    output = capsys.readouterr().out
    assert f"date={date.today().isoformat()}" in output
    assert "items_seen=2 items_saved=1 analyses_saved=1 failures=0" in output
    assert calls == [
        ("init_db", session_factory),
        (session_factory, news_client, analyzer),
        sources,
    ]


def test_daily_news_dag_imports_without_airflow():
    dag_path = Path("pipelines/dags/daily_news_intelligence.py")

    namespace = runpy.run_path(str(dag_path))

    assert namespace["dag"] is None
    assert namespace["task"] is None


def test_daily_news_dag_task_invokes_main_with_empty_argv():
    source = Path("pipelines/dags/daily_news_intelligence.py").read_text(encoding="utf-8")

    assert "main([])" in source
