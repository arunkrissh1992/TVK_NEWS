from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path


def test_run_daily_news_help_works_as_direct_script():
    result = subprocess.run(
        [sys.executable, "pipelines/run_daily_news.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--date" in result.stdout


def test_daily_news_dag_imports_without_airflow():
    dag_path = Path("pipelines/dags/daily_news_intelligence.py")

    namespace = runpy.run_path(str(dag_path))

    assert namespace["dag"] is None
    assert namespace["task"] is None
