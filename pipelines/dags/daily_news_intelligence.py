from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from airflow.decorators import dag, task
except ImportError:
    dag = None
    task = None


if dag and task:

    @dag(
        dag_id="daily_news_intelligence",
        start_date=datetime(2026, 5, 1),
        schedule="0 6 * * *",
        catchup=False,
        tags=["media-intelligence", "news"],
    )
    def daily_news_intelligence():
        @task
        def run_daily_news_pipeline():
            from pipelines.run_daily_news import main

            main([])

        @task
        def resolve_entities():
            # Refresh the knowledge graph so dossiers, scorecards, the map and
            # spike detection reflect the night's new coverage.
            from pipelines.resolve_entities import main

            main([])

        @task
        def run_flywheel():
            # Self-improving loop: human corrections → gold, teacher→silver,
            # train, eval, gated promote. Idempotent; never auto-promotes without
            # beating the live model on the gold test set.
            from pipelines.flywheel import main

            main(["--teacher", "openai"])

        ingest = run_daily_news_pipeline()
        resolved = resolve_entities()
        learned = run_flywheel()
        ingest >> resolved >> learned

    daily_news_intelligence()
