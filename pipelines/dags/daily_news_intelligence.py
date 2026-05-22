from __future__ import annotations

from datetime import datetime

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

            main()

        run_daily_news_pipeline()

    daily_news_intelligence()
