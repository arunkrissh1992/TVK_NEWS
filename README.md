# Tamil Nadu Public Media Intelligence

Tamil-first public media intelligence platform for newspaper, social, and video analysis.

## Milestone 1

Daily newspaper ingestion and AI analysis.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest
```

## Run Daily News Pipeline

Available after the upcoming Milestone 1 CLI entrypoint task:

```powershell
python pipelines/run_daily_news.py --date 2026-05-21
```
