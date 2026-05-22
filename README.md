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

```powershell
python pipelines/run_daily_news.py --date 2026-05-21
```

## Operator Dashboard

Run the API:

```powershell
uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

The internal review console is available at `http://127.0.0.1:8000/dashboard`.

For confidential deployments, set `OPERATOR_API_TOKEN` and send the token as `X-TNMI-Operator-Token` for dashboard and review APIs. This is a first local guard only; production deployments still need SSO/RBAC, private networking, audit logs, and managed secrets.

## Run X Recent Pipeline

X ingestion uses approved official X API access only. Configure `X_BEARER_TOKEN`, review `configs/sources.x_handles.yaml`, and enable approved handles before running:

```powershell
python pipelines/run_x_recent.py --limit-handles 100 --max-results 50 --mock-ai
```

Remove `--mock-ai` only when `OPENAI_API_KEY` is configured for real item classification.
