# Review Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first confidential operator console for human review of AI newspaper analysis.

**Architecture:** Keep the current newspaper pipeline intact and add a review/audit layer beside `ai_analysis`. A focused dashboard service will aggregate review queue, stance, severity, department, and district metrics for both JSON APIs and a lightweight server-rendered FastAPI console. The console is intentionally plain HTML/CSS now so production can later swap in Next.js without changing storage contracts.

**Tech Stack:** FastAPI, SQLAlchemy, Pydantic, Jinja2, pytest, SQLite for tests, PostgreSQL-compatible schema.

---

## File Structure

- Modify `src/tnmi/contracts.py`: add review status and review request/response contracts.
- Modify `src/tnmi/storage.py`: add `ReviewDecisionRecord` table and review persistence helpers.
- Create `src/tnmi/dashboard.py`: query service for dashboard aggregates and review queue rows.
- Modify `apps/api/main.py`: expose review/dashboard JSON endpoints and HTML console routes.
- Create `apps/api/templates/dashboard.html`: internal operator dashboard.
- Create `apps/api/static/dashboard.css`: restrained console styling for scan-heavy work.
- Modify `src/tnmi/config.py`: add optional operator token setting for confidential endpoints.
- Modify `db/schema.sql`: keep hand-written schema aligned with SQLAlchemy models.
- Modify `.env.example`: document optional operator token.
- Modify `README.md`: document dashboard/review usage.
- Create `tests/test_review_storage.py`: review table and helper tests.
- Create `tests/test_dashboard.py`: aggregate/query service tests.
- Modify `tests/test_api.py`: endpoint, auth, and HTML console tests.

---

### Task 1: Review Storage Foundation

**Files:**
- Modify: `src/tnmi/contracts.py`
- Modify: `src/tnmi/storage.py`
- Modify: `db/schema.sql`
- Create: `tests/test_review_storage.py`
- Modify: `tests/test_storage.py`

- [ ] **Step 1: Write failing storage tests**

Add `tests/test_review_storage.py`:

```python
from sqlalchemy import select
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from tests.test_storage import make_analysis, make_item
from tnmi.contracts import ReviewDecisionCreate, ReviewStatus, Stance
from tnmi.storage import (
    ReviewDecisionRecord,
    create_session_factory,
    get_latest_review_decision,
    init_db,
    save_ai_analysis,
    save_raw_item,
    save_review_decision,
)


def test_save_review_decision_records_operator_audit_row(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'review.db'}")
    init_db(session_factory)

    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        analysis = save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        decision = save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=analysis.id,
                reviewer_name="analyst-1",
                status=ReviewStatus.ESCALATED,
                note="Allegation requires department confirmation.",
                corrected_stance=Stance.MIXED,
                corrected_summary="Reviewed as mixed because the article praises one response and raises one concern.",
            ),
        )
        session.commit()

    assert decision.analysis_id == analysis.id
    assert decision.status == "escalated"
    assert decision.corrected_stance == "mixed"


def test_get_latest_review_decision_returns_newest_row(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'review.db'}")
    init_db(session_factory)

    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        analysis = save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=analysis.id,
                reviewer_name="analyst-1",
                status=ReviewStatus.APPROVED,
                note="Initial review.",
            ),
        )
        latest = save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=analysis.id,
                reviewer_name="lead-1",
                status=ReviewStatus.ESCALATED,
                note="Escalate after supervisor check.",
            ),
        )
        found = get_latest_review_decision(session, analysis.id)
        session.commit()

    assert found is not None
    assert found.id == latest.id
    assert found.reviewer_name == "lead-1"


def test_review_decision_postgresql_ddl_has_audit_fields():
    ddl = str(CreateTable(ReviewDecisionRecord.__table__).compile(dialect=postgresql.dialect()))

    assert "review_decisions" in ddl
    assert "analysis_id" in ddl
    assert "reviewer_name" in ddl
    assert "corrected_summary" in ddl
```

- [ ] **Step 2: Run failing storage tests**

Run:

```powershell
python -m pytest tests/test_review_storage.py -q
```

Expected: fail because `ReviewDecisionCreate`, `ReviewStatus`, `ReviewDecisionRecord`, and helpers do not exist.

- [ ] **Step 3: Add review contracts**

In `src/tnmi/contracts.py`, add:

```python
class ReviewStatus(StrEnum):
    APPROVED = "approved"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"
    CORRECTED = "corrected"


class ReviewDecisionCreate(BaseModel):
    analysis_id: int
    reviewer_name: str = Field(min_length=1, max_length=128)
    status: ReviewStatus
    note: str = Field(default="", max_length=4000)
    corrected_stance: Stance | None = None
    corrected_relevance: GovernmentRelevance | None = None
    corrected_summary: str | None = Field(default=None, max_length=4000)
```

- [ ] **Step 4: Add review storage model and helpers**

In `src/tnmi/storage.py`, import `ReviewDecisionCreate` and add:

```python
class ReviewDecisionRecord(Base):
    __tablename__ = "review_decisions"

    id: Mapped[int] = mapped_column(ID_TYPE, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(ID_TYPE, ForeignKey("ai_analysis.id", ondelete="CASCADE"), index=True)
    reviewer_name: Mapped[str] = mapped_column(String(128), index=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    note: Mapped[str] = mapped_column(Text, default="", server_default="")
    corrected_stance: Mapped[str | None] = mapped_column(String(32), nullable=True)
    corrected_relevance: Mapped[str | None] = mapped_column(String(32), nullable=True)
    corrected_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        server_default=func.now(),
    )


def save_review_decision(session: Session, decision: ReviewDecisionCreate) -> ReviewDecisionRecord:
    record = ReviewDecisionRecord(
        analysis_id=decision.analysis_id,
        reviewer_name=decision.reviewer_name.strip(),
        status=decision.status.value,
        note=decision.note.strip(),
        corrected_stance=decision.corrected_stance.value if decision.corrected_stance else None,
        corrected_relevance=decision.corrected_relevance.value if decision.corrected_relevance else None,
        corrected_summary=decision.corrected_summary.strip() if decision.corrected_summary else None,
    )
    session.add(record)
    session.flush()
    return record


def get_latest_review_decision(session: Session, analysis_id: int) -> ReviewDecisionRecord | None:
    return session.scalar(
        select(ReviewDecisionRecord)
        .where(ReviewDecisionRecord.analysis_id == analysis_id)
        .order_by(ReviewDecisionRecord.created_at.desc(), ReviewDecisionRecord.id.desc())
        .limit(1)
    )
```

- [ ] **Step 5: Update SQL schema**

Append to `db/schema.sql`:

```sql
CREATE TABLE review_decisions (
    id BIGSERIAL PRIMARY KEY,
    analysis_id BIGINT NOT NULL REFERENCES ai_analysis(id) ON DELETE CASCADE,
    reviewer_name TEXT NOT NULL,
    status TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    corrected_stance TEXT,
    corrected_relevance TEXT,
    corrected_summary TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_review_decisions_analysis_id ON review_decisions(analysis_id);
CREATE INDEX idx_review_decisions_status ON review_decisions(status);
CREATE INDEX idx_review_decisions_reviewer_name ON review_decisions(reviewer_name);
```

- [ ] **Step 6: Run storage tests**

Run:

```powershell
python -m pytest tests/test_review_storage.py tests/test_storage.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit storage foundation**

Run:

```powershell
git add src/tnmi/contracts.py src/tnmi/storage.py db/schema.sql tests/test_review_storage.py tests/test_storage.py
git commit -m "feat: add review decision storage"
```

---

### Task 2: Dashboard Query Service

**Files:**
- Create: `src/tnmi/dashboard.py`
- Create: `tests/test_dashboard.py`

- [ ] **Step 1: Write failing dashboard service tests**

Add `tests/test_dashboard.py`:

```python
from tests.test_storage import make_analysis, make_item
from tnmi.contracts import GovernmentRelevance, ReviewDecisionCreate, ReviewStatus, Severity, Stance
from tnmi.dashboard import get_dashboard_summary, list_review_queue
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item, save_review_decision


def test_dashboard_summary_counts_analysis_and_review_status(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'dashboard.db'}")
    init_db(session_factory)

    negative = make_analysis().model_copy(
        update={
            "stance_toward_government": Stance.NEGATIVE,
            "severity": Severity.HIGH,
            "government_relevance": GovernmentRelevance.HIGH,
            "needs_human_review": True,
            "department": "transport",
            "district": "Chennai",
            "summary_english": "Negative road issue.",
        }
    )
    positive = make_analysis().model_copy(
        update={
            "stance_toward_government": Stance.POSITIVE,
            "severity": Severity.LOW,
            "needs_human_review": False,
            "department": "health",
            "district": "Madurai",
            "summary_english": "Positive health item.",
        }
    )

    with session_factory() as session:
        raw_one = save_raw_item(session, make_item().model_copy(update={"source_url": "https://example.com/one"}))
        analysis_one = save_ai_analysis(session, raw_one.id, negative, model_name="mock", prompt_version="v1")
        raw_two = save_raw_item(session, make_item().model_copy(update={"source_url": "https://example.com/two"}))
        save_ai_analysis(session, raw_two.id, positive, model_name="mock", prompt_version="v1")
        save_review_decision(
            session,
            ReviewDecisionCreate(
                analysis_id=analysis_one.id,
                reviewer_name="analyst-1",
                status=ReviewStatus.ESCALATED,
                note="Needs department confirmation.",
            ),
        )
        summary = get_dashboard_summary(session)
        session.commit()

    assert summary["total_items"] == 2
    assert summary["total_analyses"] == 2
    assert summary["needs_human_review"] == 1
    assert summary["reviewed"] == 1
    assert summary["stance_counts"] == {"negative": 1, "positive": 1}
    assert summary["severity_counts"]["high"] == 1
    assert summary["department_counts"]["transport"] == 1
    assert summary["district_counts"]["Chennai"] == 1


def test_review_queue_prioritizes_unreviewed_high_severity_items(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'dashboard.db'}")
    init_db(session_factory)

    high = make_analysis().model_copy(
        update={
            "stance_toward_government": Stance.NEGATIVE,
            "severity": Severity.CRITICAL,
            "needs_human_review": True,
            "confidence": 0.55,
            "summary_english": "Critical allegation.",
        }
    )

    with session_factory() as session:
        raw = save_raw_item(session, make_item())
        analysis = save_ai_analysis(session, raw.id, high, model_name="mock", prompt_version="v1")
        queue = list_review_queue(session, limit=10)
        session.commit()

    assert queue[0]["analysis_id"] == analysis.id
    assert queue[0]["review_status"] == "pending"
    assert queue[0]["severity"] == "critical"
    assert queue[0]["stance"] == "negative"
```

- [ ] **Step 2: Run failing dashboard tests**

Run:

```powershell
python -m pytest tests/test_dashboard.py -q
```

Expected: fail because `tnmi.dashboard` does not exist.

- [ ] **Step 3: Implement dashboard query service**

Create `src/tnmi/dashboard.py`:

```python
from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from tnmi.storage import AIAnalysisRecord, RawItemRecord, ReviewDecisionRecord, get_latest_review_decision


def _count_values(values: list[str | None]) -> dict[str, int]:
    return dict(Counter(value for value in values if value))


def get_dashboard_summary(session: Session) -> dict[str, Any]:
    analyses = session.scalars(select(AIAnalysisRecord)).all()
    reviewed_analysis_ids = set(session.scalars(select(ReviewDecisionRecord.analysis_id)).all())
    return {
        "total_items": session.scalar(select(func.count()).select_from(RawItemRecord)) or 0,
        "total_analyses": len(analyses),
        "needs_human_review": sum(1 for row in analyses if row.needs_human_review),
        "reviewed": len(reviewed_analysis_ids),
        "pending_review": sum(1 for row in analyses if row.needs_human_review and row.id not in reviewed_analysis_ids),
        "stance_counts": _count_values([row.stance_toward_government for row in analyses]),
        "severity_counts": _count_values([row.severity for row in analyses]),
        "department_counts": _count_values([row.department for row in analyses]),
        "district_counts": _count_values([row.district for row in analyses]),
    }


def _queue_query() -> Select[tuple[AIAnalysisRecord, RawItemRecord]]:
    severity_rank = case(
        (AIAnalysisRecord.severity == "critical", 4),
        (AIAnalysisRecord.severity == "high", 3),
        (AIAnalysisRecord.severity == "medium", 2),
        (AIAnalysisRecord.severity == "low", 1),
        else_=0,
    )
    return (
        select(AIAnalysisRecord, RawItemRecord)
        .join(RawItemRecord, RawItemRecord.id == AIAnalysisRecord.raw_item_id)
        .where(AIAnalysisRecord.needs_human_review.is_(True))
        .order_by(
            severity_rank.desc(),
            AIAnalysisRecord.confidence.asc(),
            AIAnalysisRecord.created_at.desc(),
            AIAnalysisRecord.id.desc(),
        )
    )


def list_review_queue(session: Session, *, limit: int = 50) -> list[dict[str, Any]]:
    bounded_limit = max(1, min(limit, 200))
    rows = session.execute(_queue_query().limit(bounded_limit)).all()
    queue: list[dict[str, Any]] = []
    for analysis, item in rows:
        latest = get_latest_review_decision(session, analysis.id)
        if latest is not None:
            continue
        queue.append(
            {
                "analysis_id": analysis.id,
                "raw_item_id": item.id,
                "review_status": "pending",
                "source_name": item.source_name,
                "source_url": item.source_url,
                "title": item.title,
                "published_at": item.published_at,
                "language": item.language,
                "stance": analysis.stance_toward_government,
                "severity": analysis.severity,
                "department": analysis.department,
                "district": analysis.district,
                "summary": analysis.summary_english or analysis.summary_original,
                "confidence": analysis.confidence,
                "evidence": analysis.evidence_quotes_english or analysis.evidence_quotes_original,
            }
        )
    return queue
```

- [ ] **Step 4: Run dashboard tests**

Run:

```powershell
python -m pytest tests/test_dashboard.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit dashboard query service**

Run:

```powershell
git add src/tnmi/dashboard.py tests/test_dashboard.py
git commit -m "feat: add dashboard summary service"
```

---

### Task 3: Review and Dashboard JSON APIs

**Files:**
- Modify: `src/tnmi/config.py`
- Modify: `apps/api/main.py`
- Modify: `tests/test_api.py`
- Modify: `.env.example`

- [ ] **Step 1: Write failing API tests**

Append to `tests/test_api.py`:

```python
from tnmi.contracts import ReviewDecisionCreate, ReviewStatus
from tnmi.storage import save_review_decision


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
```

- [ ] **Step 2: Run failing API tests**

Run:

```powershell
python -m pytest tests/test_api.py -q
```

Expected: fail because the new endpoints do not exist.

- [ ] **Step 3: Add optional operator token setting**

In `src/tnmi/config.py`, add to `Settings`:

```python
operator_api_token: str | None = None
```

In `.env.example`, add:

```text
OPERATOR_API_TOKEN=
```

- [ ] **Step 4: Implement API endpoints**

In `apps/api/main.py`, import:

```python
from fastapi import Depends, Header, HTTPException

from tnmi.contracts import ReviewDecisionCreate
from tnmi.dashboard import get_dashboard_summary, list_review_queue
from tnmi.storage import ReviewDecisionRecord, get_latest_review_decision, save_review_decision
```

Add helpers:

```python
def require_operator(x_tnmi_operator_token: str | None = Header(default=None)) -> None:
    token = Settings().operator_api_token
    if token and x_tnmi_operator_token != token:
        raise HTTPException(status_code=401, detail="Operator token required")
```

Add routes:

```python
@app.get("/dashboard/summary", dependencies=[Depends(require_operator)])
def dashboard_summary() -> dict[str, object]:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        return get_dashboard_summary(session)


@app.get("/review/queue", dependencies=[Depends(require_operator)])
def review_queue(limit: int = 50) -> list[dict[str, object]]:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        return list_review_queue(session, limit=limit)


@app.post("/review/decisions", dependencies=[Depends(require_operator)])
def create_review_decision(decision: ReviewDecisionCreate) -> dict[str, object]:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        record = save_review_decision(session, decision)
        session.commit()
        return _review_decision_payload(record)


@app.get("/review/decisions/{analysis_id}", dependencies=[Depends(require_operator)])
def latest_review_decision(analysis_id: int) -> dict[str, object]:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        record = get_latest_review_decision(session, analysis_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Review decision not found")
        return _review_decision_payload(record)
```

- [ ] **Step 5: Add review payload helper**

In `apps/api/main.py`, add:

```python
def _review_decision_payload(record: ReviewDecisionRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "analysis_id": record.analysis_id,
        "reviewer_name": record.reviewer_name,
        "status": record.status,
        "note": record.note,
        "corrected_stance": record.corrected_stance,
        "corrected_relevance": record.corrected_relevance,
        "corrected_summary": record.corrected_summary,
        "created_at": record.created_at,
    }
```

- [ ] **Step 6: Run API tests**

Run:

```powershell
python -m pytest tests/test_api.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit JSON APIs**

Run:

```powershell
git add src/tnmi/config.py apps/api/main.py tests/test_api.py .env.example
git commit -m "feat: expose review dashboard APIs"
```

---

### Task 4: Internal HTML Operator Console

**Files:**
- Create: `apps/api/templates/dashboard.html`
- Create: `apps/api/static/dashboard.css`
- Modify: `apps/api/main.py`
- Modify: `tests/test_api.py`

- [ ] **Step 1: Write failing HTML console test**

Append to `tests/test_api.py`:

```python
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
    assert "TN Media Intelligence" in response.text
    assert "Needs review." in response.text
    assert "Pending Review" in response.text
```

- [ ] **Step 2: Run failing HTML test**

Run:

```powershell
python -m pytest tests/test_api.py::test_dashboard_html_renders_summary_and_review_queue -q
```

Expected: fail because `/dashboard` does not exist.

- [ ] **Step 3: Add dashboard HTML route**

In `apps/api/main.py`, add:

```python
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_operator)])
def dashboard_page(request: Request) -> HTMLResponse:
    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        summary = get_dashboard_summary(session)
        queue = list_review_queue(session, limit=50)
    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "summary": summary,
            "queue": queue,
        },
    )
```

- [ ] **Step 4: Create dashboard template**

Create `apps/api/templates/dashboard.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>TN Media Intelligence</title>
    <link rel="stylesheet" href="/static/dashboard.css">
  </head>
  <body>
    <main class="shell">
      <header class="topbar">
        <div>
          <p class="eyebrow">Confidential Operator Console</p>
          <h1>TN Media Intelligence</h1>
        </div>
        <div class="status-pill">Newspaper Review</div>
      </header>

      <section class="metric-grid" aria-label="Dashboard summary">
        <article><span>Total Items</span><strong>{{ summary.total_items }}</strong></article>
        <article><span>Analyses</span><strong>{{ summary.total_analyses }}</strong></article>
        <article><span>Pending Review</span><strong>{{ summary.pending_review }}</strong></article>
        <article><span>Reviewed</span><strong>{{ summary.reviewed }}</strong></article>
      </section>

      <section class="split">
        <div>
          <h2>Stance</h2>
          <dl>
            {% for label, count in summary.stance_counts.items() %}
              <div><dt>{{ label }}</dt><dd>{{ count }}</dd></div>
            {% endfor %}
          </dl>
        </div>
        <div>
          <h2>Severity</h2>
          <dl>
            {% for label, count in summary.severity_counts.items() %}
              <div><dt>{{ label }}</dt><dd>{{ count }}</dd></div>
            {% endfor %}
          </dl>
        </div>
      </section>

      <section>
        <h2>Pending Review</h2>
        <div class="queue">
          {% for item in queue %}
            <article class="queue-item">
              <div class="queue-meta">
                <span>{{ item.source_name }}</span>
                <span>{{ item.department }}</span>
                <span>{{ item.district }}</span>
                <span>{{ item.severity }}</span>
              </div>
              <h3>{{ item.title or "Untitled item" }}</h3>
              <p>{{ item.summary }}</p>
              <a href="{{ item.source_url }}" rel="noreferrer" target="_blank">Open source</a>
            </article>
          {% else %}
            <p class="empty">No pending review items.</p>
          {% endfor %}
        </div>
      </section>
    </main>
  </body>
</html>
```

- [ ] **Step 5: Create dashboard CSS**

Create `apps/api/static/dashboard.css`:

```css
:root {
  color-scheme: light;
  font-family: Inter, "Segoe UI", Arial, sans-serif;
  background: #f4f6f8;
  color: #1d2733;
}

body {
  margin: 0;
  background: #f4f6f8;
}

.shell {
  width: min(1180px, calc(100% - 32px));
  margin: 0 auto;
  padding: 28px 0 48px;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding-bottom: 24px;
}

.eyebrow {
  margin: 0 0 6px;
  color: #5b6776;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

h1, h2, h3, p {
  margin-top: 0;
}

h1 {
  margin-bottom: 0;
  font-size: clamp(1.8rem, 4vw, 2.7rem);
}

h2 {
  font-size: 1rem;
}

.status-pill {
  border: 1px solid #c9d2dc;
  border-radius: 999px;
  padding: 8px 12px;
  color: #334155;
  background: #ffffff;
}

.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
}

.metric-grid article,
.split > div,
.queue-item {
  border: 1px solid #d7dee7;
  border-radius: 8px;
  background: #ffffff;
}

.metric-grid article {
  padding: 16px;
}

.metric-grid span {
  display: block;
  color: #5b6776;
  font-size: 0.85rem;
}

.metric-grid strong {
  display: block;
  margin-top: 8px;
  font-size: 2rem;
}

.split {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
  margin: 16px 0 24px;
}

.split > div {
  padding: 16px;
}

dl {
  margin: 0;
}

dl div {
  display: flex;
  justify-content: space-between;
  padding: 8px 0;
  border-top: 1px solid #eef2f6;
}

dl div:first-child {
  border-top: 0;
}

dt {
  text-transform: capitalize;
  color: #4b5563;
}

dd {
  margin: 0;
  font-weight: 700;
}

.queue {
  display: grid;
  gap: 12px;
}

.queue-item {
  padding: 16px;
}

.queue-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 10px;
  color: #5b6776;
  font-size: 0.82rem;
}

.queue-meta span {
  border: 1px solid #d7dee7;
  border-radius: 999px;
  padding: 4px 8px;
  background: #f8fafc;
}

.queue-item h3 {
  margin-bottom: 8px;
  font-size: 1rem;
}

.queue-item p {
  color: #334155;
  line-height: 1.5;
}

a {
  color: #0f766e;
  font-weight: 700;
}

.empty {
  color: #5b6776;
}

@media (max-width: 760px) {
  .topbar,
  .split {
    grid-template-columns: 1fr;
    display: grid;
  }

  .metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
```

- [ ] **Step 6: Run HTML/API tests**

Run:

```powershell
python -m pytest tests/test_api.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit HTML console**

Run:

```powershell
git add apps/api/main.py apps/api/templates/dashboard.html apps/api/static/dashboard.css tests/test_api.py
git commit -m "feat: add operator dashboard page"
```

---

### Task 5: Confidential Endpoint Guard

**Files:**
- Modify: `apps/api/main.py`
- Modify: `tests/test_api.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing auth tests**

Append to `tests/test_api.py`:

```python
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
```

- [ ] **Step 2: Run failing auth test**

Run:

```powershell
python -m pytest tests/test_api.py::test_operator_token_blocks_confidential_dashboard_endpoint -q
```

Expected: fail if the guard has not been wired to all confidential endpoints.

- [ ] **Step 3: Ensure guard applies to confidential routes**

Confirm these routes include `dependencies=[Depends(require_operator)]`:

```python
@app.get("/dashboard", response_class=HTMLResponse, dependencies=[Depends(require_operator)])
@app.get("/dashboard/summary", dependencies=[Depends(require_operator)])
@app.get("/review/queue", dependencies=[Depends(require_operator)])
@app.post("/review/decisions", dependencies=[Depends(require_operator)])
@app.get("/review/decisions/{analysis_id}", dependencies=[Depends(require_operator)])
```

- [ ] **Step 4: Document confidential dashboard usage**

In `README.md`, add:

```markdown
## Operator Dashboard

The internal review console is available at `http://127.0.0.1:8000/dashboard` when the API is running.

For confidential deployments, set `OPERATOR_API_TOKEN` and send the token as `X-TNMI-Operator-Token` for dashboard and review APIs. This is a first local guard only; production deployments still need SSO/RBAC, private networking, audit logs, and managed secrets.
```

- [ ] **Step 5: Run API tests**

Run:

```powershell
python -m pytest tests/test_api.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit endpoint guard documentation**

Run:

```powershell
git add apps/api/main.py tests/test_api.py README.md
git commit -m "docs: document confidential dashboard guard"
```

---

### Task 6: Final Verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Validate Docker Compose**

Run:

```powershell
docker compose config
```

Expected: valid Compose output.

- [ ] **Step 3: Check whitespace**

Run:

```powershell
git diff --check master...HEAD
```

Expected: no output.

- [ ] **Step 4: Inspect branch status**

Run:

```powershell
git status --short --branch
```

Expected: clean branch `feature/review-dashboard`.

- [ ] **Step 5: Record Turbovec decision in final notes**

Use [RyanCodrai/turbovec](https://github.com/RyanCodrai/turbovec) later for optional private semantic search experiments, not for this milestone. It is attractive for air-gapped, compressed vector retrieval, but Milestone 2 needs trusted human review workflow more than approximate vector search.

---

## Self-Review

- Spec coverage: This plan implements the next production layer after daily newspaper analysis: human review, dashboard summaries, confidential route guarding, and operator documentation. It deliberately excludes X, Instagram, video, and vector retrieval because those are separate production subsystems.
- Placeholder scan: No task relies on placeholder implementation text.
- Type consistency: `ReviewDecisionCreate`, `ReviewStatus`, `ReviewDecisionRecord`, `get_dashboard_summary`, and `list_review_queue` are introduced before later tasks use them.
