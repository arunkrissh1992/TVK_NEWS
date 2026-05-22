# Engineering Quality Bar

This document defines the standard for all future work on the Tamil Nadu Public Media Intelligence product.

## Purpose

The product handles confidential public media intelligence for government use. The codebase must therefore be reliable, secure, auditable, and maintainable from the beginning. Prototype speed is useful, but it must not come from poor architecture, weak tests, unsafe data access, or unclear behavior.

## Senior-Level Coding Standard

All code should be written as if another senior engineer will operate, debug, and extend it six months later.

Required qualities:

- Clear module boundaries.
- Explicit input and output contracts.
- Typed Pydantic models or dataclasses where structured data crosses boundaries.
- Small functions with one clear purpose.
- Boring, standard library or established dependency usage where possible.
- Useful names that describe domain meaning.
- No hidden global state unless it is configuration or an intentional singleton.
- No broad exception swallowing without counters, logs, or clear failure handling.
- No copy-paste architecture.
- No unrelated refactors in feature branches.

## Maintainability Rules

The system is expected to grow from newspapers to X, Instagram/video, retrieval, dashboards, reports, and alerts. Every implementation must preserve that growth path.

Rules:

- New source types must normalize into the common item contract.
- Source connectors must stay isolated from storage, AI, and reporting internals.
- Cursors and checkpoints must be durable and reusable.
- AI providers must stay behind analyzer interfaces.
- Report generation must use stored evidence, not ad hoc recomputation.
- Database changes must be reflected in SQLAlchemy models, schema SQL, and tests.
- Config files must be safe by default, with risky production sources inactive until approved.

## Testing Standard

Testing must scale with risk. High-risk code needs stronger tests.

Minimum expectations:

- Unit tests for contracts, parsing, normalization, and helpers.
- Pipeline tests for idempotency, reruns, rollback, and failure counters.
- Storage tests for uniqueness, foreign keys, JSON fields, and PostgreSQL DDL.
- API tests for public and confidential endpoints.
- CLI tests for help output, invalid inputs, and missing credentials.
- Mock clients for external services; no live API dependency in normal tests.
- Browser checks for meaningful UI work.

Any code touching ingestion, external URLs, credentials, AI classification, review workflow, reports, or security must include failure-path tests.

## Security Standard

Security is a product requirement, not a later hardening task.

Rules:

- Use official APIs and approved data providers only.
- No unauthorized scraping in production code.
- No secrets in source control.
- No private-message collection.
- No citizen blacklists.
- No unbounded URL fetching.
- Validate and normalize external inputs before storage or network access.
- Keep raw evidence and AI conclusions auditable.
- Human review is required for allegations, sensitive claims, low confidence outputs, and escalation decisions.
- Local development services should bind to `127.0.0.1`.
- Confidential endpoints need authentication controls, and production still requires SSO/RBAC, private networking, managed secrets, and audit logs.

## AI And Evidence Standard

AI outputs must be useful, but never treated as ground truth without evidence.

Rules:

- Preserve original Tamil, English, or mixed-language text.
- Store evidence quotes with analysis.
- Distinguish sentiment from stance toward the Tamil Nadu Government.
- Mark allegations and sensitive claims for human review.
- Keep prompt versions and model names with analysis rows.
- Use strict structured outputs where possible.
- Do not infer beyond available evidence.
- Reports must include confidence and coverage limitations where relevant.

## Review Standard

Before finishing a feature:

1. Read the diff as a reviewer.
2. Check for security regressions.
3. Check for maintainability problems.
4. Check if the tests prove the important behavior.
5. Run verification commands.
6. Document the result clearly.

Recommended verification:

```powershell
python -m pytest -q
docker compose config
git diff --check HEAD
git status --short --branch
```

## Quality Gate

Do not call work complete if any of the following are true:

- tests are missing for risky behavior,
- a source connector bypasses official or approved access,
- credentials are required but missing behavior is unclear,
- database schema and ORM models disagree,
- AI output is not evidence-backed,
- reruns can duplicate data,
- confidential endpoints are exposed without a guard,
- frontend text overlaps or breaks on mobile,
- verification has not been run.

The goal is simple: every phase should feel like senior, production-grade engineering work.
