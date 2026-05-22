# TN Media Intelligence Engineering Rules

This repository must be treated as a confidential, production-grade public media intelligence product. Every code change must meet a senior engineering quality bar.

## Non-Negotiable Rules

1. Code quality and standards come first.
2. Maintainability is required, not optional.
3. Use best-practice, standard, boring production patterns unless there is a documented reason not to.
4. Every meaningful change needs appropriate tests, security consideration, and verification.
5. Work must look like it was done by an experienced senior engineer: clear, careful, reviewed, and easy to operate.

## Required Engineering Behavior

- Keep changes scoped to the requested feature or fix.
- Prefer explicit contracts, typed models, small focused modules, and clear data flow.
- Do not add clever abstractions unless they remove real complexity or match an existing project pattern.
- Do not use unauthorized scraping, unsafe shortcuts, hidden credentials, or unreviewed access paths.
- Preserve raw evidence and auditability for ingestion, AI analysis, and human review decisions.
- Treat Tamil, English, and mixed-language content carefully; do not erase original text or evidence.
- Keep external-source connectors isolated behind interfaces so API/provider changes do not rewrite the product.
- Prefer idempotent, retry-safe pipeline behavior.
- Fail clearly when credentials, config, quotas, or approved data access are missing.

## Testing And Verification

Before a change is considered complete, run the smallest relevant tests during development and the full verification before finishing:

```powershell
python -m pytest -q
docker compose config
git diff --check HEAD
git status --short --branch
```

Additional checks are required when relevant:

- Security-sensitive ingestion changes need tests for blocked URLs, invalid inputs, credentials, and access boundaries.
- Pipeline changes need idempotency and rerun tests.
- AI analysis changes need schema, prompt, mock-provider, and failure-path tests.
- Dashboard or frontend changes need browser verification at desktop and mobile widths.
- Database changes need SQLAlchemy and PostgreSQL DDL coverage.

## Security Baseline

- Use official APIs or approved providers only.
- Never commit secrets or tokens.
- Bind local services to loopback unless there is a documented reason.
- Protect confidential dashboard and review endpoints.
- Keep audit trails for human review, escalation, and corrections.
- Avoid exposing unnecessary personal data in reports.
- Any sensitive allegation must remain evidence-backed and require human review.

## Definition Of Done

A task is done only when:

- the implementation is complete and cohesive,
- relevant docs/config are updated,
- tests cover the important behavior and failure paths,
- verification commands pass,
- git status is clean or intentionally explained,
- the final report states what changed and what was verified.
