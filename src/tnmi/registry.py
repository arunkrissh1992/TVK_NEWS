"""Model registry and the promotion gate.

Every trained or evaluated model version is recorded with its gold-test
scorecard. A candidate is promoted to *live* only when it beats the current live
model's ``primary_metric`` by at least ``min_delta``. This is the single rule
that makes continuous training safe: a regressed or collapsed model can be
trained and registered, but it will never serve traffic because it can't clear
the gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tnmi.eval import EvalReport
from tnmi.storage import ModelRegistryRecord


def primary_metric_from_report(report: EvalReport) -> float:
    """The single number the gate compares.

    Mean macro-F1 across fields — more honest than raw accuracy when classes are
    imbalanced (most news is neutral/low-relevance, so accuracy alone is easy to
    game). Falls back to overall accuracy when no per-field metrics exist.
    """
    f1s = [m.macro_f1 for m in report.per_field.values()]
    if f1s:
        return sum(f1s) / len(f1s)
    return report.overall_accuracy


def register_model(
    session: Session,
    *,
    model_name: str,
    version: str,
    primary_metric: float,
    kind: str = "classifier",
    metrics: dict[str, Any] | None = None,
    eval_examples: int = 0,
    artifact_uri: str = "",
    notes: str = "",
) -> ModelRegistryRecord:
    """Record (or update) a model version and its scorecard. Does not promote."""
    existing = session.scalar(
        select(ModelRegistryRecord).where(
            ModelRegistryRecord.model_name == model_name,
            ModelRegistryRecord.version == version,
        )
    )
    if existing is not None:
        existing.primary_metric = primary_metric
        existing.kind = kind
        existing.metrics_json = metrics or {}
        existing.eval_examples = eval_examples
        existing.artifact_uri = artifact_uri
        existing.notes = notes
        session.flush()
        return existing

    record = ModelRegistryRecord(
        model_name=model_name,
        version=version,
        kind=kind,
        metrics_json=metrics or {},
        primary_metric=primary_metric,
        eval_examples=eval_examples,
        artifact_uri=artifact_uri,
        notes=notes,
        is_live=False,
    )
    session.add(record)
    session.flush()
    return record


def get_live_model(session: Session, model_name: str) -> ModelRegistryRecord | None:
    return session.scalar(
        select(ModelRegistryRecord).where(
            ModelRegistryRecord.model_name == model_name,
            ModelRegistryRecord.is_live.is_(True),
        )
    )


def list_models(session: Session, model_name: str | None = None) -> list[ModelRegistryRecord]:
    stmt = select(ModelRegistryRecord)
    if model_name is not None:
        stmt = stmt.where(ModelRegistryRecord.model_name == model_name)
    stmt = stmt.order_by(ModelRegistryRecord.created_at.desc(), ModelRegistryRecord.id.desc())
    return list(session.execute(stmt).scalars().all())


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    candidate_metric: float
    incumbent_metric: float | None


def promote_if_better(
    session: Session,
    *,
    model_name: str,
    version: str,
    min_delta: float = 0.0,
) -> PromotionDecision:
    """Promote a candidate to live only if it beats the incumbent by min_delta.

    The first model for a name is promoted unconditionally (nothing to beat).
    Promotion demotes the previous live model. A losing candidate stays
    registered but never goes live.
    """
    candidate = session.scalar(
        select(ModelRegistryRecord).where(
            ModelRegistryRecord.model_name == model_name,
            ModelRegistryRecord.version == version,
        )
    )
    if candidate is None:
        raise ValueError(f"No registered model {model_name!r} version {version!r}")

    incumbent = get_live_model(session, model_name)

    if incumbent is None:
        candidate.is_live = True
        session.flush()
        return PromotionDecision(
            promoted=True,
            reason="first model for this name — promoted unconditionally",
            candidate_metric=candidate.primary_metric,
            incumbent_metric=None,
        )

    if incumbent.id == candidate.id:
        return PromotionDecision(
            promoted=True,
            reason="candidate is already live",
            candidate_metric=candidate.primary_metric,
            incumbent_metric=incumbent.primary_metric,
        )

    threshold = incumbent.primary_metric + min_delta
    if candidate.primary_metric >= threshold:
        incumbent.is_live = False
        candidate.is_live = True
        session.flush()
        return PromotionDecision(
            promoted=True,
            reason=(
                f"candidate {candidate.primary_metric:.4f} ≥ "
                f"incumbent {incumbent.primary_metric:.4f} + {min_delta:.4f}"
            ),
            candidate_metric=candidate.primary_metric,
            incumbent_metric=incumbent.primary_metric,
        )

    return PromotionDecision(
        promoted=False,
        reason=(
            f"candidate {candidate.primary_metric:.4f} < "
            f"incumbent {incumbent.primary_metric:.4f} + {min_delta:.4f} — kept incumbent"
        ),
        candidate_metric=candidate.primary_metric,
        incumbent_metric=incumbent.primary_metric,
    )
