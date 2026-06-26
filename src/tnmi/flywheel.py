"""The self-improving flywheel — one orchestrated pass over the whole loop.

    gather → classify (already done by ingest) → validate → label
          → train → evaluate on frozen gold → gated promote

Each step is also callable on its own; ``run_flywheel`` chains them with safe
defaults. Nothing in this loop can degrade the live system: a worse candidate
is registered but the promotion gate (``tnmi.registry.promote_if_better``)
keeps the incumbent live. Humans stay in the loop through the review queue —
disagreements route there, and corrections come back as gold labels on the
next pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    Sentiment,
    Severity,
    Stance,
)
from tnmi.eval import SupportsAnalyze, evaluate_classifier
from tnmi.labeling import (
    dataset_stats,
    promote_corrections_to_gold,
    record_bronze_from_analysis,
)
from tnmi.registry import (
    PromotionDecision,
    get_live_model,
    list_models,
    primary_metric_from_report,
    promote_if_better,
    register_model,
)
from tnmi.storage import AIAnalysisRecord, RawItemRecord
from tnmi.training import Trainer, TrainingResult, build_distillation_dataset
from tnmi.validation import validate_analysis


def model_health(session: Session, *, model_name: str = "tvk-tamil-classifier") -> dict[str, Any]:
    """Operator-facing readout of the learning loop — label tallies, the live
    model and its gold-test metric, and the latest registered candidate. Lets
    the dashboard SHOW the flywheel working (or show it has not run yet)."""
    stats = dataset_stats(session)
    by_tier = stats.get("by_tier", {})
    live = get_live_model(session, model_name)
    models = list_models(session, model_name)
    latest = models[0] if models else None
    return {
        "gold": by_tier.get("gold", 0),
        "silver": by_tier.get("silver", 0),
        "bronze": by_tier.get("bronze", 0),
        "total_labels": stats.get("total_labels", 0),
        "registered_models": len(models),
        "live_version": live.version if live else "",
        "live_metric": round(live.primary_metric, 4) if live else None,
        "live_eval_examples": live.eval_examples if live else 0,
        "latest_version": latest.version if latest else "",
        "latest_metric": round(latest.primary_metric, 4) if latest else None,
        "has_run": stats.get("total_labels", 0) > 0,
    }


def _record_to_analysis(record: AIAnalysisRecord) -> AIAnalysis:
    """Rehydrate the stored analysis into the contract model so the validator
    can read its fields uniformly."""
    return AIAnalysis(
        government_relevance=GovernmentRelevance(record.government_relevance),
        stance_toward_government=Stance(record.stance_toward_government),
        tvk_relevance=GovernmentRelevance(record.tvk_relevance) if record.tvk_relevance else None,
        tvk_portrayal=Stance(record.tvk_portrayal) if record.tvk_portrayal else None,
        sentiment=Sentiment(record.sentiment),
        target=record.target or "",
        political_actors=list(record.political_actors or []),
        department=record.department or "general",
        district=record.district or "unspecified",
        scheme=record.scheme,
        topic=record.topic or "news item",
        issue_category=record.issue_category or "general",
        people_issue=bool(record.people_issue),
        public_issue=record.public_issue or "",
        severity=Severity(record.severity),
        summary_original=record.summary_original or "",
        summary_english=record.summary_english or "",
        confidence=float(record.confidence or 0.0),
        needs_human_review=bool(record.needs_human_review),
    )


def latest_analyses(session: Session, *, limit: int | None = None) -> list[AIAnalysisRecord]:
    """Latest analysis per raw item (highest id wins per item)."""
    rows = session.execute(
        select(AIAnalysisRecord).order_by(
            AIAnalysisRecord.raw_item_id.asc(), AIAnalysisRecord.id.desc()
        )
    ).scalars().all()
    latest: dict[int, AIAnalysisRecord] = {}
    for record in rows:
        latest.setdefault(record.raw_item_id, record)
    out = list(latest.values())
    out.sort(key=lambda r: r.raw_item_id)
    if limit is not None:
        out = out[:limit]
    return out


@dataclass
class FlywheelReport:
    gold_promoted: int = 0
    bronze_written: int = 0
    items_validated: int = 0
    silver_written: int = 0
    routed_to_review: int = 0
    training_examples: int = 0
    trained_version: str = ""
    eval_total: int = 0
    primary_metric: float = 0.0
    promotion: PromotionDecision | None = None
    label_stats: dict[str, Any] = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "gold_promoted": self.gold_promoted,
            "bronze_written": self.bronze_written,
            "items_validated": self.items_validated,
            "silver_written": self.silver_written,
            "routed_to_review": self.routed_to_review,
            "training_examples": self.training_examples,
            "trained_version": self.trained_version,
            "eval_total": self.eval_total,
            "primary_metric": round(self.primary_metric, 4),
            "promotion": (
                {
                    "promoted": self.promotion.promoted,
                    "reason": self.promotion.reason,
                    "candidate_metric": self.promotion.candidate_metric,
                    "incumbent_metric": self.promotion.incumbent_metric,
                }
                if self.promotion
                else None
            ),
            "label_stats": self.label_stats,
        }


def harvest_labels(
    session: Session,
    *,
    teacher: SupportsAnalyze | None = None,
    high_confidence_threshold: float = 0.85,
    limit: int | None = None,
) -> FlywheelReport:
    """Steps 1–3 of the loop: gold from human corrections, bronze from every
    latest analysis, silver via teacher-agreement / confidence routing."""
    report = FlywheelReport()
    report.gold_promoted = promote_corrections_to_gold(session)

    raw_by_id: dict[int, RawItemRecord] = {}
    if teacher is not None:
        from tnmi.eval import _raw_to_item  # reuse the same rehydration

    for record in latest_analyses(session, limit=limit):
        report.bronze_written += len(record_bronze_from_analysis(session, record))
        teacher_item = None
        if teacher is not None:
            raw = raw_by_id.get(record.raw_item_id) or session.get(RawItemRecord, record.raw_item_id)
            if raw is not None:
                raw_by_id[record.raw_item_id] = raw
                teacher_item = _raw_to_item(raw)
        result = validate_analysis(
            session,
            raw_item_id=record.raw_item_id,
            student_analysis=_record_to_analysis(record),
            student_model=record.model_name or "unknown",
            analysis_id=record.id,
            teacher=teacher if teacher_item is not None else None,
            teacher_item=teacher_item,
            high_confidence_threshold=high_confidence_threshold,
        )
        report.items_validated += 1
        report.silver_written += result.silver_written
        if result.needs_review:
            report.routed_to_review += 1

    report.label_stats = dataset_stats(session)
    return report


def train_and_gate(
    session: Session,
    *,
    trainer: Trainer,
    candidate_analyzer: SupportsAnalyze,
    model_name: str = "tvk-tamil-classifier",
    base_model: str = "stub-base",
    output_dir: str = "artifacts",
    min_delta: float = 0.0,
) -> tuple[TrainingResult | None, FlywheelReport]:
    """Steps 4–6: train on curated labels, evaluate the candidate on the frozen
    gold test set, register, and promote only if it beats the live model.

    ``candidate_analyzer`` is whatever serves the candidate's predictions — on
    the GPU box, an adapter that loads the freshly trained artifact; in dry
    runs, the current analyzer cascade.
    """
    report = FlywheelReport()
    examples = build_distillation_dataset(session)
    report.training_examples = len(examples)
    if not examples:
        return None, report

    result = trainer.train(
        examples, model_name=model_name, base_model=base_model, output_dir=output_dir
    )
    report.trained_version = result.version

    eval_report = evaluate_classifier(session, candidate_analyzer)
    report.eval_total = eval_report.total
    report.primary_metric = primary_metric_from_report(eval_report)

    register_model(
        session,
        model_name=model_name,
        version=result.version,
        primary_metric=report.primary_metric,
        metrics=eval_report.as_dict(),
        eval_examples=eval_report.total,
        artifact_uri=result.artifact_uri,
        notes=f"trainer={result.metadata.get('trainer', '?')} base={result.base_model}",
    )
    if eval_report.total == 0:
        # No gold yardstick yet — never promote blind.
        report.promotion = PromotionDecision(
            promoted=False,
            reason="no gold test labels yet — promotion requires a measurable gate",
            candidate_metric=report.primary_metric,
            incumbent_metric=None,
        )
        return result, report

    report.promotion = promote_if_better(
        session, model_name=model_name, version=result.version, min_delta=min_delta
    )
    return result, report


def run_flywheel(
    session: Session,
    *,
    trainer: Trainer,
    candidate_analyzer: SupportsAnalyze,
    teacher: SupportsAnalyze | None = None,
    model_name: str = "tvk-tamil-classifier",
    base_model: str = "stub-base",
    output_dir: str = "artifacts",
    min_delta: float = 0.0,
    high_confidence_threshold: float = 0.85,
    validate_limit: int | None = None,
) -> FlywheelReport:
    """One full pass. Safe to run nightly; every step is idempotent."""
    harvest = harvest_labels(
        session,
        teacher=teacher,
        high_confidence_threshold=high_confidence_threshold,
        limit=validate_limit,
    )
    _result, gate = train_and_gate(
        session,
        trainer=trainer,
        candidate_analyzer=candidate_analyzer,
        model_name=model_name,
        base_model=base_model,
        output_dir=output_dir,
        min_delta=min_delta,
    )
    # Merge the two halves into one report.
    harvest.training_examples = gate.training_examples
    harvest.trained_version = gate.trained_version
    harvest.eval_total = gate.eval_total
    harvest.primary_metric = gate.primary_metric
    harvest.promotion = gate.promotion
    harvest.label_stats = dataset_stats(session)
    return harvest
