"""Teacher→student validation with confidence routing.

This is the safety valve that keeps the flywheel from eating its own tail. A
cheap *student* model produces a label; an independent, stronger *teacher* model
re-judges the same article. The two are never the same model, so agreement is
real corroboration, not a model agreeing with itself.

Routing:
  * agree  → write a SILVER label (trusted enough for bulk training).
  * disagree → DON'T trust either; flag the item for human review, which later
               becomes a GOLD label via ``labeling.promote_corrections_to_gold``.

With no teacher available, fall back to the student's own confidence: only
high-confidence outputs become silver; the rest stay bronze and go to review.
This is active learning — humans spend their time only on what the machines are
unsure about.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field

from sqlalchemy.orm import Session

from tnmi.contracts import LABEL_FIELDS, AIAnalysis, LabelProvenance, LabelTier, NormalizedItem
from tnmi.eval import SupportsAnalyze, extract_field_value
from tnmi.labeling import record_label
from tnmi.storage import AIAnalysisRecord


DEFAULT_HIGH_CONFIDENCE = 0.85


@dataclass
class FieldVerdict:
    field: str
    student_value: str
    teacher_value: str | None
    agreed: bool
    tier_written: str | None  # "silver" when promoted, else None


@dataclass
class ValidationResult:
    raw_item_id: int
    agreed: list[str] = dataclass_field(default_factory=list)
    disagreed: list[str] = dataclass_field(default_factory=list)
    silver_written: int = 0
    needs_review: bool = False
    verdicts: list[FieldVerdict] = dataclass_field(default_factory=list)


def mark_for_review(session: Session, analysis_id: int) -> None:
    """Route an item to the human review queue (which is driven by
    ``AIAnalysisRecord.needs_human_review``)."""
    record = session.get(AIAnalysisRecord, analysis_id)
    if record is not None and not record.needs_human_review:
        record.needs_human_review = True
        session.flush()


def validate_analysis(
    session: Session,
    *,
    raw_item_id: int,
    student_analysis: AIAnalysis,
    student_model: str,
    analysis_id: int | None = None,
    teacher: SupportsAnalyze | None = None,
    teacher_item: NormalizedItem | None = None,
    high_confidence_threshold: float = DEFAULT_HIGH_CONFIDENCE,
    fields: tuple[str, ...] = LABEL_FIELDS,
    write_labels: bool = True,
) -> ValidationResult:
    """Validate a student analysis and route each field to silver or to review.

    When ``teacher`` and ``teacher_item`` are given, agreement is judged against
    the teacher's independent analysis. Otherwise the student's own confidence
    decides. Silver labels are written for agreed fields; disagreement flags the
    item for human review.
    """
    result = ValidationResult(raw_item_id=raw_item_id)

    teacher_analysis: AIAnalysis | None = None
    if teacher is not None and teacher_item is not None:
        teacher_analysis = teacher.analyze(teacher_item)

    student_conf = float(student_analysis.confidence or 0.0)

    for fld in fields:
        student_value = extract_field_value(student_analysis, fld)

        if teacher_analysis is not None:
            teacher_value = extract_field_value(teacher_analysis, fld)
            agreed = student_value == teacher_value
            provenance = LabelProvenance.TEACHER_MODEL
            validator = f"{student_model}+{teacher.model_name}"
            confidence = student_conf
        else:
            teacher_value = None
            agreed = student_conf >= high_confidence_threshold
            provenance = LabelProvenance.AI_HIGH_CONF
            validator = student_model
            confidence = student_conf

        tier_written: str | None = None
        if agreed:
            result.agreed.append(fld)
            if write_labels:
                record_label(
                    session,
                    raw_item_id=raw_item_id,
                    field=fld,
                    value=student_value,
                    tier=LabelTier.SILVER,
                    provenance=provenance,
                    confidence=confidence,
                    validator=validator,
                    analysis_id=analysis_id,
                )
                result.silver_written += 1
                tier_written = LabelTier.SILVER.value
        else:
            result.disagreed.append(fld)

        result.verdicts.append(
            FieldVerdict(
                field=fld,
                student_value=student_value,
                teacher_value=teacher_value,
                agreed=agreed,
                tier_written=tier_written,
            )
        )

    result.needs_review = bool(result.disagreed)
    if result.needs_review and analysis_id is not None and write_labels:
        mark_for_review(session, analysis_id)

    return result
