"""The Bronze/Silver/Gold labeled-data store — the spine of the learning
flywheel.

Every label for every classifier field lands here with a quality *tier* and a
*provenance*, so the system can (a) grow training signal in bulk from cheap AI
labels, (b) capture human corrections as trusted gold, and (c) never lose track
of where a label came from. The eval harness (``tnmi.eval``) draws its frozen
test set only from gold rows; the trainer (``tnmi.training``) trains on
silver+gold. Nothing here trains a model — it only curates the data that does.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from tnmi.contracts import LABEL_FIELDS, LabelProvenance, LabelTier
from tnmi.storage import (
    AIAnalysisRecord,
    LabeledExampleRecord,
    RawItemRecord,
    ReviewDecisionRecord,
)


# A corrected review field maps to the classifier field it supervises.
_CORRECTION_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("corrected_stance", "tvk_portrayal"),
    ("corrected_relevance", "government_relevance"),
)


def split_bucket(raw_item_id: int, field: str) -> int:
    """Deterministic [0,100) bucket from (raw_item_id, field).

    Stable across runs and processes (uses sha256, not Python's salted hash),
    so the held-out gold test set has fixed membership — an item never drifts
    between train and test between two invocations.
    """
    digest = hashlib.sha256(f"{raw_item_id}:{field}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def _stringify(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def record_label(
    session: Session,
    *,
    raw_item_id: int,
    field: str,
    value: Any,
    tier: str,
    provenance: str,
    confidence: float = 0.0,
    validator: str = "",
    analysis_id: int | None = None,
) -> LabeledExampleRecord:
    """Upsert one label at one tier. Re-recording the same (item, field, tier)
    overwrites the value — the latest verdict at that tier wins."""
    tier = LabelTier(tier).value
    provenance = LabelProvenance(provenance).value
    value_str = _stringify(value)
    existing = session.scalar(
        select(LabeledExampleRecord).where(
            LabeledExampleRecord.raw_item_id == raw_item_id,
            LabeledExampleRecord.field == field,
            LabeledExampleRecord.tier == tier,
        )
    )
    if existing is not None:
        existing.value = value_str
        existing.provenance = provenance
        existing.confidence = confidence
        existing.validator = validator
        if analysis_id is not None:
            existing.analysis_id = analysis_id
        session.flush()
        return existing

    record = LabeledExampleRecord(
        raw_item_id=raw_item_id,
        analysis_id=analysis_id,
        field=field,
        value=value_str,
        tier=tier,
        provenance=provenance,
        confidence=confidence,
        validator=validator,
        split_bucket=split_bucket(raw_item_id, field),
    )
    session.add(record)
    session.flush()
    return record


def record_bronze_from_analysis(
    session: Session,
    analysis: AIAnalysisRecord,
    *,
    fields: tuple[str, ...] = LABEL_FIELDS,
) -> list[LabeledExampleRecord]:
    """Write bronze (raw, unverified) labels for an analysis record's fields.

    This is the entry point every fresh classification flows through — bulk,
    cheap, untrusted signal that a teacher model or a human can later promote.
    """
    out: list[LabeledExampleRecord] = []
    for field in fields:
        if not hasattr(analysis, field):
            continue
        out.append(
            record_label(
                session,
                raw_item_id=analysis.raw_item_id,
                field=field,
                value=getattr(analysis, field),
                tier=LabelTier.BRONZE,
                provenance=LabelProvenance.AI,
                confidence=float(analysis.confidence or 0.0),
                validator=analysis.model_name or "",
                analysis_id=analysis.id,
            )
        )
    return out


def promote_corrections_to_gold(session: Session) -> int:
    """Turn human review corrections into gold labels — for free.

    Every operator who fixed a card's stance/relevance has produced a verified
    label; we mirror those into the gold tier so the eval set and training data
    grow with use. Returns the number of gold labels written/updated.
    """
    rows = session.execute(
        select(ReviewDecisionRecord, AIAnalysisRecord.raw_item_id)
        .join(AIAnalysisRecord, AIAnalysisRecord.id == ReviewDecisionRecord.analysis_id)
        .order_by(ReviewDecisionRecord.created_at.asc(), ReviewDecisionRecord.id.asc())
    ).all()
    count = 0
    for decision, raw_item_id in rows:
        for attr, field in _CORRECTION_FIELD_MAP:
            corrected = getattr(decision, attr, None)
            if not corrected:
                continue
            record_label(
                session,
                raw_item_id=raw_item_id,
                field=field,
                value=corrected,
                tier=LabelTier.GOLD,
                provenance=LabelProvenance.HUMAN,
                confidence=1.0,
                validator=decision.reviewer_name or "reviewer",
                analysis_id=decision.analysis_id,
            )
            count += 1
    return count


@dataclass(frozen=True)
class LabelRow:
    raw_item_id: int
    field: str
    value: str
    tier: str
    provenance: str
    confidence: float
    split_bucket: int
    title: str | None
    text: str


def _best_tier_rank(tier: str) -> int:
    return {LabelTier.GOLD: 3, LabelTier.SILVER: 2, LabelTier.BRONZE: 1}.get(LabelTier(tier), 0)


def export_dataset(
    session: Session,
    *,
    fields: tuple[str, ...] | None = None,
    tiers: tuple[str, ...] = (LabelTier.SILVER.value, LabelTier.GOLD.value),
    exclude_split_buckets: range | tuple[int, ...] | None = None,
) -> list[LabelRow]:
    """Materialise training rows: one per (item, field), text + label joined.

    Keeps only the highest available tier per (item, field) among ``tiers`` so a
    gold correction always supersedes its silver/bronze ancestor. ``exclude_split_buckets``
    lets the trainer hold out the eval test set.
    """
    wanted_tiers = {LabelTier(t).value for t in tiers}
    excluded = set(exclude_split_buckets) if exclude_split_buckets is not None else set()

    stmt = (
        select(LabeledExampleRecord, RawItemRecord)
        .join(RawItemRecord, RawItemRecord.id == LabeledExampleRecord.raw_item_id)
        .where(LabeledExampleRecord.tier.in_(wanted_tiers))
    )
    if fields is not None:
        stmt = stmt.where(LabeledExampleRecord.field.in_(fields))

    best: dict[tuple[int, str], tuple[LabeledExampleRecord, RawItemRecord]] = {}
    for label, raw in session.execute(stmt).all():
        if label.split_bucket in excluded:
            continue
        key = (label.raw_item_id, label.field)
        current = best.get(key)
        if current is None or _best_tier_rank(label.tier) > _best_tier_rank(current[0].tier):
            best[key] = (label, raw)

    rows: list[LabelRow] = []
    for label, raw in best.values():
        rows.append(
            LabelRow(
                raw_item_id=label.raw_item_id,
                field=label.field,
                value=label.value,
                tier=label.tier,
                provenance=label.provenance,
                confidence=label.confidence,
                split_bucket=label.split_bucket,
                title=raw.title,
                text=raw.clean_text_original or raw.raw_text_original or "",
            )
        )
    rows.sort(key=lambda r: (r.field, r.raw_item_id))
    return rows


def dataset_stats(session: Session) -> dict[str, Any]:
    """Counts by tier / field / provenance — the health readout of the flywheel."""
    rows = session.execute(
        select(
            LabeledExampleRecord.tier,
            LabeledExampleRecord.field,
            LabeledExampleRecord.provenance,
        )
    ).all()
    tiers: Counter[str] = Counter()
    fields: Counter[str] = Counter()
    provenance: Counter[str] = Counter()
    for tier, field, prov in rows:
        tiers[tier] += 1
        fields[field] += 1
        provenance[prov] += 1
    return {
        "total_labels": len(rows),
        "by_tier": dict(tiers),
        "by_field": dict(fields),
        "by_provenance": dict(provenance),
    }
