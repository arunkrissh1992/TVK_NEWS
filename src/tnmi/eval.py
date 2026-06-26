"""The evaluation harness — the yardstick that gates every "smarter" change.

A held-out slice of the *gold* tier (human-verified) is the frozen test set. Any
classifier — the current prompt, a new prompt, a freshly fine-tuned model — is
scored against it with per-field accuracy / precision / recall / F1. Nothing is
promoted unless it beats the live model here (see ``tnmi.registry``). Without
this, "auto-training" is flying blind; with it, model collapse is impossible to
miss because a worse model simply fails the gate.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from tnmi.contracts import LABEL_FIELDS, AIAnalysis, LabelTier, NormalizedItem, SourceType
from tnmi.labeling import LabelRow, export_dataset
from tnmi.storage import RawItemRecord


# Items whose split bucket falls in this range are the frozen TEST set: never
# trained on, only ever measured. The trainer excludes exactly this range.
HELD_OUT_TEST_BUCKETS = range(0, 20)  # 20% of gold


class SupportsAnalyze(Protocol):
    model_name: str

    def analyze(self, item: NormalizedItem) -> AIAnalysis: ...


@dataclass
class ClassMetrics:
    label: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass
class FieldMetrics:
    field: str
    support: int
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: list[ClassMetrics] = dataclass_field(default_factory=list)


@dataclass
class EvalReport:
    total: int
    overall_accuracy: float
    per_field: dict[str, FieldMetrics] = dataclass_field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "overall_accuracy": round(self.overall_accuracy, 4),
            "per_field": {
                name: {
                    "support": m.support,
                    "accuracy": round(m.accuracy, 4),
                    "macro_precision": round(m.macro_precision, 4),
                    "macro_recall": round(m.macro_recall, 4),
                    "macro_f1": round(m.macro_f1, 4),
                    "per_class": [
                        {
                            "label": c.label,
                            "precision": round(c.precision, 4),
                            "recall": round(c.recall, 4),
                            "f1": round(c.f1, 4),
                            "support": c.support,
                        }
                        for c in m.per_class
                    ],
                }
                for name, m in self.per_field.items()
            },
        }


_NO_PREDICTION = "<none>"


def extract_field_value(analysis: AIAnalysis, field: str) -> str:
    """Stringify a classifier output the same way labels are stored, so gold
    and prediction compare apples-to-apples."""
    value = getattr(analysis, field, None)
    if value is None:
        return _NO_PREDICTION
    if isinstance(value, bool):
        return "true" if value else "false"
    # StrEnum members stringify to their value; plain str passes through.
    return getattr(value, "value", value) if not isinstance(value, str) else value


def gold_test_rows(
    session: Session,
    *,
    held_out: range = HELD_OUT_TEST_BUCKETS,
    fields: tuple[str, ...] = LABEL_FIELDS,
) -> list[LabelRow]:
    """The frozen test set: gold labels whose split bucket is held out."""
    rows = export_dataset(session, fields=fields, tiers=(LabelTier.GOLD.value,))
    held = set(held_out)
    return [r for r in rows if r.split_bucket in held]


def score_predictions(
    gold: list[tuple[int, str, str]],
    predictions: dict[tuple[int, str], str],
) -> EvalReport:
    """Pure metric computation: gold = (raw_item_id, field, value) triples,
    predictions keyed by (raw_item_id, field). Missing predictions count wrong."""
    by_field_gold: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for raw_item_id, fld, gold_value in gold:
        pred = predictions.get((raw_item_id, fld), _NO_PREDICTION)
        by_field_gold[fld].append((gold_value, pred))

    per_field: dict[str, FieldMetrics] = {}
    total = 0
    total_correct = 0
    for fld, pairs in by_field_gold.items():
        support = len(pairs)
        correct = sum(1 for g, p in pairs if g == p)
        total += support
        total_correct += correct

        labels = sorted({g for g, _ in pairs} | {p for _, p in pairs if p != _NO_PREDICTION})
        class_metrics: list[ClassMetrics] = []
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for label in labels:
            tp = sum(1 for g, p in pairs if g == label and p == label)
            fp = sum(1 for g, p in pairs if g != label and p == label)
            fn = sum(1 for g, p in pairs if g == label and p != label)
            gold_support = sum(1 for g, _ in pairs if g == label)
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            # Macro averages only over classes that actually appear in gold.
            if gold_support:
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
            class_metrics.append(
                ClassMetrics(label=label, precision=precision, recall=recall, f1=f1, support=gold_support)
            )

        per_field[fld] = FieldMetrics(
            field=fld,
            support=support,
            accuracy=correct / support if support else 0.0,
            macro_precision=sum(precisions) / len(precisions) if precisions else 0.0,
            macro_recall=sum(recalls) / len(recalls) if recalls else 0.0,
            macro_f1=sum(f1s) / len(f1s) if f1s else 0.0,
            per_class=class_metrics,
        )

    return EvalReport(
        total=total,
        overall_accuracy=total_correct / total if total else 0.0,
        per_field=per_field,
    )


def _raw_to_item(raw: RawItemRecord) -> NormalizedItem:
    try:
        source_type = SourceType(raw.source_type)
    except ValueError:
        source_type = SourceType.NEWS
    return NormalizedItem(
        source_type=source_type,
        source_name=raw.source_name,
        source_url=raw.source_url,
        published_at=raw.published_at,
        language=raw.language or "ta",
        title=raw.title,
        raw_text_original=raw.raw_text_original or raw.clean_text_original or "",
        clean_text_original=raw.clean_text_original or raw.raw_text_original or "",
    )


def predict_with_analyzer(
    session: Session,
    analyzer: SupportsAnalyze,
    raw_item_ids: list[int],
    *,
    fields: tuple[str, ...] = LABEL_FIELDS,
) -> dict[tuple[int, str], str]:
    """Run a classifier over the given items and extract its field outputs."""
    if not raw_item_ids:
        return {}
    raws = session.execute(
        select(RawItemRecord).where(RawItemRecord.id.in_(raw_item_ids))
    ).scalars().all()
    predictions: dict[tuple[int, str], str] = {}
    for raw in raws:
        analysis = analyzer.analyze(_raw_to_item(raw))
        for fld in fields:
            predictions[(raw.id, fld)] = extract_field_value(analysis, fld)
    return predictions


def evaluate_classifier(
    session: Session,
    analyzer: SupportsAnalyze,
    *,
    held_out: range = HELD_OUT_TEST_BUCKETS,
    fields: tuple[str, ...] = LABEL_FIELDS,
) -> EvalReport:
    """Score a classifier against the frozen gold test set."""
    test_rows = gold_test_rows(session, held_out=held_out, fields=fields)
    gold = [(r.raw_item_id, r.field, r.value) for r in test_rows]
    item_ids = sorted({r.raw_item_id for r in test_rows})
    predictions = predict_with_analyzer(session, analyzer, item_ids, fields=fields)
    return score_predictions(gold, predictions)
