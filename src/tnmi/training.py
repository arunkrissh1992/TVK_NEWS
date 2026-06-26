"""The trainer layer — turn curated labels into a model, GPU-ready.

Two things live here:

1. ``build_distillation_dataset`` / ``export_jsonl`` — assemble silver+gold labels
   into (article → labels) training rows, **excluding the frozen eval test set**
   so we never train on what we measure on.

2. A ``Trainer`` interface with two implementations:
   * ``StubTrainer`` — deterministic, no GPU, no heavy deps. Lets CI and dry-runs
     exercise the full flywheel (dataset → train → register → gate) on a laptop.
   * ``PeftLoraTrainer`` — a real LoRA fine-tune that runs on the GPU box. It
     lazy-imports torch/transformers/peft so importing this module stays cheap and
     dependency-free; it raises a clear error if the training stack is absent.

The deployable "political intelligence" model is the artifact this produces,
re-trained as labels accumulate, and only ever promoted through the eval gate in
``tnmi.registry``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Protocol

from sqlalchemy.orm import Session

from tnmi.contracts import LABEL_FIELDS, LabelTier
from tnmi.eval import HELD_OUT_TEST_BUCKETS
from tnmi.labeling import export_dataset


@dataclass
class TrainingExample:
    raw_item_id: int
    title: str | None
    text: str
    labels: dict[str, str] = dataclass_field(default_factory=dict)


def build_distillation_dataset(
    session: Session,
    *,
    fields: tuple[str, ...] = LABEL_FIELDS,
    tiers: tuple[str, ...] = (LabelTier.SILVER.value, LabelTier.GOLD.value),
    exclude_held_out: bool = True,
) -> list[TrainingExample]:
    """Group the best-tier labels into one training row per article.

    ``exclude_held_out=True`` drops the gold test buckets so the trainer can
    never see the examples the eval gate will judge it on — the firewall that
    keeps reported scores honest.
    """
    excl = HELD_OUT_TEST_BUCKETS if exclude_held_out else None
    rows = export_dataset(session, fields=fields, tiers=tiers, exclude_split_buckets=excl)
    by_item: dict[int, TrainingExample] = {}
    for r in rows:
        ex = by_item.get(r.raw_item_id)
        if ex is None:
            ex = TrainingExample(raw_item_id=r.raw_item_id, title=r.title, text=r.text)
            by_item[r.raw_item_id] = ex
        ex.labels[r.field] = r.value
    return [by_item[k] for k in sorted(by_item)]


def export_jsonl(examples: list[TrainingExample], path: str | Path) -> int:
    """Write the dataset as JSONL — the hand-off format for an external GPU job."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(
                json.dumps(
                    {
                        "raw_item_id": ex.raw_item_id,
                        "title": ex.title,
                        "text": ex.text,
                        "labels": ex.labels,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return len(examples)


def dataset_fingerprint(examples: list[TrainingExample]) -> str:
    """Deterministic 12-hex id of a dataset's content — used as a model version so
    the same data always yields the same version (idempotent, reproducible)."""
    hasher = hashlib.sha256()
    for ex in sorted(examples, key=lambda e: e.raw_item_id):
        payload = f"{ex.raw_item_id}|" + ",".join(f"{k}={ex.labels[k]}" for k in sorted(ex.labels))
        hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()[:12]


@dataclass
class TrainingResult:
    model_name: str
    version: str
    base_model: str
    artifact_uri: str
    num_examples: int
    metadata: dict[str, Any] = dataclass_field(default_factory=dict)


class Trainer(Protocol):
    def train(
        self,
        examples: list[TrainingExample],
        *,
        model_name: str,
        base_model: str,
        output_dir: str | Path,
    ) -> TrainingResult: ...


class StubTrainer:
    """A deterministic no-GPU trainer for tests, CI, and dry-runs.

    It does not learn anything — it records the dataset and emits a reproducible
    version + artifact path so the registry, promotion gate, and orchestration
    can be exercised end-to-end without hardware. Swap in ``PeftLoraTrainer`` on
    the GPU box for real weights.
    """

    name = "stub-trainer"

    def train(
        self,
        examples: list[TrainingExample],
        *,
        model_name: str,
        base_model: str = "stub-base",
        output_dir: str | Path = "artifacts",
    ) -> TrainingResult:
        version = dataset_fingerprint(examples) if examples else "empty"
        artifact_uri = str(Path(output_dir) / f"{model_name}-{version}")
        label_counts: dict[str, int] = {}
        for ex in examples:
            for fld in ex.labels:
                label_counts[fld] = label_counts.get(fld, 0) + 1
        return TrainingResult(
            model_name=model_name,
            version=version,
            base_model=base_model,
            artifact_uri=artifact_uri,
            num_examples=len(examples),
            metadata={"trainer": self.name, "label_counts": label_counts},
        )


class PeftLoraTrainer:
    """Real LoRA fine-tune of a causal LM, run on the GPU box.

    Distillation target: given the article, the model emits the JSON label block
    the teacher/humans agreed on. Dependencies (torch, transformers, peft,
    datasets) are imported lazily so this module is importable anywhere; calling
    ``train`` without the stack raises a clear, actionable error.
    """

    name = "peft-lora-trainer"

    def __init__(
        self,
        *,
        epochs: int = 3,
        learning_rate: float = 2e-4,
        lora_r: int = 16,
        lora_alpha: int = 32,
        max_length: int = 1024,
        batch_size: int = 4,
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.max_length = max_length
        self.batch_size = batch_size

    @staticmethod
    def format_target(labels: dict[str, str]) -> str:
        return json.dumps({k: labels[k] for k in sorted(labels)}, ensure_ascii=False)

    @staticmethod
    def format_prompt(title: str | None, text: str) -> str:
        head = (title + "\n") if title else ""
        return (
            "Classify this Tamil Nadu news article for the TVK political-intelligence "
            "system. Return JSON with fields government_relevance, tvk_relevance, "
            "stance_toward_government, tvk_portrayal, people_issue, issue_category, "
            f"severity.\n\nArticle:\n{head}{text}\n\nLabels:"
        )

    def _require_stack(self):  # pragma: no cover - exercised only on the GPU box
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import peft  # noqa: F401
            import datasets  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PeftLoraTrainer needs the training stack. On the GPU box run:\n"
                "  pip install 'torch' 'transformers>=4.44' 'peft>=0.12' 'datasets' 'accelerate'\n"
                "then retry, or use StubTrainer for a no-GPU dry run."
            ) from exc

    def train(  # pragma: no cover - requires GPU + heavy deps, not run in CI
        self,
        examples: list[TrainingExample],
        *,
        model_name: str,
        base_model: str,
        output_dir: str | Path,
    ) -> TrainingResult:
        self._require_stack()
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer as HFTrainer,
            TrainingArguments,
        )

        version = dataset_fingerprint(examples) if examples else "empty"
        out_dir = Path(output_dir) / f"{model_name}-{version}"
        out_dir.mkdir(parents=True, exist_ok=True)

        tokenizer = AutoTokenizer.from_pretrained(base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def _to_text(ex: TrainingExample) -> str:
            return self.format_prompt(ex.title, ex.text) + " " + self.format_target(ex.labels)

        ds = Dataset.from_dict({"text": [_to_text(ex) for ex in examples]})
        ds = ds.map(
            lambda b: tokenizer(b["text"], truncation=True, max_length=self.max_length),
            batched=True,
            remove_columns=["text"],
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
            ),
        )

        trainer = HFTrainer(
            model=model,
            args=TrainingArguments(
                output_dir=str(out_dir),
                num_train_epochs=self.epochs,
                learning_rate=self.learning_rate,
                per_device_train_batch_size=self.batch_size,
                logging_steps=10,
                save_strategy="epoch",
                report_to=[],
            ),
            train_dataset=ds,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        return TrainingResult(
            model_name=model_name,
            version=version,
            base_model=base_model,
            artifact_uri=str(out_dir),
            num_examples=len(examples),
            metadata={"trainer": self.name, "epochs": self.epochs, "lora_r": self.lora_r},
        )
