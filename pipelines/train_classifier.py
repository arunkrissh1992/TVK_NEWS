"""Build the distillation dataset and train a classifier version.

By default uses the no-GPU ``StubTrainer`` so the whole flow (dataset → train →
register) runs anywhere. On the GPU box pass ``--peft`` with a base model to run
a real LoRA fine-tune. Promotion to live is intentionally NOT done here — a new
model must be evaluated against the gold test set first (see
``pipelines.flywheel`` / ``tnmi.registry.promote_if_better``).

Examples:

    python -m pipelines.train_classifier --export artifacts/train.jsonl
    python -m pipelines.train_classifier --peft --base-model sarvamai/sarvam-1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tnmi.config import Settings
from tnmi.registry import register_model
from tnmi.storage import create_session_factory, init_db
from tnmi.training import (
    PeftLoraTrainer,
    StubTrainer,
    build_distillation_dataset,
    export_jsonl,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a classifier version from curated labels.")
    parser.add_argument("--model-name", default="tvk-tamil-classifier")
    parser.add_argument("--peft", action="store_true", help="real LoRA fine-tune (needs GPU + deps)")
    parser.add_argument("--base-model", default="stub-base", help="base model for --peft")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--export", type=Path, default=None, help="also write the dataset as JSONL here")
    parser.add_argument(
        "--include-held-out",
        action="store_true",
        help="DANGER: train on the eval test set too (only for debugging)",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    factory = create_session_factory(settings.database_url)
    init_db(factory)

    with factory() as session:
        examples = build_distillation_dataset(session, exclude_held_out=not args.include_held_out)

    if not examples:
        print(
            "No silver/gold training data yet. Run the validator to create silver "
            "labels and promote human corrections to gold first."
        )
        return 0

    print(f"Training examples: {len(examples)} (held-out test set excluded: {not args.include_held_out})")
    if args.export:
        export_jsonl(examples, args.export)
        print(f"Wrote dataset → {args.export}")

    trainer = PeftLoraTrainer() if args.peft else StubTrainer()
    result = trainer.train(
        examples,
        model_name=args.model_name,
        base_model=args.base_model,
        output_dir=args.output_dir,
    )
    print(f"Trained {result.model_name} v{result.version} → {result.artifact_uri}")

    with factory() as session:
        register_model(
            session,
            model_name=result.model_name,
            version=result.version,
            primary_metric=0.0,  # set by the eval step before promotion
            metrics={},
            eval_examples=0,
            artifact_uri=result.artifact_uri,
            notes=f"trainer={getattr(trainer, 'name', 'unknown')} base={result.base_model}",
        )
        session.commit()
    print("Registered (not promoted — evaluate against gold first).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
