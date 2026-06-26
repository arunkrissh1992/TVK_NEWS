"""Run one full pass of the self-improving flywheel.

    corrections → gold     (humans are the yardstick)
    analyses    → bronze   (raw signal)
    validation  → silver   (teacher-agreement / high confidence) or review
    silver+gold → train    (held-out gold test set excluded)
    candidate   → eval     (frozen gold test set)
    gate        → promote  (only if it beats the live model)

Safe to run nightly via cron / the DAGs folder. With no flags it uses the
no-GPU StubTrainer and the mock analyzer as the candidate — a full dry run of
the machinery. On the GPU box, use --peft with a real base model and point the
candidate at the trained artifact.

Examples:

    python -m pipelines.flywheel                        # dry run, no GPU needed
    python -m pipelines.flywheel --teacher openai       # teacher-validated silver
    python -m pipelines.flywheel --peft --base-model sarvamai/sarvam-1
    python -m pipelines.flywheel --json flywheel-report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tnmi.ai import MockAIAnalyzer, OpenAIAnalyzer
from tnmi.config import Settings
from tnmi.flywheel import run_flywheel
from tnmi.local_models import LocalTamilAnalyzer
from tnmi.storage import create_session_factory, init_db
from tnmi.training import PeftLoraTrainer, StubTrainer


def _analyzer(kind: str, settings: Settings):
    if kind == "openai":
        api_key = getattr(settings, "openai_api_key", None)
        if not api_key:
            raise SystemExit("No OpenAI API key configured for the requested analyzer.")
        return OpenAIAnalyzer(api_key=api_key)
    if kind == "local-tamil":
        return LocalTamilAnalyzer()
    return MockAIAnalyzer()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One pass of the self-improving flywheel.")
    parser.add_argument("--model-name", default="tvk-tamil-classifier")
    parser.add_argument("--peft", action="store_true", help="real LoRA fine-tune (GPU box)")
    parser.add_argument("--base-model", default="stub-base")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument(
        "--teacher",
        choices=["none", "openai", "local-tamil"],
        default="none",
        help="independent model that corroborates student labels into silver",
    )
    parser.add_argument(
        "--candidate",
        choices=["mock", "openai", "local-tamil"],
        default="mock",
        help="analyzer evaluated against the gold test set as the promotion candidate",
    )
    parser.add_argument("--min-delta", type=float, default=0.0, help="required improvement to promote")
    parser.add_argument("--confidence", type=float, default=0.85, help="silver threshold without a teacher")
    parser.add_argument("--limit", type=int, default=None, help="max items to validate this pass")
    parser.add_argument("--json", type=Path, default=None, help="write the full report as JSON here")
    args = parser.parse_args(argv)

    settings = Settings()
    factory = create_session_factory(settings.database_url)
    init_db(factory)

    teacher = None if args.teacher == "none" else _analyzer(args.teacher, settings)
    candidate = _analyzer(args.candidate, settings)
    trainer = PeftLoraTrainer() if args.peft else StubTrainer()

    with factory() as session:
        report = run_flywheel(
            session,
            trainer=trainer,
            candidate_analyzer=candidate,
            teacher=teacher,
            model_name=args.model_name,
            base_model=args.base_model,
            output_dir=str(args.output_dir),
            min_delta=args.min_delta,
            high_confidence_threshold=args.confidence,
            validate_limit=args.limit,
        )
        session.commit()

    payload = report.as_dict()
    print("Flywheel pass complete")
    print("-" * 60)
    print(f"gold from corrections : {payload['gold_promoted']}")
    print(f"bronze written        : {payload['bronze_written']}")
    print(f"items validated       : {payload['items_validated']}")
    print(f"silver written        : {payload['silver_written']}")
    print(f"routed to review      : {payload['routed_to_review']}")
    print(f"training examples     : {payload['training_examples']}")
    if payload["trained_version"]:
        print(f"trained version       : {payload['trained_version']}")
        print(f"eval examples (gold)  : {payload['eval_total']}")
        print(f"primary metric        : {payload['primary_metric']}")
        promo = payload["promotion"]
        if promo:
            verdict = "PROMOTED" if promo["promoted"] else "kept incumbent"
            print(f"promotion gate        : {verdict} — {promo['reason']}")
    print(f"label store           : {payload['label_stats']}")

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
