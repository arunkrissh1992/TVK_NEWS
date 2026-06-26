"""Score a classifier against the frozen gold test set.

The gold test set is the held-out slice of human-verified labels (see
``tnmi.eval.HELD_OUT_TEST_BUCKETS``). This command never trains anything — it
only measures — and its numbers are the gate every model promotion must clear.

Examples:

    python -m pipelines.eval_classifier                 # eval the mock classifier
    python -m pipelines.eval_classifier --openai        # eval the OpenAI classifier
    python -m pipelines.eval_classifier --promote-corrections   # mirror human
                                                        # corrections to gold first
    python -m pipelines.eval_classifier --json report.json
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
from tnmi.eval import evaluate_classifier
from tnmi.labeling import promote_corrections_to_gold
from tnmi.local_models import LocalTamilAnalyzer
from tnmi.storage import create_session_factory, init_db


def _build_analyzer(args: argparse.Namespace, settings: Settings):
    if args.openai:
        api_key = getattr(settings, "openai_api_key", None)
        if not api_key:
            raise SystemExit("No OpenAI API key configured; set it or drop --openai.")
        return OpenAIAnalyzer(api_key=api_key)
    if args.local_tamil:
        return LocalTamilAnalyzer()
    return MockAIAnalyzer()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a classifier on the gold test set.")
    parser.add_argument("--openai", action="store_true", help="evaluate the OpenAI classifier")
    parser.add_argument("--local-tamil", action="store_true", help="evaluate the local Tamil classifier")
    parser.add_argument(
        "--promote-corrections",
        action="store_true",
        help="mirror human review corrections into gold labels before scoring",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the full report as JSON here")
    args = parser.parse_args(argv)

    settings = Settings()
    factory = create_session_factory(settings.database_url)
    init_db(factory)

    analyzer = _build_analyzer(args, settings)
    with factory() as session:
        if args.promote_corrections:
            promoted = promote_corrections_to_gold(session)
            session.commit()
            print(f"Promoted {promoted} human corrections to gold.")
        report = evaluate_classifier(session, analyzer)

    payload = report.as_dict()
    if report.total == 0:
        print(
            "No gold test labels yet. Capture human corrections in the review "
            "queue (then --promote-corrections) to build the gold set."
        )
        return 0

    print(f"Model: {analyzer.model_name}")
    print(f"Test examples: {report.total}   Overall accuracy: {payload['overall_accuracy']}")
    print("-" * 60)
    for name, m in payload["per_field"].items():
        print(
            f"{name:26s} acc={m['accuracy']:.3f}  "
            f"macroF1={m['macro_f1']:.3f}  n={m['support']}"
        )

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
