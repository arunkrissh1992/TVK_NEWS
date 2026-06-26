"""Seed the canonical entity roster and resolve all analyses into the graph.

    python -m pipelines.resolve_entities            # seed + resolve everything
    python -m pipelines.resolve_entities --stats    # print machine-readable stats

Idempotent — safe to re-run after every seed edit or reanalysis pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tnmi.config import Settings
from tnmi.resolver import resolve_all
from tnmi.storage import create_session_factory, init_db


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, default=None, help="entities seed YAML (default: settings)")
    parser.add_argument("--stats", action="store_true", help="print stats as JSON")
    args = parser.parse_args(argv)

    settings = Settings()
    seed_path = args.seed or settings.entities_seed_config
    if not Path(seed_path).exists():
        parser.error(f"seed file not found: {seed_path}")

    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)
    with session_factory() as session:
        stats = resolve_all(session, seed_path=seed_path)
        session.commit()

    payload = stats.as_dict()
    if args.stats:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(
        f"entities seeded/synced: {payload['entities_seeded']} | "
        f"items resolved: {payload['items_processed']} | "
        f"edges created: {payload['mentions_created']} "
        f"(replaced {payload['mentions_replaced']}) | "
        f"resolution rate: {payload['resolution_rate']:.0%} "
        f"({payload['candidate_surfaces']} candidate surfaces queued)"
    )
    if payload["top_candidates"]:
        print("top unresolved surfaces (confirm in configs/entities.seed.yaml):")
        for surface, count in payload["top_candidates"]:
            print(f"  {count:>4} × {surface}")


if __name__ == "__main__":
    main()
