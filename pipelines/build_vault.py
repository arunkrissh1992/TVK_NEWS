"""Build the knowledge vault — resolve entities, then render markdown dossiers.

    python -m pipelines.build_vault              # resolve + render (the default)
    python -m pipelines.build_vault --render     # render only (skip resolution)
    python -m pipelines.build_vault --resolve    # resolve only (skip rendering)

Open the resulting vault/ folder in Obsidian for graph view, backlinks and
search — or just grep it. Idempotent and offline: no AI calls involved.
"""

from __future__ import annotations

import argparse
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
from tnmi.vault import build_vault


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolve", action="store_true", help="resolve entities only")
    parser.add_argument("--render", action="store_true", help="render markdown only")
    parser.add_argument("--vault-dir", type=Path, default=None, help="override vault output dir")
    args = parser.parse_args(argv)
    do_resolve = args.resolve or not args.render
    do_render = args.render or not args.resolve

    settings = Settings()
    vault_dir = args.vault_dir or settings.vault_dir
    session_factory = create_session_factory(settings.database_url)
    init_db(session_factory)

    with session_factory() as session:
        if do_resolve:
            stats = resolve_all(session, seed_path=settings.entities_seed_config)
            session.commit()
            payload = stats.as_dict()
            print(
                f"resolve: {payload['items_processed']} items → "
                f"{payload['mentions_created']} new edges, "
                f"rate {payload['resolution_rate']:.0%}, "
                f"{payload['candidate_surfaces']} candidates queued"
            )
        if do_render:
            vstats = build_vault(session, vault_dir)
            payload = vstats.as_dict()
            print(
                f"render: {payload['entities_rendered']} dossiers "
                f"({payload['dossiers_written']} written, "
                f"{payload['dossiers_unchanged']} unchanged) → {vault_dir}/ | "
                f"{payload['candidates_listed']} candidates listed on Home"
            )


if __name__ == "__main__":
    main()
