"""Collapse the ai_analysis table to the current-version analysis per article.

Re-running ingestion / re-analysis across many (model, prompt_version) pairs
leaves several superseded rows per article (the demo DB reached ~5.7k rows for
~1.06k articles). The dashboard already dedupes to the latest at read time, but
the bloat slows clustering and inflates the analysis counts.

This prunes every row whose ``prompt_version`` is not the keep-version — but
ONLY after verifying that every article that currently has any analysis still
retains at least one keep-version analysis. If pruning would orphan any
article, it aborts and changes nothing. Always run ``--dry-run`` first and keep
a database backup (the dashboard ships one alongside re-analysis).

Usage:
    python -m pipelines.prune_superseded_analyses --dry-run
    python -m pipelines.prune_superseded_analyses            # actually prune
    python -m pipelines.prune_superseded_analyses --keep-version tvk-portrayal-v18
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from tnmi.ai import PROMPT_VERSION
from tnmi.config import Settings
from tnmi.storage import AIAnalysisRecord, create_session_factory


class PruneAborted(RuntimeError):
    """Raised when pruning would leave an analysed article with no analysis."""


@dataclass(frozen=True)
class PruneResult:
    keep_version: str
    rows_before: int
    rows_keep: int
    rows_to_delete: int
    rows_deleted: int
    orphans_blocked: int
    dry_run: bool


def _orphan_count(session: Session, *, keep_version: str) -> int:
    """Articles that HAVE an analysis but NONE at the keep-version. Pruning is
    only safe when this is zero."""
    has_any = select(AIAnalysisRecord.raw_item_id).distinct().subquery()
    has_keep = (
        select(AIAnalysisRecord.raw_item_id)
        .where(AIAnalysisRecord.prompt_version == keep_version)
        .distinct()
        .subquery()
    )
    return int(
        session.scalar(
            select(func.count())
            .select_from(has_any)
            .where(has_any.c.raw_item_id.not_in(select(has_keep.c.raw_item_id)))
        )
        or 0
    )


def prune_superseded(
    session: Session, *, keep_version: str, dry_run: bool
) -> PruneResult:
    """Delete every analysis whose prompt_version != keep_version, guarding that
    no article is left without an analysis. Caller commits."""
    rows_before = int(session.scalar(select(func.count()).select_from(AIAnalysisRecord)) or 0)
    rows_keep = int(
        session.scalar(
            select(func.count())
            .select_from(AIAnalysisRecord)
            .where(AIAnalysisRecord.prompt_version == keep_version)
        )
        or 0
    )
    rows_to_delete = rows_before - rows_keep

    orphans = _orphan_count(session, keep_version=keep_version)
    if orphans:
        raise PruneAborted(
            f"{orphans} analysed article(s) have no '{keep_version}' analysis — "
            "re-run the re-analysis at this prompt version before pruning."
        )

    rows_deleted = 0
    if not dry_run and rows_to_delete:
        result = session.execute(
            delete(AIAnalysisRecord).where(AIAnalysisRecord.prompt_version != keep_version)
        )
        rows_deleted = int(result.rowcount or 0)

    return PruneResult(
        keep_version=keep_version,
        rows_before=rows_before,
        rows_keep=rows_keep,
        rows_to_delete=rows_to_delete,
        rows_deleted=rows_deleted,
        orphans_blocked=orphans,
        dry_run=dry_run,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-version",
        default=PROMPT_VERSION,
        help=f"prompt_version to keep (default: current {PROMPT_VERSION})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be deleted without changing anything",
    )
    args = parser.parse_args(argv)

    settings = Settings()
    session_factory = create_session_factory(settings.database_url)
    with session_factory() as session:
        result = prune_superseded(
            session, keep_version=args.keep_version, dry_run=args.dry_run
        )
        if not args.dry_run:
            session.commit()

    mode = "DRY RUN — nothing changed" if result.dry_run else "pruned"
    print(
        f"{mode}: keep_version={result.keep_version} "
        f"rows_before={result.rows_before} rows_keep={result.rows_keep} "
        f"rows_to_delete={result.rows_to_delete} rows_deleted={result.rows_deleted}"
    )


if __name__ == "__main__":
    main()
