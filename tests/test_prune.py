import pytest

from tests.test_storage import make_analysis, make_item
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item
from pipelines.prune_superseded_analyses import PruneAborted, prune_superseded


def _seed(session_factory):
    """Two articles, each with an old and a current-version analysis."""
    with session_factory() as session:
        a = save_raw_item(session, make_item().model_copy(update={"source_url": "https://e.com/a"}))
        b = save_raw_item(session, make_item().model_copy(update={"source_url": "https://e.com/b"}))
        for raw in (a, b):
            save_ai_analysis(session, raw.id, make_analysis(), model_name="local-tamil-keywords", prompt_version="old-v1")
            save_ai_analysis(session, raw.id, make_analysis(), model_name="local-tamil-keywords", prompt_version="old-v2")
            save_ai_analysis(session, raw.id, make_analysis(), model_name="local-tamil-keywords", prompt_version="keep-v3")
        session.commit()
        return a.id, b.id


def test_prune_dry_run_changes_nothing(tmp_path):
    sf = create_session_factory(f"sqlite:///{tmp_path / 'p.db'}")
    init_db(sf)
    _seed(sf)
    with sf() as session:
        result = prune_superseded(session, keep_version="keep-v3", dry_run=True)
        session.commit()
    assert result.rows_before == 6
    assert result.rows_keep == 2
    assert result.rows_to_delete == 4
    assert result.rows_deleted == 0  # dry run


def test_prune_keeps_one_current_per_item(tmp_path):
    sf = create_session_factory(f"sqlite:///{tmp_path / 'p.db'}")
    init_db(sf)
    a_id, b_id = _seed(sf)
    with sf() as session:
        result = prune_superseded(session, keep_version="keep-v3", dry_run=False)
        session.commit()
    assert result.rows_deleted == 4
    with sf() as session:
        from sqlalchemy import select, func
        from tnmi.storage import AIAnalysisRecord
        total = session.scalar(select(func.count()).select_from(AIAnalysisRecord))
        assert total == 2
        # every article retains exactly one current analysis
        for raw_id in (a_id, b_id):
            rows = session.scalars(
                select(AIAnalysisRecord).where(AIAnalysisRecord.raw_item_id == raw_id)
            ).all()
            assert len(rows) == 1
            assert rows[0].prompt_version == "keep-v3"


def test_prune_aborts_when_item_would_be_orphaned(tmp_path):
    sf = create_session_factory(f"sqlite:///{tmp_path / 'p.db'}")
    init_db(sf)
    with sf() as session:
        raw = save_raw_item(session, make_item())
        # Only an OLD analysis exists — no keep-version row.
        save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="old-v1")
        session.commit()
    with sf() as session:
        with pytest.raises(PruneAborted):
            prune_superseded(session, keep_version="keep-v3", dry_run=False)
        session.commit()
        # nothing deleted
        from sqlalchemy import select, func
        from tnmi.storage import AIAnalysisRecord
        assert session.scalar(select(func.count()).select_from(AIAnalysisRecord)) == 1
