"""Knowledge vault rendering — idempotency, human-section preservation, links."""

from __future__ import annotations

from datetime import datetime, timezone

from tnmi.resolver import resolve_all
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item
from tnmi.vault import build_vault
from tests.test_resolver import SEED_PATH, _analysis, _item


def _build(tmp_path, *, vault_name="vault"):
    factory = create_session_factory(f"sqlite:///{tmp_path / 'vault.db'}")
    init_db(factory)
    vault_dir = tmp_path / vault_name
    with factory() as session:
        raw = save_raw_item(session, _item())
        save_ai_analysis(
            session,
            raw.id,
            _analysis(
                actors=["Vijay", "Chief Minister", "DMK"],
                district="மதுரை",
                department="Transport / road safety",
            ),
            model_name="ollama/gemma2:2b",
            prompt_version="tvk-portrayal-v16",
        )
        second = save_raw_item(session, _item(url="https://example.com/b"))
        save_ai_analysis(
            session,
            second.id,
            _analysis(actors=["TVK", "Mystery Person"], district="Chennai"),
            model_name="ollama/gemma2:2b",
            prompt_version="tvk-portrayal-v16",
        )
        session.commit()
        resolve_all(session, seed_path=SEED_PATH)
        session.commit()
        stats = build_vault(session, vault_dir)
        session.commit()
    return factory, vault_dir, stats


def _snapshot(vault_dir):
    return {
        path.relative_to(vault_dir): path.read_bytes()
        for path in sorted(vault_dir.rglob("*"))
        if path.is_file()
    }


def test_build_vault_renders_expected_dossiers(tmp_path):
    _, vault_dir, stats = _build(tmp_path)
    assert (vault_dir / "Home.md").exists()
    assert (vault_dir / "Calendar.md").exists()
    assert (vault_dir / "_meta" / "CONVENTIONS.md").exists()
    assert (vault_dir / ".obsidian" / "graph.json").exists()
    assert (vault_dir / "Entities" / "People" / "vijay.md").exists()
    assert (vault_dir / "Entities" / "Parties" / "dmk.md").exists()
    assert (vault_dir / "Geography" / "Districts" / "madurai.md").exists()
    assert (vault_dir / "Government" / "Departments" / "dept-transport.md").exists()
    assert (vault_dir / "Sources" / "source-the-hindu-chennai.md").exists()
    # Candidates are listed on Home, never given dossiers.
    assert not list(vault_dir.rglob("candidate-*.md"))
    assert stats.entities_rendered > 40  # roster + districts + sources
    assert stats.candidates_listed == 1


def test_vijay_dossier_links_and_cites_evidence(tmp_path):
    _, vault_dir, _ = _build(tmp_path)
    text = (vault_dir / "Entities" / "People" / "vijay.md").read_text(encoding="utf-8")
    assert "^raw-" in text  # every bullet cites its raw item
    assert "[[madurai|Madurai]]" in text  # wikilinked to the district
    assert "[[dmk|DMK]]" in text  # co-actor link
    assert "விஜய்" in text  # Tamil alias in frontmatter
    assert "## Evidence log" in text


def test_home_lists_risks_and_candidates(tmp_path):
    _, vault_dir, _ = _build(tmp_path)
    home = (vault_dir / "Home.md").read_text(encoding="utf-8")
    assert "war room" in home
    assert "## Top risks" in home
    assert "`Mystery Person`" in home  # candidate queued for confirmation
    assert "[[vijay|Vijay]]" in home or "[[vijay]]" in home


def test_double_render_is_byte_identical(tmp_path):
    factory, vault_dir, _ = _build(tmp_path)
    before = _snapshot(vault_dir)
    with factory() as session:
        stats = build_vault(session, vault_dir)
    after = _snapshot(vault_dir)
    assert before == after
    assert stats.dossiers_written == 0  # nothing changed → nothing rewritten


def test_human_notes_survive_regeneration(tmp_path):
    factory, vault_dir, _ = _build(tmp_path)
    dossier = vault_dir / "Entities" / "People" / "vijay.md"
    text = dossier.read_text(encoding="utf-8")
    marked = text.replace(
        "_Notes added here survive every regeneration._",
        "Ground report: cadre meeting planned in Madurai west.",
    )
    dossier.write_text(marked, encoding="utf-8")

    with factory() as session:
        build_vault(session, vault_dir)

    regenerated = dossier.read_text(encoding="utf-8")
    assert "Ground report: cadre meeting planned in Madurai west." in regenerated
