import pytest

from tnmi.storage import RawItemRecord, save_raw_item
from tnmi.tenancy import (
    ControlPlane,
    TenantConfig,
    hash_password,
    slugify,
    verify_password,
)

from tests.test_storage import make_item


def _control(tmp_path) -> ControlPlane:
    return ControlPlane(
        f"sqlite:///{tmp_path / 'control.db'}",
        tenants_dir=tmp_path / "tenants",
    )


def test_password_hash_roundtrip():
    encoded = hash_password("s3cret!")
    assert encoded.startswith("pbkdf2$")
    assert verify_password("s3cret!", encoded) is True
    assert verify_password("wrong", encoded) is False


def test_slugify():
    assert slugify("Tamilaga Vettri Kazhagam") == "tamilaga-vettri-kazhagam"
    assert slugify("  DMK!! ") == "dmk"
    assert slugify("") == "tenant"


def test_provision_tenant_creates_isolated_database(tmp_path):
    control = _control(tmp_path)
    tvk = control.provision_tenant(
        name="TVK War Room",
        config=TenantConfig(display_name="TVK", subject_party="TVK", governing=True),
        slug="tvk",
        seed_entities=False,
    )
    assert tvk.slug == "tvk"
    assert "tvk.db" in tvk.database_url
    # Re-provisioning is idempotent — returns the same tenant, no clobber.
    again = control.provision_tenant(name="TVK War Room", slug="tvk", seed_entities=False)
    assert again.id == tvk.id


def test_tenant_databases_are_fully_isolated(tmp_path):
    """The core SaaS guarantee: a row written in tenant A's database is
    invisible in tenant B's. Rival parties never see each other's data."""
    control = _control(tmp_path)
    tvk = control.provision_tenant(name="TVK", slug="tvk", seed_entities=False)
    dmk = control.provision_tenant(name="DMK", slug="dmk", seed_entities=False)

    tvk_factory = control.session_factory_for(tvk)
    dmk_factory = control.session_factory_for(dmk)

    with tvk_factory() as session:
        save_raw_item(session, make_item().model_copy(update={"title": "TVK secret briefing"}))
        session.commit()

    with tvk_factory() as session:
        assert session.query(RawItemRecord).count() == 1
        assert session.query(RawItemRecord).first().title == "TVK secret briefing"

    # DMK's database must be empty — different database entirely.
    with dmk_factory() as session:
        assert session.query(RawItemRecord).count() == 0


def test_api_key_resolves_to_correct_tenant(tmp_path):
    control = _control(tmp_path)
    tvk = control.provision_tenant(name="TVK", slug="tvk", seed_entities=False)
    dmk = control.provision_tenant(name="DMK", slug="dmk", seed_entities=False)

    tvk_key, record = control.issue_api_key(tenant=tvk, label="dashboard")
    assert tvk_key.startswith("tvk_")
    assert record.key_prefix == tvk_key[:12]

    resolved = control.authenticate_api_key(tvk_key)
    assert resolved is not None
    assert resolved.slug == "tvk"
    # A different tenant's key never resolves to TVK.
    dmk_key, _ = control.issue_api_key(tenant=dmk)
    assert control.authenticate_api_key(dmk_key).slug == "dmk"
    # Garbage and empty keys resolve to nothing.
    assert control.authenticate_api_key("tvk_deadbeef") is None
    assert control.authenticate_api_key("") is None


def test_suspended_tenant_key_is_rejected(tmp_path):
    control = _control(tmp_path)
    tvk = control.provision_tenant(name="TVK", slug="tvk", seed_entities=False)
    key, _ = control.issue_api_key(tenant=tvk)
    with control.session() as session:
        from tnmi.tenancy import TenantRecord

        session.get(TenantRecord, tvk.id).status = "suspended"
        session.commit()
    assert control.authenticate_api_key(key) is None


def test_users_memberships_and_roles(tmp_path):
    control = _control(tmp_path)
    tvk = control.provision_tenant(name="TVK", slug="tvk", seed_entities=False)
    dmk = control.provision_tenant(name="DMK", slug="dmk", seed_entities=False)
    user = control.create_user(email="Analyst@example.com", name="Analyst", password="pw")

    control.add_membership(user=user, tenant=tvk, role="owner")
    control.add_membership(user=user, tenant=dmk, role="viewer")

    assert control.role_of(user_id=user.id, tenant_id=tvk.id) == "owner"
    assert control.role_of(user_id=user.id, tenant_id=dmk.id) == "viewer"
    memberships = {t.slug: role for t, role in control.memberships_for(user)}
    assert memberships == {"tvk": "owner", "dmk": "viewer"}

    # Auth works case-insensitively on email.
    assert control.authenticate_user(email="analyst@example.com", password="pw").id == user.id
    assert control.authenticate_user(email="analyst@example.com", password="nope") is None

    with pytest.raises(ValueError):
        control.add_membership(user=user, tenant=tvk, role="superadmin")


def test_tenant_config_defaults_to_tvk_lens(tmp_path):
    control = _control(tmp_path)
    tenant = control.provision_tenant(name="Default", slug="default", seed_entities=False)
    config = control.tenant_config(tenant)
    assert config.subject_party == "TVK"
    assert config.governing is True
    assert config.geography_pack == "tamil_nadu"


def test_build_analyzer_carries_tenant_lens():
    """Per-tenant ingest builds an OpenAI analyzer wired to that tenant's lens."""
    from pipelines.run_daily_news import build_analyzer

    class S:
        openai_api_key = "sk-test"
        openai_model_item_classifier = "gpt-x"

    analyzer = build_analyzer(S(), mock_ai=False, subject="DMK", leader="Stalin", governing=False)
    assert analyzer._lens == {"subject": "DMK", "leader": "Stalin", "governing": False}
    # Default lens unchanged.
    assert build_analyzer(S(), mock_ai=False)._lens == {
        "subject": "TVK", "leader": "Vijay", "governing": True,
    }


def test_usage_counts_analyses_per_tenant(tmp_path):
    from tnmi.storage import save_ai_analysis, save_raw_item
    from tests.test_storage import make_analysis

    control = _control(tmp_path)
    tvk = control.provision_tenant(name="TVK", slug="tvk", seed_entities=False)
    control.provision_tenant(name="DMK", slug="dmk", seed_entities=False)
    with control.session_factory_for(tvk)() as session:
        for i in range(3):
            raw = save_raw_item(session, make_item().model_copy(update={"source_url": f"https://e.com/{i}"}))
            save_ai_analysis(session, raw.id, make_analysis(), model_name="mock", prompt_version="v1")
        session.commit()

    usage = {u["slug"]: u for u in control.usage()}
    assert usage["tvk"]["analyses"] == 3
    assert usage["tvk"]["raw_items"] == 3
    assert usage["dmk"]["analyses"] == 0  # isolated — no leakage
