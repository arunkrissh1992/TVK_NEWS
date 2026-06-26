"""Multi-tenant control plane — the SaaS foundation.

This platform serves *adversarial* customers (rival parties), so isolation is
silo, not pooled: every tenant gets its OWN database. This module is the small
shared control plane on top — it knows the tenants, their users, their API
keys, and their per-tenant "lens" config, and it hands each request the right
isolated session factory.

Design:
- ``ControlBase`` tables (tenants / users / memberships / api_keys) live in a
  separate control database, never co-mingled with tenant data.
- ``provision_tenant`` creates a brand-new isolated database for a tenant,
  initialises its schema, and seeds its entity roster.
- ``session_factory_for(tenant)`` returns (and caches) the SQLAlchemy session
  factory bound to that tenant's database — the rest of the app is unchanged.
- Auth resolves an API key to exactly one tenant; a query in tenant A's
  database can never see tenant B's rows because they are different databases.

Single-tenant deployments ignore all of this and keep using
``Settings.database_url`` directly.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from tnmi.storage import create_session_factory, init_db

_ID = BigInteger().with_variant(Integer, "sqlite")
_JSON = JSON().with_variant(JSONB, "postgresql")

VALID_ROLES = ("owner", "analyst", "viewer")
_API_KEY_PREFIX = "tvk"
_PBKDF2_ROUNDS = 240_000


class ControlBase(DeclarativeBase):
    pass


class TenantRecord(ControlBase):
    """One customer workspace. ``database_url`` points at its isolated DB;
    ``config_json`` holds its :class:`TenantConfig` (the political lens)."""

    __tablename__ = "tenants"
    __table_args__ = (UniqueConstraint("slug", name="uq_tenants_slug"),)

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    slug: Mapped[str] = mapped_column(String(80), index=True)
    name: Mapped[str] = mapped_column(String(255))
    database_url: Mapped[str] = mapped_column(Text)
    config_json: Mapped[dict[str, Any]] = mapped_column(_JSON, default=dict, server_default=text("'{}'"))
    status: Mapped[str] = mapped_column(String(16), default="active", server_default="active", index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )


class UserRecord(ControlBase):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), index=True)
    name: Mapped[str] = mapped_column(String(255), default="", server_default="")
    password_hash: Mapped[str] = mapped_column(Text, default="", server_default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )


class MembershipRecord(ControlBase):
    """A user's role within a tenant — the same person can belong to several."""

    __tablename__ = "memberships"
    __table_args__ = (UniqueConstraint("user_id", "tenant_id", name="uq_membership_user_tenant"),)

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(_ID, ForeignKey("users.id", ondelete="CASCADE"), index=True)
    tenant_id: Mapped[int] = mapped_column(_ID, ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16), default="viewer", server_default="viewer")


class ApiKeyRecord(ControlBase):
    """A hashed API key bound to one tenant. The raw key is shown once at
    creation and never stored — only its SHA-256 hash and a display prefix."""

    __tablename__ = "api_keys"
    __table_args__ = (UniqueConstraint("key_hash", name="uq_api_key_hash"),)

    id: Mapped[int] = mapped_column(_ID, primary_key=True, autoincrement=True)
    tenant_id: Mapped[int] = mapped_column(_ID, ForeignKey("tenants.id", ondelete="CASCADE"), index=True)
    label: Mapped[str] = mapped_column(String(120), default="", server_default="")
    key_prefix: Mapped[str] = mapped_column(String(24), default="", server_default="")
    key_hash: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), server_default=func.now()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class TenantConfig(BaseModel):
    """A tenant's political lens — what makes one engine serve many parties.

    TVK/Tamil-Nadu is the default; another party flips ``subject_party`` and
    ``governing`` and points at its own roster and sources, with no code change.
    """

    display_name: str = ""
    subject_party: str = "TVK"
    subject_leader: str = "Vijay"
    subject_entity_slug: str = "vijay"
    # True when the subject party holds government — drives the positive/negative
    # lens (governance wins are positive, failures negative) vs an opposition
    # lens (the subject benefits when the government is portrayed badly).
    governing: bool = True
    geography_pack: str = "tamil_nadu"
    languages: list[str] = Field(default_factory=lambda: ["ta", "en"])
    entities_seed: str = "configs/entities.seed.yaml"
    sources_config: str = "configs/sources.newspapers.yaml"
    mla_roster: str = "configs/mla_roster.json"
    prompt_overrides: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Password + API-key crypto (stdlib only — no new dependency)


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), _PBKDF2_ROUNDS)
    return f"pbkdf2${_PBKDF2_ROUNDS}${salt}${digest.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        scheme, rounds, salt, expected = encoded.split("$")
        if scheme != "pbkdf2":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds))
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(digest.hex(), expected)


def _hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode()).hexdigest()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (value or "").strip().lower()).strip("-")
    return slug or "tenant"


# ---------------------------------------------------------------------------
# Control plane


class ControlPlane:
    """Owns the shared control database (tenants/users/keys) and the cache of
    per-tenant session factories."""

    def __init__(self, control_database_url: str, *, tenants_dir: str | Path = "tenants") -> None:
        self.control_database_url = control_database_url
        self.tenants_dir = Path(tenants_dir)
        self._control_factory = create_session_factory(control_database_url)
        ControlBase.metadata.create_all(self._control_factory.kw["bind"])
        self._tenant_factories: dict[str, sessionmaker[Session]] = {}

    def session(self) -> Session:
        return self._control_factory()

    # -- tenant lifecycle ---------------------------------------------------

    def provision_tenant(
        self,
        *,
        name: str,
        config: TenantConfig | None = None,
        slug: str | None = None,
        database_url: str | None = None,
        seed_entities: bool = True,
    ) -> TenantRecord:
        """Create a tenant AND its own isolated database, schema and roster.

        Idempotent on slug: re-provisioning an existing tenant returns it
        without clobbering its data.
        """
        slug = slug or slugify(name)
        config = config or TenantConfig(display_name=name)
        with self.session() as session:
            existing = session.scalar(select(TenantRecord).where(TenantRecord.slug == slug))
            if existing is not None:
                return existing
            if database_url is None:
                self.tenants_dir.mkdir(parents=True, exist_ok=True)
                database_url = f"sqlite:///{self.tenants_dir / (slug + '.db')}"
            tenant = TenantRecord(
                slug=slug, name=name, database_url=database_url, config_json=config.model_dump()
            )
            session.add(tenant)
            session.commit()
            session.refresh(tenant)

        # Build the tenant's isolated database.
        factory = self._factory_for_url(database_url)
        if seed_entities:
            from tnmi.resolver import sync_seed_entities

            seed_path = Path(config.entities_seed)
            if seed_path.exists():
                with factory() as tsession:
                    sync_seed_entities(tsession, seed_path)
                    tsession.commit()
        return tenant

    def usage(self) -> list[dict[str, Any]]:
        """Per-tenant usage = rows already in each tenant's DB. The analyses ARE
        the billing record, so the meter is a row count — no events table.
        ponytail: add a `since=` window when you invoice by period."""
        from sqlalchemy import func as _func

        from tnmi.storage import AIAnalysisRecord, RawItemRecord

        out: list[dict[str, Any]] = []
        for tenant in self.list_tenants():
            with self.session_factory_for(tenant)() as session:
                analyses = session.scalar(select(_func.count()).select_from(AIAnalysisRecord)) or 0
                items = session.scalar(select(_func.count()).select_from(RawItemRecord)) or 0
            out.append(
                {
                    "slug": tenant.slug,
                    "name": tenant.name,
                    "status": tenant.status,
                    "raw_items": int(items),
                    "analyses": int(analyses),
                }
            )
        return out

    def get_tenant(self, slug: str) -> TenantRecord | None:
        with self.session() as session:
            return session.scalar(select(TenantRecord).where(TenantRecord.slug == slug))

    def list_tenants(self) -> list[TenantRecord]:
        with self.session() as session:
            return list(session.scalars(select(TenantRecord).order_by(TenantRecord.slug)))

    def tenant_config(self, tenant: TenantRecord) -> TenantConfig:
        return TenantConfig.model_validate(tenant.config_json or {})

    # -- per-tenant session factories --------------------------------------

    def _factory_for_url(self, database_url: str) -> sessionmaker[Session]:
        factory = self._tenant_factories.get(database_url)
        if factory is None:
            factory = create_session_factory(database_url)
            init_db(factory)  # idempotent: create_all + lightweight migrations
            self._tenant_factories[database_url] = factory
        return factory

    def session_factory_for(self, tenant: TenantRecord) -> sessionmaker[Session]:
        """The isolated session factory for a tenant's own database."""
        return self._factory_for_url(tenant.database_url)

    # -- users + membership -------------------------------------------------

    def create_user(self, *, email: str, name: str = "", password: str = "") -> UserRecord:
        with self.session() as session:
            existing = session.scalar(select(UserRecord).where(UserRecord.email == email.lower()))
            if existing is not None:
                return existing
            user = UserRecord(
                email=email.lower(), name=name, password_hash=hash_password(password) if password else ""
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            return user

    def authenticate_user(self, *, email: str, password: str) -> UserRecord | None:
        with self.session() as session:
            user = session.scalar(select(UserRecord).where(UserRecord.email == email.lower()))
            if user and user.password_hash and verify_password(password, user.password_hash):
                return user
            return None

    def add_membership(self, *, user: UserRecord, tenant: TenantRecord, role: str = "viewer") -> MembershipRecord:
        if role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")
        with self.session() as session:
            existing = session.scalar(
                select(MembershipRecord).where(
                    MembershipRecord.user_id == user.id, MembershipRecord.tenant_id == tenant.id
                )
            )
            if existing is not None:
                existing.role = role
                session.commit()
                session.refresh(existing)
                return existing
            membership = MembershipRecord(user_id=user.id, tenant_id=tenant.id, role=role)
            session.add(membership)
            session.commit()
            session.refresh(membership)
            return membership

    def memberships_for(self, user: UserRecord) -> list[tuple[TenantRecord, str]]:
        with self.session() as session:
            rows = session.execute(
                select(TenantRecord, MembershipRecord.role)
                .join(MembershipRecord, MembershipRecord.tenant_id == TenantRecord.id)
                .where(MembershipRecord.user_id == user.id)
                .order_by(TenantRecord.slug)
            ).all()
            return [(tenant, role) for tenant, role in rows]

    def role_of(self, *, user_id: int, tenant_id: int) -> str | None:
        with self.session() as session:
            return session.scalar(
                select(MembershipRecord.role).where(
                    MembershipRecord.user_id == user_id, MembershipRecord.tenant_id == tenant_id
                )
            )

    # -- API keys -----------------------------------------------------------

    def issue_api_key(self, *, tenant: TenantRecord, label: str = "") -> tuple[str, ApiKeyRecord]:
        """Mint a key for a tenant. The raw key is returned ONCE; only its hash
        is stored."""
        raw_key = f"{_API_KEY_PREFIX}_{secrets.token_hex(24)}"
        with self.session() as session:
            record = ApiKeyRecord(
                tenant_id=tenant.id,
                label=label,
                key_prefix=raw_key[: 12],
                key_hash=_hash_api_key(raw_key),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return raw_key, record

    def authenticate_api_key(self, raw_key: str) -> TenantRecord | None:
        """Resolve a raw API key to its tenant (and stamp last_used), or None."""
        if not raw_key:
            return None
        key_hash = _hash_api_key(raw_key.strip())
        with self.session() as session:
            record = session.scalar(select(ApiKeyRecord).where(ApiKeyRecord.key_hash == key_hash))
            if record is None:
                return None
            tenant = session.get(TenantRecord, record.tenant_id)
            if tenant is None or tenant.status != "active":
                return None
            record.last_used_at = datetime.now(timezone.utc)
            session.commit()
            session.expunge(tenant)
            return tenant
