"""Provision a new SaaS tenant — an isolated workspace for one party/candidate.

Creates the tenant's OWN database (silo isolation), seeds its entity roster,
records its political lens, and mints a first API key. The raw key is printed
ONCE — store it; it is never recoverable.

Examples:

    python -m pipelines.provision_tenant --name "TVK War Room" --party TVK --governing
    python -m pipelines.provision_tenant --name "Party X" --slug party-x \
        --party "Party X" --subject-slug their-leader --opposition
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
from tnmi.tenancy import ControlPlane, TenantConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provision a new SaaS tenant.")
    parser.add_argument("--name", required=True, help="display name, e.g. 'TVK War Room'")
    parser.add_argument("--slug", default=None, help="url-safe id (default: slugified name)")
    parser.add_argument("--party", default="TVK", help="subject party this tenant tracks")
    parser.add_argument("--subject-slug", default="vijay", help="entity slug of the party's leader")
    gov = parser.add_mutually_exclusive_group()
    gov.add_argument("--governing", action="store_true", help="subject party holds government (default)")
    gov.add_argument("--opposition", action="store_true", help="subject party is in opposition")
    parser.add_argument("--geography", default="tamil_nadu", help="geography pack")
    parser.add_argument("--no-key", action="store_true", help="do not mint an API key")
    parser.add_argument("--no-seed", action="store_true", help="skip entity-roster seeding")
    args = parser.parse_args(argv)

    settings = Settings()
    control = ControlPlane(settings.control_database_url, tenants_dir=settings.tenants_dir)
    config = TenantConfig(
        display_name=args.name,
        subject_party=args.party,
        subject_entity_slug=args.subject_slug,
        governing=not args.opposition,
        geography_pack=args.geography,
    )
    tenant = control.provision_tenant(
        name=args.name, slug=args.slug, config=config, seed_entities=not args.no_seed
    )
    print(f"Provisioned tenant '{tenant.slug}'")
    print(f"  database : {tenant.database_url}")
    print(f"  lens     : party={config.subject_party}, governing={config.governing}, geo={config.geography_pack}")
    if not args.no_key:
        raw_key, _ = control.issue_api_key(tenant=tenant, label="initial")
        print(f"  API key  : {raw_key}")
        print("  ^ store this now — it is shown only once and cannot be recovered.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
