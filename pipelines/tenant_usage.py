"""Per-tenant usage report — the billing input (analyses processed per tenant).

    python -m pipelines.tenant_usage
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tnmi.config import Settings
from tnmi.tenancy import ControlPlane


def main() -> int:
    settings = Settings()
    control = ControlPlane(settings.control_database_url, tenants_dir=settings.tenants_dir)
    rows = control.usage()
    if not rows:
        print("No tenants provisioned.")
        return 0
    print(f"{'tenant':24} {'status':10} {'articles':>9} {'analyses':>9}")
    for r in rows:
        print(f"{r['slug']:24} {r['status']:10} {r['raw_items']:>9} {r['analyses']:>9}")
    print(f"\ntotal analyses billed: {sum(r['analyses'] for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
