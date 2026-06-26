"""Email today's intelligence brief to the leadership office.

The brief is the same "so what" the dashboard shows on top — this just pushes it
so leadership receives it each morning instead of having to open the app.

Config via env (no secrets in code):
    SMTP_HOST, SMTP_PORT (default 587), SMTP_USER, SMTP_PASS
    BRIEF_FROM, BRIEF_TO (comma-separated)

    python -m pipelines.send_brief --dry-run    # print, don't send
    python -m pipelines.send_brief               # send via SMTP
"""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
from datetime import date
from email.mime.text import MIMEText
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tnmi.config import Settings
from tnmi.dashboard import build_daily_brief
from tnmi.storage import create_session_factory


_TONE_MARK = {"good": "✅", "bad": "🔴", "watch": "🟡"}


def render_text(lines: list[dict]) -> str:
    out = [f"TVK Intelligence Brief — {date.today().isoformat()}", ""]
    for line in lines:
        mark = _TONE_MARK.get(line.get("tone"), "•")
        detail = f" — {line['detail']}" if line.get("detail") else ""
        out.append(f"{mark} {line['title']}{detail}")
    if not lines:
        out.append("No notable signals today.")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Email today's intelligence brief.")
    parser.add_argument("--dry-run", action="store_true", help="print instead of sending")
    args = parser.parse_args(argv)

    settings = Settings()
    factory = create_session_factory(settings.database_url)
    with factory() as session:
        lines = build_daily_brief(session)
    body = render_text(lines)

    if args.dry_run:
        print(body)
        return 0

    host = os.getenv("SMTP_HOST")
    recipients = [r.strip() for r in os.getenv("BRIEF_TO", "").split(",") if r.strip()]
    if not host or not recipients:
        print("SMTP_HOST and BRIEF_TO must be set to send (or use --dry-run).", file=sys.stderr)
        return 2

    msg = MIMEText(body)
    msg["Subject"] = f"TVK Intelligence Brief — {date.today().isoformat()}"
    msg["From"] = os.getenv("BRIEF_FROM", os.getenv("SMTP_USER", "tvk-intel@localhost"))
    msg["To"] = ", ".join(recipients)

    with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587"))) as smtp:
        smtp.starttls()
        if os.getenv("SMTP_USER"):
            smtp.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASS", ""))
        smtp.send_message(msg)
    print(f"Sent brief to {len(recipients)} recipient(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
