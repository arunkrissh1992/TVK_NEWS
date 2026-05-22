from __future__ import annotations

import re

TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "unknown"

    tamil_count = len(TAMIL_RE.findall(stripped))
    latin_count = len(LATIN_RE.findall(stripped))

    if tamil_count and latin_count:
        return "ta-en-mixed"
    if tamil_count:
        return "ta"
    if latin_count:
        return "en"
    return "unknown"
