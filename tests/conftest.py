"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_dashboard_caches():
    """Every test gets a fresh briefing cache. The dashboard module keeps a
    30-second in-process cache for the clustered narrative cards to keep
    repeat HTTP hits cheap; tests need that cache cleared between cases so
    one test's DB state doesn't leak into the next."""
    from tnmi.dashboard import invalidate_briefing_cache

    invalidate_briefing_cache()
    yield
    invalidate_briefing_cache()
