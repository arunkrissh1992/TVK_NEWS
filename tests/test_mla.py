from tnmi.mla import (
    load_roster,
    mlas_by_district,
    party_css,
    party_seat_counts,
    party_short,
    roster_label,
)


def test_roster_loads_full_assembly():
    roster = load_roster()
    members = roster["members"]
    assert len(members) == 234
    numbers = {m["no"] for m in members}
    assert numbers == set(range(1, 235))
    # The CURRENT assembly — elected 2026, TVK government.
    assert roster_label().startswith("17th Tamil Nadu Legislative Assembly")


def test_every_member_resolves_to_a_canonical_district():
    grouped = mlas_by_district()
    total = sum(len(v) for v in grouped.values())
    assert total == 234  # nobody dropped by district-name normalization
    # Chennai must carry its well-known constituencies.
    chennai = {m["constituency"] for m in grouped["Chennai"]}
    assert "Kolathur" in chennai
    assert "Mylapore" in chennai


def test_party_seat_counts_sum_to_234_and_rank_tvk_first():
    counts = party_seat_counts()
    assert sum(row["seats"] for row in counts) == 234
    # 2026 result: TVK is the ruling party.
    assert counts[0]["party_short"] == "TVK"
    assert counts[0]["seats"] > 100


def test_chief_minister_is_vijay_of_tvk():
    roster = load_roster()
    cm_rows = [m for m in roster["members"] if m["mla"] == "C. Joseph Vijay"]
    assert cm_rows, "the CM must appear in the roster"
    assert all(m["party"] == "Tamilaga Vettri Kazhagam" for m in cm_rows)


def test_party_short_and_css_mappings():
    assert party_short("Dravida Munnetra Kazhagam") == "DMK"
    assert party_css("All India Anna Dravida Munnetra Kazhagam") == "aiadmk"
    assert party_css("Tamilaga Vettri Kazhagam") == "tvk"
    # Unknown parties degrade gracefully.
    assert party_css("Some New Party") == "other"
    assert party_short("Some New Party") == "Some New Party"
