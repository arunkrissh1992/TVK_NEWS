from tnmi.districts import (
    DISTRICT_TILES,
    canonical_department,
    canonical_district,
    summarize_by_department,
    summarize_by_district,
)


def test_district_tiles_cover_all_38_districts_with_unique_positions():
    assert len(DISTRICT_TILES) == 38
    positions = [(col, row) for col, row, _short in DISTRICT_TILES.values()]
    assert len(set(positions)) == 38  # no two districts share a tile


def test_canonical_district_matches_exact_and_case_insensitive():
    assert canonical_district("Chennai") == "Chennai"
    assert canonical_district("chennai") == "Chennai"
    assert canonical_district("MADURAI") == "Madurai"


def test_canonical_district_resolves_common_variants():
    assert canonical_district("Trichy") == "Tiruchirappalli"
    assert canonical_district("Tuticorin") == "Thoothukudi"
    assert canonical_district("Kanchipuram") == "Kancheepuram"
    assert canonical_district("Villupuram") == "Viluppuram"
    assert canonical_district("Madurai district") == "Madurai"


def test_canonical_district_resolves_tamil_names():
    assert canonical_district("சென்னை") == "Chennai"
    assert canonical_district("மதுரை மாவட்டம்") == "Madurai"
    assert canonical_district("கோவை") == "Coimbatore"


def test_canonical_district_rejects_non_districts():
    assert canonical_district("unspecified") is None
    assert canonical_district("") is None
    assert canonical_district(None) is None
    assert canonical_district("Mumbai") is None
    assert canonical_district("Tamil Nadu") is None


def _item(district, category, issue=""):
    return {
        "district": district,
        "display_category": category,
        "public_issue": issue,
        "department": "health" if issue else "general",
    }


def test_summarize_by_district_counts_categories_and_issues():
    items = [
        _item("Chennai", "negative", "water shortage"),
        _item("chennai", "people", "water shortage"),
        _item("Trichy", "positive"),
        _item("unspecified", "neutral"),  # → unmapped
        _item("Mumbai", "negative"),  # → unmapped
    ]
    summary = summarize_by_district(items)
    by_name = {t["district"]: t for t in summary["tiles"]}

    chennai = by_name["Chennai"]
    assert chennai["total"] == 2
    assert chennai["negative"] == 1
    assert chennai["people"] == 1
    assert chennai["dominant"] == "negative"  # negative outranks people
    assert chennai["top_issues"][0] == {"issue": "water shortage", "count": 2}

    trichy = by_name["Tiruchirappalli"]
    assert trichy["total"] == 1
    assert trichy["dominant"] == "positive"

    assert by_name["Madurai"]["total"] == 0
    assert by_name["Madurai"]["dominant"] == "quiet"
    assert summary["unmapped_total"] == 2
    assert summary["mapped_total"] == 3
    assert len(summary["tiles"]) == 38  # every district always renders


def test_canonical_department_matches_keyword_families():
    # Health, police and government families all collapse to one canonical name.
    assert canonical_department("health") == "Health & Family Welfare"
    assert canonical_department("Hospital administration") == "Health & Family Welfare"
    assert canonical_department("Police") == "Home (Police)"
    assert canonical_department("Law enforcement") == "Home (Police)"
    assert canonical_department("Law/order") == "Home (Police)"
    assert canonical_department("Law and order") == "Home (Police)"


def test_canonical_department_collapses_government_variants():
    # The most fragmented bucket — every phrasing lands on one rail.
    for variant in (
        "Tamil Nadu Government",
        "Government of Tamil Nadu",
        "TN Govt",
        "State Government",
        "Secretariat",
    ):
        assert canonical_department(variant) == "Government of Tamil Nadu", variant


def test_canonical_department_strips_affixes_and_normalises_unknowns():
    # Unrecognised departments survive, but distinct free-text spellings of the
    # same office group together after affix-stripping + case normalisation.
    assert canonical_department("Department of Sports") == "Sports"
    assert canonical_department("Sports Department") == "Sports"
    assert canonical_department("sports") == "Sports"
    assert canonical_department("SPORTS") == "Sports"


def test_canonical_department_rejects_generic_placeholders():
    assert canonical_department("general") is None
    assert canonical_department("Unspecified") is None
    assert canonical_department("department") is None
    assert canonical_department("") is None
    assert canonical_department(None) is None


def test_summarize_by_department_ranks_by_volume():
    items = [
        _item("Chennai", "negative", "hospital staffing"),  # health
        _item("Madurai", "people", "clinic shortage"),  # health
        _item("Salem", "positive"),  # general → not a real department, dropped
    ]
    departments = summarize_by_department(items)
    # "general" is a placeholder, so only the health rail survives.
    assert len(departments) == 1
    assert departments[0]["department"] == "Health & Family Welfare"
    assert departments[0]["label"] == "Health & Family Welfare"
    assert departments[0]["total"] == 2
    assert departments[0]["negative"] == 1
    assert departments[0]["people"] == 1
    assert departments[0]["dominant"] == "negative"


def test_summarize_by_department_deduplicates_near_duplicate_names():
    # Four free-text names the Gemma LLM might emit collapse to two rails.
    items = [
        {"district": "Chennai", "display_category": "negative",
         "department": "Tamil Nadu Government"},
        {"district": "Madurai", "display_category": "people",
         "department": "Government of Tamil Nadu"},
        {"district": "Salem", "display_category": "negative",
         "department": "Law enforcement"},
        {"district": "Trichy", "display_category": "negative",
         "department": "Police"},
    ]
    departments = summarize_by_department(items)
    labels = [d["department"] for d in departments]
    # Tie on total (2 each) → alphabetical by canonical name.
    assert labels == ["Government of Tamil Nadu", "Home (Police)"]
    by_dept = {d["department"]: d for d in departments}
    assert by_dept["Government of Tamil Nadu"]["total"] == 2
    assert by_dept["Home (Police)"]["total"] == 2
    # The rail row's data-department equals its display label, and both equal the
    # canonical name the cards carry — so the data-department filter still matches.
    for dept in departments:
        assert dept["department"] == dept["label"]


def test_detect_district_finds_english_and_tamil_mentions():
    from tnmi.districts import detect_district

    assert detect_district("Protest in Madurai over water supply", "") == "Madurai"
    assert detect_district("", "மதுரையில் போராட்டம் நடைபெற்றது") == "Madurai"
    # Title mention wins over a different district in the body.
    assert detect_district("Salem hospital upgraded", "Patients from Erode visited") == "Salem"
    assert detect_district("Trichy corporation news", "") == "Tiruchirappalli"


def test_detect_district_is_not_fooled_by_substrings():
    from tnmi.districts import detect_district

    # 'theni' inside "strengthening", 'salem' inside "Jerusalem" — the exact
    # bugs that let international stories onto the TN map.
    assert detect_district("Pakistan strengthening ties with Iran", "") is None
    assert detect_district("Peace talks in Jerusalem continue", "") is None
    # 'erode' the verb is not Erode the district…
    assert detect_district("Scandals erode public trust", "") is None
    # …but the capitalised place name is.
    assert detect_district("Erode farmers stage protest", "") == "Erode"
