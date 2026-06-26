from tnmi.responsibility import resolve_responsibility

# A fixed district roster so the test doesn't depend on the shipped JSON.
_ROSTER = {
    "Chennai": [
        {"no": 1, "constituency": "Egmore", "mla": "A", "party": "TVK",
         "party_short": "TVK", "party_css": "tvk"},
        {"no": 2, "constituency": "Mylapore", "mla": "B", "party": "DMK",
         "party_short": "DMK", "party_css": "dmk"},
    ],
}


def test_people_issue_maps_to_department_district_and_cm():
    card = {
        "public_issue": "drinking water shortage",
        "department": "general",
        "district": "Chennai",
        "district_canonical": "Chennai",
        "people_issue": True,
        "portrayal_kind": "people",
        "action_priority": "high",
        "title": "Chennai water shortage",
    }
    r = resolve_responsibility(card, _ROSTER)
    assert r["actionable"] is True
    assert r["department"] == "Municipal Admin & Water Supply"
    assert r["district"] == "Chennai"
    assert r["collector"] == "Chennai Collector"
    assert r["mla_count"] == 2
    assert r["escalate_cm"] is True  # high priority problem → CM


def test_low_priority_problem_does_not_escalate_to_cm():
    card = {
        "public_issue": "road or traffic issue",
        "district": "Salem",
        "district_canonical": "Salem",
        "people_issue": True,
        "portrayal_kind": "people",
        "action_priority": "low",
    }
    r = resolve_responsibility(card, _ROSTER)
    assert r["department"] == "Highways & Transport"
    assert r["escalate_cm"] is False


def test_positive_coverage_is_not_actionable():
    card = {
        "public_issue": "",
        "department": "general",
        "district": "unspecified",
        "district_canonical": "",
        "people_issue": False,
        "portrayal_kind": "positive",
        "action_priority": "low",
        "title": "CM inaugurates new park",
    }
    r = resolve_responsibility(card, _ROSTER)
    assert r["actionable"] is False
    assert r["department"] == ""
    assert r["mlas"] == []


def test_explicit_department_used_when_no_keyword_match():
    card = {
        "public_issue": "",
        "department": "tourism",
        "district": "unspecified",
        "district_canonical": "",
        "people_issue": False,
        "portrayal_kind": "negative",
        "action_priority": "medium",
        "title": "Tourism board criticised",
    }
    r = resolve_responsibility(card, _ROSTER)
    assert r["department"] == "Tourism"
    assert r["actionable"] is True
