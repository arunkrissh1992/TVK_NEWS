from tnmi.contracts import GovernmentRelevance, Severity, Stance
from tnmi.quality import audit_analysis_quality
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item
from tests.test_storage import make_analysis, make_item


def test_quality_audit_flags_school_fire_with_generic_action(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'quality.db'}")
    init_db(session_factory)
    school_fire_text = (
        "சென்னை பள்ளி அருகே தீ விபத்து ஏற்பட்டதால் மாணவர்கள் வெளியேற்றப்பட்டனர். "
        "பெற்றோர் பாதுகாப்பு குறித்து கவலை தெரிவித்தனர். "
        "தீயணைப்பு வீரர்கள் சம்பவ இடத்தில் பணியில் ஈடுபட்டனர். "
        "தமிழகத்தில் உள்ளூர் அதிகாரிகள் நடவடிக்கை எடுக்க வேண்டும் என்று மக்கள் கூறினர்."
    )
    item = make_item().model_copy(
        update={
            "source_name": "Polimer News",
            "source_url": "https://example.com/school-fire",
            "title": "சென்னை பள்ளி அருகே தீ விபத்து",
            "raw_text_original": school_fire_text,
            "clean_text_original": school_fire_text,
        }
    )
    bad_analysis = make_analysis().model_copy(
        update={
            "government_relevance": GovernmentRelevance.LOW,
            "stance_toward_government": Stance.NEUTRAL,
            "tvk_portrayal": Stance.NEUTRAL,
            "severity": Severity.LOW,
            "people_issue": False,
            "recommended_step": "Address the concern through the appropriate department.",
            "action_owner": "",
            "needs_human_review": False,
        }
    )

    with session_factory() as session:
        raw = save_raw_item(session, item)
        save_ai_analysis(session, raw.id, bad_analysis, model_name="mock", prompt_version="old")
        issues = audit_analysis_quality(session)
        session.commit()

    issue_codes = {issue["issue_code"] for issue in issues}
    assert "people_issue_missing" in issue_codes
    assert "serious_issue_low_severity" in issue_codes
    assert "serious_issue_not_reviewed" in issue_codes
    assert "generic_next_step" in issue_codes
    assert "missing_action_owner" in issue_codes


def test_quality_audit_flags_direct_vijay_story_with_low_tvk_relevance(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'quality.db'}")
    init_db(session_factory)
    text = (
        "தலைவர் விஜய் தலைமையிலான தவெக நிர்வாகிகள் சென்னையில் ஆலோசனை கூட்டம் நடத்தினர். "
        "தமிழ்நாட்டின் பல மாவட்ட நிர்வாகிகள் கூட்டத்தில் கலந்து கொண்டனர். "
        "கட்சியின் அடுத்த கட்ட பணிகள் குறித்து ஆலோசிக்கப்பட்டது."
    )
    item = make_item().model_copy(
        update={
            "source_name": "Example News",
            "source_url": "https://example.com/vijay",
            "title": "தலைவர் விஜய் தலைமையில் தவெக ஆலோசனை",
            "raw_text_original": text,
            "clean_text_original": text,
        }
    )
    bad_analysis = make_analysis().model_copy(
        update={
            "government_relevance": GovernmentRelevance.MEDIUM,
            "tvk_relevance": GovernmentRelevance.MEDIUM,
            "tvk_portrayal": Stance.NEUTRAL,
            "people_issue": False,
        }
    )

    with session_factory() as session:
        raw = save_raw_item(session, item)
        save_ai_analysis(session, raw.id, bad_analysis, model_name="mock", prompt_version="old")
        issues = audit_analysis_quality(session)
        session.commit()

    issue_codes = {issue["issue_code"] for issue in issues}
    assert "tvk_relevance_missing" in issue_codes


def test_quality_audit_does_not_force_people_issue_for_school_ceremony(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'quality.db'}")
    init_db(session_factory)
    text = (
        "சென்னை மாநகராட்சி பள்ளி கட்டிட திறப்பு விழாவில் மேயரும் எம்.எல்.ஏவும் கலந்து கொண்டனர். "
        "கட்டிட வாயிலில் ரிப்பன் வெட்டி திறந்து வைத்தனர். "
        "நிகழ்ச்சியில் அதிகாரிகள் மற்றும் கட்சி நிர்வாகிகள் கலந்து கொண்டனர்."
    )
    item = make_item().model_copy(
        update={
            "source_name": "Example News",
            "source_url": "https://example.com/school-ceremony",
            "title": "பள்ளி கட்டிட திறப்பு விழா",
            "raw_text_original": text,
            "clean_text_original": text,
        }
    )
    analysis = make_analysis().model_copy(
        update={
            "government_relevance": GovernmentRelevance.MEDIUM,
            "people_issue": False,
            "recommended_step": "Media monitoring desk: monitor if parent or student service impact emerges.",
        }
    )

    with session_factory() as session:
        raw = save_raw_item(session, item)
        save_ai_analysis(session, raw.id, analysis, model_name="mock", prompt_version="old")
        issues = audit_analysis_quality(session)
        session.commit()

    issue_codes = {issue["issue_code"] for issue in issues}
    assert "people_issue_missing" not in issue_codes


def test_quality_audit_does_not_force_tvk_high_for_generic_cm_story(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'quality.db'}")
    init_db(session_factory)
    text = (
        "The Chief Minister reviewed awards for police personnel in the city. "
        "Officials said the department will publish the list after verification. "
        "The report did not mention any party organisation."
    )
    item = make_item().model_copy(
        update={
            "source_name": "Example News",
            "source_url": "https://example.com/cm-review",
            "title": "Chief Minister reviews police awards",
            "raw_text_original": text,
            "clean_text_original": text,
        }
    )
    analysis = make_analysis().model_copy(
        update={
            "government_relevance": GovernmentRelevance.MEDIUM,
            "tvk_relevance": GovernmentRelevance.MEDIUM,
            "people_issue": False,
            "recommended_step": "Policy research team: monitor award-list publication for public impact.",
        }
    )

    with session_factory() as session:
        raw = save_raw_item(session, item)
        save_ai_analysis(session, raw.id, analysis, model_name="mock", prompt_version="old")
        issues = audit_analysis_quality(session)
        session.commit()

    issue_codes = {issue["issue_code"] for issue in issues}
    assert "tvk_relevance_missing" not in issue_codes
