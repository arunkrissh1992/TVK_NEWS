from __future__ import annotations

from tnmi.chat import answer_question, retrieve_chat_evidence
from tnmi.contracts import AIAnalysis, NormalizedItem, SourceType
from tnmi.storage import create_session_factory, init_db, save_ai_analysis, save_raw_item
from tests.test_storage import make_analysis


def _item(*, url: str, title: str, text: str) -> NormalizedItem:
    return NormalizedItem(
        source_type=SourceType.NEWS,
        source_name="Example Tamil Daily",
        source_url=url,
        published_at=None,
        language="en",
        title=title,
        raw_text_original=text,
        clean_text_original=text,
    )


def _analysis(**updates: object) -> AIAnalysis:
    data = make_analysis().model_dump()
    data.update(updates)
    return AIAnalysis.model_validate(data)


class _FakeChatProvider:
    model_name = "fake-ai"

    def __init__(self) -> None:
        self.questions: list[str] = []
        self.evidence_titles: list[str] = []

    def answer(self, question, evidence):  # type: ignore[no-untyped-def]
        self.questions.append(question)
        self.evidence_titles = [item.title for item in evidence]
        return "AI answer grounded in: " + ", ".join(self.evidence_titles)


def test_retrieve_chat_evidence_ranks_matching_articles_first(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'chat.db'}")
    init_db(session_factory)
    with session_factory() as session:
        road = save_raw_item(
            session,
            _item(
                url="https://example.com/road",
                title="Chennai road repairs after public complaints",
                text="Residents reported damaged Chennai roads and officials promised repairs.",
            ),
        )
        save_ai_analysis(
            session,
            road.id,
            _analysis(
                stance_toward_government="negative",
                topic="Chennai road repairs",
                summary_english="Road damage complaints were reported in Chennai.",
                negative_points=["damaged roads"],
                evidence_quotes_original=["Residents reported damaged Chennai roads"],
                district="Chennai",
            ),
            model_name="mock",
            prompt_version="v1",
        )
        welfare = save_raw_item(
            session,
            _item(
                url="https://example.com/welfare",
                title="Welfare camp receives public appreciation",
                text="Families thanked officials after a welfare camp helped residents.",
            ),
        )
        save_ai_analysis(
            session,
            welfare.id,
            _analysis(
                topic="welfare camp",
                summary_english="A welfare camp received appreciation.",
                positive_points=["public appreciation"],
                evidence_quotes_original=["Families thanked officials"],
            ),
            model_name="mock",
            prompt_version="v1",
        )
        session.commit()

        evidence = retrieve_chat_evidence(session, "What are the Chennai road complaints?", limit=2)

    assert [item.title for item in evidence] == ["Chennai road repairs after public complaints"]
    assert evidence[0].source_url == "https://example.com/road"
    assert "Residents reported damaged Chennai roads" in evidence[0].snippet


def test_answer_question_uses_ai_provider_with_retrieved_evidence(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'answer.db'}")
    init_db(session_factory)
    provider = _FakeChatProvider()
    with session_factory() as session:
        raw = save_raw_item(
            session,
            _item(
                url="https://example.com/tvk",
                title="TVK members visit flood affected families",
                text="TVK members visited flood affected families and requested relief support.",
            ),
        )
        save_ai_analysis(
            session,
            raw.id,
            _analysis(
                target="TVK members",
                topic="flood relief visit",
                party_action="TVK members visited flood affected families.",
                people_impact="Flood affected families requested relief support.",
                summary_english="TVK members visited flood affected families.",
                evidence_quotes_original=["TVK members visited flood affected families"],
            ),
            model_name="mock",
            prompt_version="v1",
        )
        session.commit()

        answer = answer_question(session, "What did TVK members do for flood families?", provider=provider)

    assert answer.used_ai is True
    assert answer.model_name == "fake-ai"
    assert "AI answer grounded in" in answer.answer
    assert provider.questions == ["What did TVK members do for flood families?"]
    assert provider.evidence_titles == ["TVK members visit flood affected families"]
    assert answer.evidence[0].source_url == "https://example.com/tvk"


def test_answer_question_refuses_when_no_stored_evidence_matches(tmp_path):
    session_factory = create_session_factory(f"sqlite:///{tmp_path / 'empty.db'}")
    init_db(session_factory)
    provider = _FakeChatProvider()
    with session_factory() as session:
        answer = answer_question(session, "What happened in a cricket match?", provider=provider)

    assert answer.used_ai is False
    assert answer.model_name == "evidence-only"
    assert "do not have enough evidence" in answer.answer.lower()
    assert answer.evidence == []
    assert provider.questions == []
