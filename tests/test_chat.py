from __future__ import annotations

from tnmi.chat import (
    ChatEvidence,
    ChatTurn,
    OllamaChatProvider,
    _build_chat_prompt,
    answer_question,
    build_retrieval_query,
    retrieve_chat_evidence,
)
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

    def answer(self, question, evidence, *, dossier_context=""):  # type: ignore[no-untyped-def]
        self.questions.append(question)
        self.evidence_titles = [item.title for item in evidence]
        self.dossier_context = dossier_context
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


def _chat_evidence(**updates: object) -> ChatEvidence:
    base: dict[str, object] = dict(
        raw_item_id=1,
        analysis_id=1,
        title="TVK members visit flood families",
        source_name="Example Daily",
        source_url="https://example.com/tvk",
        published_at=None,
        language="en",
        stance="negative",
        relevance="high",
        summary="TVK members visited flood affected families.",
        snippet="TVK members visited flood affected families",
        department="general",
        district="Chennai",
        topic="flood relief",
        confidence=0.8,
        needs_human_review=False,
    )
    base.update(updates)
    return ChatEvidence(**base)


class _FakeStreamClient:
    """Stand-in for ollama.Client whose generate() streams fragments."""

    def __init__(self, fragments: list[str]) -> None:
        self._fragments = list(fragments)
        self.calls: list[dict] = []

    def generate(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)

        def _gen():
            for fragment in self._fragments:
                yield {"response": fragment}

        return _gen()


def test_build_retrieval_query_prepends_prior_user_question():
    history = [
        ChatTurn(role="user", content="water problems in Chennai"),
        ChatTurn(role="assistant", content="Several issues were reported."),
    ]
    assert (
        build_retrieval_query("what about the second one?", history)
        == "water problems in Chennai what about the second one?"
    )
    assert build_retrieval_query("just this", []) == "just this"


def test_build_chat_prompt_includes_conversation_history():
    evidence = [_chat_evidence()]
    history = [
        ChatTurn(role="user", content="Tell me about flood relief"),
        ChatTurn(role="assistant", content="TVK visited families. [1]"),
    ]
    prompt = _build_chat_prompt(question="who paid for it?", evidence=evidence, history=history)
    assert "Conversation so far" in prompt
    assert "Tell me about flood relief" in prompt
    assert "who paid for it?" in prompt

    plain = _build_chat_prompt(question="hi there", evidence=evidence)
    assert "Conversation so far" not in plain


def test_ollama_provider_stream_answer_yields_fragments_without_dropping_spaces():
    provider = OllamaChatProvider(model="gemma2:2b", host="http://localhost:11434")
    fake = _FakeStreamClient(["The ", "flood ", "relief", "."])
    provider._client = fake  # type: ignore[attr-defined]

    out = "".join(provider.stream_answer("q", [_chat_evidence()]))

    assert out == "The flood relief."
    assert fake.calls[0]["stream"] is True
    assert fake.calls[0]["model"] == "gemma2:2b"


def test_ollama_provider_stream_answer_refuses_without_evidence():
    provider = OllamaChatProvider(model="gemma2:2b", host="http://localhost:11434")
    provider._client = _FakeStreamClient(["unused"])  # type: ignore[attr-defined]

    out = "".join(provider.stream_answer("q", []))

    assert "do not have enough evidence" in out.lower()


def test_resolve_question_entities_and_dossier_context(tmp_path):
    """Naming an entity in the question pulls its aggregate dossier brief —
    the over-time context that keyword retrieval cannot give."""
    from tnmi.chat import build_dossier_context, resolve_question_entities
    from tnmi.contracts import GovernmentRelevance, Stance
    from tnmi.entity_api import invalidate_entity_cache
    from tnmi.resolver import resolve_all
    from tnmi.storage import (
        create_session_factory,
        init_db,
        save_ai_analysis,
        save_raw_item,
    )
    from tests.test_storage import make_analysis, make_item

    factory = create_session_factory(f"sqlite:///{tmp_path / 'chat-entity.db'}")
    init_db(factory)
    with factory() as session:
        for i in range(3):
            raw = save_raw_item(
                session,
                make_item().model_copy(
                    update={
                        "source_url": f"https://e.com/v{i}",
                        "title": "Vijay scheme",
                        "raw_text_original": "Vijay launched a welfare scheme in Chennai.",
                        "clean_text_original": "Vijay launched a welfare scheme in Chennai.",
                    }
                ),
            )
            save_ai_analysis(
                session,
                raw.id,
                make_analysis().model_copy(
                    update={
                        "political_actors": ["Vijay (CM)"],
                        "tvk_portrayal": Stance.POSITIVE,
                        "government_relevance": GovernmentRelevance.HIGH,
                    }
                ),
                model_name="mock",
                prompt_version="v19",
            )
        session.commit()
        resolve_all(session, seed_path="configs/entities.seed.yaml")
        session.commit()
        invalidate_entity_cache()

        assert resolve_question_entities(session, "How is Vijay being covered?") == ["vijay"]
        assert resolve_question_entities(session, "tell me about cricket scores") == []

        context = build_dossier_context(session, "How is Vijay being covered?")
        assert "Vijay" in context
        assert "mentions" in context
        assert "favourability" in context
