from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Iterator, Literal, Protocol, Sequence

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from tnmi.entity_api import entity_dossier
from tnmi.resolver import normalize_surface
from tnmi.storage import AIAnalysisRecord, EntityAliasRecord, EntityRecord, RawItemRecord


logger = logging.getLogger(__name__)

_MAX_QUESTION_CHARS = 1000
_DEFAULT_LIMIT = 5
_MAX_LIMIT = 8
_DEFAULT_CANDIDATE_LIMIT = 1000
_MAX_CANDIDATE_LIMIT = 5000
_INSUFFICIENT_ANSWER = (
    "I do not have enough evidence in the stored newspaper data to answer that. "
    "Please pull the latest sources or ask about a topic already visible in the briefing."
)
_TOKEN_RE = re.compile(r"[\w\u0B80-\u0BFF]+", re.UNICODE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "about",
    "based",
    "brief",
    "briefing",
    "can",
    "did",
    "do",
    "does",
    "for",
    "from",
    "give",
    "happen",
    "happened",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "latest",
    "me",
    "of",
    "on",
    "please",
    "report",
    "show",
    "tell",
    "that",
    "the",
    "there",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


class ChatEvidence(BaseModel):
    raw_item_id: int
    analysis_id: int
    title: str
    source_name: str
    source_url: str
    published_at: datetime | None = None
    language: str
    stance: str
    relevance: str
    summary: str
    snippet: str
    department: str
    district: str
    topic: str
    confidence: float
    needs_human_review: bool


class ChatAnswer(BaseModel):
    answer: str
    evidence: list[ChatEvidence]
    model_name: str
    used_ai: bool


class ChatTurn(BaseModel):
    """One prior message in the conversation, used for multi-turn context.

    Only the last few turns are sent to the model — enough to resolve
    references like "the second one" without bloating the prompt.
    """

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatAIProvider(Protocol):
    model_name: str

    def answer(
        self,
        question: str,
        evidence: Sequence[ChatEvidence],
        *,
        dossier_context: str = "",
    ) -> str:
        ...


class ChatAIUnavailable(RuntimeError):
    """Raised when a configured chat model cannot produce an answer."""


class EvidenceOnlyChatProvider:
    """Deterministic fallback used when no AI model is reachable.

    It does not pretend to be a model. The caller sets used_ai=False so the
    dashboard can display the correct operational state.
    """

    model_name = "evidence-only"

    def answer(
        self,
        question: str,
        evidence: Sequence[ChatEvidence],
        *,
        dossier_context: str = "",
    ) -> str:
        del question
        if not evidence:
            return _INSUFFICIENT_ANSWER

        lines: list[str] = []
        if dossier_context.strip():
            lines.append("From the knowledge graph (aggregate across all coverage):")
            lines.extend(dossier_context.strip().splitlines())
            lines.append("")
        lines.append("From the stored newspaper evidence:")
        for index, item in enumerate(evidence[:3], start=1):
            summary = item.summary or item.snippet
            line = f"{index}. {summary} ({item.source_name}, {item.stance})."
            if item.snippet and item.snippet != summary:
                line += f" Evidence: {item.snippet}"
            lines.append(line)
        lines.append("Please open the linked source before using this in an official note.")
        return "\n".join(lines)


class OllamaChatProvider:
    """Grounded chat generation through the configured Ollama model."""

    def __init__(
        self,
        *,
        model: str,
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        try:
            from ollama import Client  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ChatAIUnavailable("ollama python package is not installed") from exc

        self._client = Client(host=host, timeout=timeout)
        self._model = model
        self.model_name = f"ollama/{model}"

    def answer(
        self,
        question: str,
        evidence: Sequence[ChatEvidence],
        *,
        dossier_context: str = "",
    ) -> str:
        if not evidence:
            return _INSUFFICIENT_ANSWER

        prompt = _build_chat_prompt(
            question=question, evidence=evidence, dossier_context=dossier_context
        )
        try:
            response = self._client.generate(
                model=self._model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": 0.1,
                    "num_predict": 700,
                },
            )
        except Exception as exc:  # noqa: BLE001 - external AI runtime boundary
            raise ChatAIUnavailable(f"Ollama chat generation failed: {exc.__class__.__name__}") from exc

        text = _ollama_response_text(response)
        if not text:
            raise ChatAIUnavailable("Ollama returned an empty chat answer")
        return text

    def stream_answer(
        self,
        question: str,
        evidence: Sequence[ChatEvidence],
        *,
        history: Sequence[ChatTurn] | None = None,
        dossier_context: str = "",
    ) -> Iterator[str]:
        """Yield the grounded answer as token fragments as the model produces
        them, so the UI can render a live, typing-style response instead of a
        ~10 s blank wait. Raises ChatAIUnavailable if the daemon is unreachable
        so the caller can fall back to the evidence-only answer."""
        if not evidence:
            yield _INSUFFICIENT_ANSWER
            return

        prompt = _build_chat_prompt(
            question=question, evidence=evidence, history=history, dossier_context=dossier_context
        )
        try:
            stream = self._client.generate(
                model=self._model,
                prompt=prompt,
                stream=True,
                options={
                    "temperature": 0.1,
                    "num_predict": 700,
                },
            )
            for chunk in stream:
                # NB: do NOT strip — chunk text carries the spaces between tokens.
                fragment = _ollama_chunk_text(chunk)
                if fragment:
                    yield fragment
        except Exception as exc:  # noqa: BLE001 - external AI runtime boundary
            raise ChatAIUnavailable(
                f"Ollama chat streaming failed: {exc.__class__.__name__}"
            ) from exc


def retrieve_chat_evidence(
    session: Session,
    question: str,
    *,
    limit: int = _DEFAULT_LIMIT,
    candidate_limit: int = _DEFAULT_CANDIDATE_LIMIT,
) -> list[ChatEvidence]:
    query = _clean_question(question)
    terms = _query_terms(query)
    if not terms:
        return []

    bounded_limit = max(1, min(limit, _MAX_LIMIT))
    bounded_candidate_limit = max(
        bounded_limit,
        min(candidate_limit, _MAX_CANDIDATE_LIMIT),
    )
    rows = session.execute(
        select(RawItemRecord, AIAnalysisRecord)
        .join(AIAnalysisRecord, AIAnalysisRecord.raw_item_id == RawItemRecord.id)
        .where(RawItemRecord.source_type == "news")
        .where(AIAnalysisRecord.government_relevance != "none")
        .order_by(AIAnalysisRecord.created_at.desc(), AIAnalysisRecord.id.desc())
        .limit(bounded_candidate_limit)
    ).all()

    seen_raw_ids: set[int] = set()
    scored: list[tuple[int, int, ChatEvidence]] = []
    for raw, analysis in rows:
        if raw.id in seen_raw_ids:
            continue
        seen_raw_ids.add(raw.id)
        score = _score_candidate(raw=raw, analysis=analysis, terms=terms, exact_query=query)
        if score <= 0:
            continue
        evidence = _to_evidence(raw=raw, analysis=analysis, terms=terms)
        scored.append((score, raw.id, evidence))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored[:bounded_limit]]


def answer_question(
    session: Session,
    question: str,
    *,
    provider: ChatAIProvider | None,
    limit: int = _DEFAULT_LIMIT,
) -> ChatAnswer:
    clean_question = _clean_question(question)
    evidence = retrieve_chat_evidence(session, clean_question, limit=limit)
    if not evidence:
        return ChatAnswer(
            answer=_INSUFFICIENT_ANSWER,
            evidence=[],
            model_name=EvidenceOnlyChatProvider.model_name,
            used_ai=False,
        )

    dossier_context = build_dossier_context(session, clean_question)

    if provider is not None:
        try:
            answer = provider.answer(
                clean_question, evidence, dossier_context=dossier_context
            ).strip()
            if answer:
                return ChatAnswer(
                    answer=answer,
                    evidence=evidence,
                    model_name=provider.model_name,
                    used_ai=True,
                )
        except Exception as exc:  # noqa: BLE001 - provider fallback boundary
            logger.warning("Chat AI provider failed; using evidence-only fallback: %s", exc.__class__.__name__)

    fallback = EvidenceOnlyChatProvider()
    return ChatAnswer(
        answer=fallback.answer(clean_question, evidence, dossier_context=dossier_context),
        evidence=evidence,
        model_name=fallback.model_name,
        used_ai=False,
    )


def _clean_question(question: str) -> str:
    return " ".join((question or "").strip().split())[:_MAX_QUESTION_CHARS]


def _query_terms(question: str) -> list[str]:
    terms: list[str] = []
    for raw_term in _TOKEN_RE.findall(question.lower()):
        term = raw_term.strip("_")
        if len(term) < 2 or term in _STOPWORDS:
            continue
        if term not in terms:
            terms.append(term)
    return terms


def _score_candidate(
    *,
    raw: RawItemRecord,
    analysis: AIAnalysisRecord,
    terms: Sequence[str],
    exact_query: str,
) -> int:
    title = _normalize(raw.title or "")
    summary = _normalize(f"{analysis.summary_english} {analysis.summary_original}")
    evidence = _normalize(" ".join(_list_text(analysis.evidence_quotes_original) + _list_text(analysis.evidence_quotes_english)))
    labels = _normalize(
        " ".join(
            [
                raw.source_name,
                analysis.stance_toward_government,
                analysis.target,
                analysis.department,
                analysis.district,
                analysis.topic,
                analysis.issue_category,
                " ".join(_list_text(analysis.positive_points)),
                " ".join(_list_text(analysis.negative_points)),
                analysis.party_action,
                analysis.people_impact,
                analysis.root_cause,
                analysis.recommended_step,
            ]
        )
    )
    body = _normalize(raw.clean_text_original or "")
    combined = " ".join([title, summary, evidence, labels, body])

    score = 0
    exact = _normalize(exact_query)
    if exact and len(exact) >= 4 and exact in combined:
        score += 12

    for term in terms:
        if term in title:
            score += 7
        if term in evidence:
            score += 6
        if term in summary:
            score += 5
        if term in labels:
            score += 4
        if term in body:
            score += 1

    if all(term in combined for term in terms):
        score += min(len(terms), 5)
    return score


def _to_evidence(
    *,
    raw: RawItemRecord,
    analysis: AIAnalysisRecord,
    terms: Sequence[str],
) -> ChatEvidence:
    summary = _first_text(analysis.summary_english, analysis.summary_original)
    return ChatEvidence(
        raw_item_id=raw.id,
        analysis_id=analysis.id,
        title=raw.title or "Untitled item",
        source_name=raw.source_name,
        source_url=raw.source_url,
        published_at=raw.published_at,
        language=raw.language,
        stance=analysis.stance_toward_government,
        relevance=analysis.government_relevance,
        summary=summary,
        snippet=_best_snippet(raw=raw, analysis=analysis, terms=terms, summary=summary),
        department=analysis.department,
        district=analysis.district,
        topic=analysis.topic,
        confidence=analysis.confidence,
        needs_human_review=analysis.needs_human_review,
    )


def _best_snippet(
    *,
    raw: RawItemRecord,
    analysis: AIAnalysisRecord,
    terms: Sequence[str],
    summary: str,
) -> str:
    quotes = _list_text(analysis.evidence_quotes_original) + _list_text(analysis.evidence_quotes_english)
    for quote in quotes:
        normalized_quote = _normalize(quote)
        if any(term in normalized_quote for term in terms):
            return _shorten(quote, 320)
    if quotes:
        return _shorten(quotes[0], 320)
    if summary:
        return _shorten(summary, 320)
    return _text_window(raw.clean_text_original or raw.raw_text_original or "", terms)


def _text_window(text: str, terms: Sequence[str], max_chars: int = 320) -> str:
    clean = " ".join((text or "").split())
    if not clean:
        return ""
    lower = _normalize(clean)
    index = -1
    for term in terms:
        index = lower.find(term)
        if index >= 0:
            break
    if index < 0:
        return _shorten(clean, max_chars)

    start = max(0, index - max_chars // 3)
    end = min(len(clean), start + max_chars)
    snippet = clean[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(clean):
        snippet += "..."
    return snippet


# Entity types worth pulling a dossier for when named in a question. Sources are
# excluded — "what does The Hindu say" is a retrieval query, not a dossier ask.
_DOSSIER_ENTITY_TYPES = ("person", "party", "office", "district", "department", "scheme")
_DOSSIER_LEAN = {
    "positive": "net favourable",
    "negative": "net critical",
    "mixed": "mixed",
    "neutral": "mostly neutral",
}


def resolve_question_entities(session: Session, question: str, *, limit: int = 3) -> list[str]:
    """Find canonical entities named in the question, longest alias first.

    Cheap and exact: scans the normalized question for every known alias so
    "how is Stalin being covered" resolves to the M.K. Stalin dossier. Short
    ASCII aliases (CM, MP) require a word-boundary hit to avoid matching inside
    other words; Tamil aliases match as substrings.
    """
    norm_q = normalize_surface(question)
    if not norm_q:
        return []
    rows = session.execute(
        select(EntityAliasRecord.normalized, EntityRecord.slug, EntityRecord.entity_type)
        .join(EntityRecord, EntityRecord.id == EntityAliasRecord.entity_id)
        .where(EntityRecord.status == "active")
        .where(EntityRecord.entity_type.in_(_DOSSIER_ENTITY_TYPES))
    ).all()
    # Longest aliases first so "m k stalin" wins over "stalin" (same entity) and
    # multi-word place names beat incidental short tokens.
    rows = sorted(rows, key=lambda r: len(r[0]), reverse=True)
    found: list[str] = []
    for normalized, slug, _etype in rows:
        if not normalized or slug in found:
            continue
        if normalized.isascii() and len(normalized) <= 3:
            if re.search(rf"\b{re.escape(normalized)}\b", norm_q) is None:
                continue
        elif normalized not in norm_q:
            continue
        found.append(slug)
        if len(found) >= limit:
            break
    return found


def _dossier_line(dossier: dict[str, Any]) -> str:
    bits = [dossier["name"]]
    descr = " / ".join(
        x for x in [dossier.get("role", "").replace("_", " "), dossier.get("party", "")] if x
    )
    if descr:
        bits.append(f"({descr})")
    fav = dossier.get("favorability")
    fav_text = f"favourability {fav}/100" if fav is not None else "favourability n/a"
    parts = [
        f"{dossier['mention_count']} mentions ({dossier['mention_count_30d']} in last 30d)",
        fav_text,
        _DOSSIER_LEAN.get(dossier.get("dominant", "neutral"), "mixed"),
    ]
    co = ", ".join(c["name"] for c in dossier.get("co_mentions", [])[:4])
    if co:
        parts.append(f"appears with {co}")
    cats = ", ".join(dossier.get("top_categories", [])[:3])
    if cats:
        parts.append(f"issues: {cats}")
    dists = ", ".join(d["district"] for d in dossier.get("top_districts", [])[:3])
    if dists:
        parts.append(f"districts: {dists}")
    return f"- {' '.join(bits)} — " + "; ".join(parts) + "."


def build_dossier_context(session: Session, question: str, *, limit: int = 3) -> str:
    """A compact aggregate brief for entities named in the question, drawn from
    the knowledge graph — the over-time context plain keyword retrieval cannot
    give. Empty string when no known entity is named."""
    slugs = resolve_question_entities(session, question, limit=limit)
    lines: list[str] = []
    for slug in slugs:
        dossier = entity_dossier(session, slug, evidence_limit=0)
        if dossier and dossier["mention_count"] > 0:
            lines.append(_dossier_line(dossier))
    return "\n".join(lines)


def _build_chat_prompt(
    *,
    question: str,
    evidence: Sequence[ChatEvidence],
    history: Sequence[ChatTurn] | None = None,
    dossier_context: str = "",
) -> str:
    evidence_blocks: list[str] = []
    include_tamil_note = _contains_tamil(question) or any(
        _contains_tamil(item.snippet) or item.language.lower().startswith("ta")
        for item in evidence
    )
    for index, item in enumerate(evidence[:_MAX_LIMIT], start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"[{index}] Source: {item.source_name}",
                    f"Title: {item.title}",
                    f"URL: {item.source_url}",
                    f"Stance: {item.stance}",
                    f"Department: {item.department}",
                    f"District: {item.district}",
                    f"Summary: {item.summary}",
                    f"Evidence quote: {item.snippet}",
                    f"Needs human review: {item.needs_human_review}",
                ]
            )
        )

    tamil_instruction = (
        "If Tamil evidence is present, add a short Tamil summary after the English answer."
        if include_tamil_note
        else "Answer in clear English."
    )

    history_block = ""
    recent = [turn for turn in (history or []) if turn.content.strip()][-6:]
    if recent:
        turns = "\n".join(
            f"{'User' if turn.role == 'user' else 'Assistant'}: {turn.content.strip()}"
            for turn in recent
        )
        history_block = (
            "Conversation so far (oldest to newest). Use it only to resolve "
            "references like \"it\" or \"the second one\"; always ground the new "
            f"answer in the evidence below:\n{turns}\n\n"
        )

    dossier_block = ""
    if dossier_context.strip():
        dossier_block = (
            "Knowledge-graph context (aggregate counts across ALL stored coverage, "
            "computed — use for over-time/standing claims like trends and "
            "favourability; the numbered evidence below is your source for "
            "specific facts and quotes):\n"
            f"{dossier_context}\n\n"
        )

    return f"""
You are a confidential public media briefing assistant for Tamilaga Vettri Kazhagam.

{history_block}Question:
{question}

{dossier_block}Stored newspaper evidence:
{chr(10).join(evidence_blocks)}

Rules:
- Use only the stored evidence and knowledge-graph context above. Do not add outside facts.
- Cite evidence numbers like [1] whenever making a claim.
- Explain positives, negatives, people issues, and TVK party actions only when supported by evidence.
- Sensitive allegations must be framed as allegations and marked for human review.
- If the evidence does not answer the question, say there is not enough evidence.
- Keep the answer suitable for senior officials: direct, factual, and non-technical.
- {tamil_instruction}
""".strip()


def build_retrieval_query(question: str, history: Sequence[ChatTurn] | None = None) -> str:
    """Enrich a short follow-up ("what about water?") with the previous user
    question so evidence retrieval still finds the relevant articles. Keyword
    scoring just sees extra terms, so this only ever helps recall."""
    if history:
        for turn in reversed(history):
            if turn.role == "user" and turn.content.strip():
                return f"{turn.content.strip()} {question}".strip()
    return question


def _ollama_response_text(response: Any) -> str:
    if isinstance(response, dict):
        return str(response.get("response", "")).strip()
    return str(getattr(response, "response", "")).strip()


def _ollama_chunk_text(chunk: Any) -> str:
    # Streaming variant: must NOT strip — fragments carry inter-token spaces.
    if isinstance(chunk, dict):
        return str(chunk.get("response", ""))
    return str(getattr(chunk, "response", ""))


def _list_text(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _first_text(*values: str | None) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _normalize(value: str) -> str:
    return " ".join((value or "").lower().split())


def _shorten(value: str, max_chars: int) -> str:
    clean = " ".join((value or "").split())
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3].rstrip() + "..."


def _contains_tamil(value: str) -> bool:
    return any("\u0B80" <= char <= "\u0BFF" for char in value or "")
