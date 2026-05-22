from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol

from openai import OpenAI

from tnmi.contracts import (
    AIAnalysis,
    GovernmentRelevance,
    NormalizedItem,
    Sentiment,
    Severity,
    Stance,
)


PROMPT_VERSION = "newspaper-stance-v1"


class AIAnalysisError(RuntimeError):
    """Raised when an AI provider cannot return a usable analysis."""


class AIAnalyzer(Protocol):
    model_name: str

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        ...


class MockAIAnalyzer:
    model_name = "mock"

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        text = f"{item.title or ''}\n{item.clean_text_original}".lower()
        relevance = (
            GovernmentRelevance.HIGH
            if "அரசு" in text or "government" in text
            else GovernmentRelevance.NONE
        )
        stance = Stance.POSITIVE if "திட்ட" in text or "scheme" in text else Stance.NEUTRAL
        return AIAnalysis(
            government_relevance=relevance,
            stance_toward_government=stance,
            sentiment=Sentiment.POSITIVE if stance == Stance.POSITIVE else Sentiment.NEUTRAL,
            target="Tamil Nadu Government" if relevance != GovernmentRelevance.NONE else "none",
            department="unknown",
            district="unknown",
            scheme=None,
            topic=item.title or "news item",
            issue_category="welfare" if stance == Stance.POSITIVE else "unknown",
            severity=Severity.LOW,
            summary_original=(item.clean_text_original[:180] or item.title or "").strip(),
            summary_english="Mock analysis summary.",
            positive_points=["Mentions a government scheme."] if stance == Stance.POSITIVE else [],
            negative_points=[],
            evidence_quotes_original=[item.clean_text_original[:120]],
            evidence_quotes_english=["Mock evidence translation."],
            confidence=0.75,
            needs_human_review=False,
        )


class OpenAIAnalyzer:
    def __init__(self, api_key: str, model_name: str = "gpt-5.4-mini") -> None:
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        prompt = build_classification_prompt(item)
        response = self.client.responses.parse(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": "Return schema-bound structured analysis for Tamil Nadu media.",
                },
                {"role": "user", "content": prompt},
            ],
            text_format=AIAnalysis,
        )
        parsed = response.output_parsed
        if parsed is None:
            raise AIAnalysisError(_missing_output_message(response))
        if isinstance(parsed, dict):
            return AIAnalysis.model_validate(parsed)
        return parsed


def build_classification_prompt(item: NormalizedItem) -> str:
    return f"""
You are analyzing public media content about the Tamil Nadu Government.

Analyze Tamil, English, Tanglish, and mixed-script content. Preserve original meaning.
Do not over-translate slogans, sarcasm, allegations, or local political phrases.
Classify stance toward the Tamil Nadu Government only from evidence in the text.
If the content is not about the Tamil Nadu Government, government_relevance must be "none".
If a claim is an allegation, needs_human_review must be true.
Populate the scheme name only when explicitly mentioned; otherwise null.

Return JSON only with this schema:
{{
  "government_relevance": "high|medium|low|none",
  "stance_toward_government": "positive|negative|neutral|mixed",
  "sentiment": "positive|negative|neutral",
  "target": "...",
  "department": "...",
  "district": "...",
  "scheme": "string|null",
  "topic": "...",
  "issue_category": "...",
  "severity": "low|medium|high|critical",
  "summary_original": "...",
  "summary_english": "...",
  "positive_points": [],
  "negative_points": [],
  "evidence_quotes_original": [],
  "evidence_quotes_english": [],
  "confidence": 0.0,
  "needs_human_review": true
}}

Source: {item.source_name}
URL: {item.source_url}
Language: {item.language}
Title: {item.title or ""}
Text:
{item.clean_text_original}
""".strip()


def _missing_output_message(response: Any) -> str:
    details: list[str] = []
    incomplete_reason = _safe_path(response, "incomplete_details", "reason")
    if incomplete_reason:
        details.append(f"incomplete_reason={incomplete_reason}")

    refusals = _collect_refusals(response)
    if refusals:
        details.append(f"refusal={'; '.join(refusals)}")

    message = "OpenAI structured analysis response did not include parsed output"
    if details:
        message = f"{message}: {', '.join(details)}"
    return message


def _collect_refusals(response: Any) -> list[str]:
    refusals: list[str] = []
    _append_text(refusals, _safe_getattr(response, "refusal"))

    for output_item in _safe_iterable(_safe_getattr(response, "output")):
        _append_text(refusals, _safe_getattr(output_item, "refusal"))
        for content_item in _safe_iterable(_safe_getattr(output_item, "content")):
            _append_text(refusals, _safe_getattr(content_item, "refusal"))

    return refusals


def _safe_path(obj: Any, *path: str) -> Any:
    current = obj
    for name in path:
        current = _safe_getattr(current, name)
        if current is None:
            return None
    return current


def _safe_getattr(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name, None)
    except Exception:
        return None


def _safe_iterable(value: Any) -> Iterable[Any]:
    if value is None or isinstance(value, str):
        return ()
    try:
        return iter(value)
    except TypeError:
        return ()


def _append_text(values: list[str], value: Any) -> None:
    if isinstance(value, str) and value:
        values.append(value)
