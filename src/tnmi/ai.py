from __future__ import annotations

import json
from typing import Protocol

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
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            text={"format": {"type": "json_object"}},
        )
        payload = json.loads(response.output_text)
        return AIAnalysis.model_validate(payload)


def build_classification_prompt(item: NormalizedItem) -> str:
    return f"""
You are analyzing public media content about the Tamil Nadu Government.

Analyze Tamil, English, Tanglish, and mixed-script content. Preserve original meaning.
Do not over-translate slogans, sarcasm, allegations, or local political phrases.
Classify stance toward the Tamil Nadu Government only from evidence in the text.
If the content is not about the Tamil Nadu Government, government_relevance must be "none".
If a claim is an allegation, needs_human_review must be true.

Return JSON only with this schema:
{{
  "government_relevance": "high|medium|low|none",
  "stance_toward_government": "positive|negative|neutral|mixed",
  "sentiment": "positive|negative|neutral",
  "target": "...",
  "department": "...",
  "district": "...",
  "scheme": null,
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
