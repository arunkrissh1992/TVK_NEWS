"""Local LLM analyser — Gemma 2 via Ollama.

Phase G fallback: when OpenAI is unavailable (quota, network, contract-side
restrictions) we want a real LLM running on the operator's machine, not the
keyword/embedding classifier. That LLM has to:

  * Run fully offline once the model is pulled (true confidentiality —
    no article text leaves the operator's network).
  * Speak Tamil and English natively.
  * Generate the full briefing fields the chief expects (summary_original,
    summary_english, party_action, people_impact, root_cause,
    recommended_step) — not just stance labels.
  * Cost zero tokens.

Gemma 2 2B (quantized, ~1.6 GB) hits this sweet spot. It runs on CPU in
~3-8 seconds per article, handles Tamil reasonably, and Ollama wraps the
deployment with a simple HTTP API at http://localhost:11434.

Pipeline:
  1. Apply the same listing-page + TN-relevance gates as ai.py / local_models.py
     — we never spend a Gemma call on an RSS shell or out-of-scope article.
  2. Build the same chief-briefing prompt as OpenAI, but with explicit JSON
     schema + JSON-mode instructions Gemma can reliably follow.
  3. POST to /api/generate with format="json", parse, validate against
     AIAnalysis. If the model returned malformed JSON, raise — the cascade
     falls through to LocalTamilAnalyzer.

Install:

    winget install Ollama.Ollama
    ollama pull gemma2:2b
    pip install ollama

After that everything runs offline. The Ollama daemon auto-starts on boot
on Windows.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from tnmi.ai import (
    _first_sentence,
    _looks_like_listing_page,
    _looks_like_tn_content,
    _not_relevant_analysis,
    _truncate,
)
from tnmi.contracts import AIAnalysis, NormalizedItem


logger = logging.getLogger(__name__)


# Default Ollama endpoint + model. The operator can override the model in
# Settings (e.g. swap gemma2:2b for gemma2:9b or llama3.2:3b without code).
_OLLAMA_HOST = "http://localhost:11434"
_DEFAULT_MODEL = "gemma2:2b"

# We want short, scannable briefing lines. Gemma 2 generates a couple of
# hundred tokens in under a few seconds on CPU; 768 is comfortably above
# our envelope.
_MAX_OUTPUT_TOKENS = 768

# Temperature 0.2 keeps the output stable across re-runs (deterministic-ish)
# while leaving enough flexibility for natural-sounding summaries.
_TEMPERATURE = 0.2


# JSON schema we send to Gemma. We deliberately list every AIAnalysis field
# with its allowed values, so Gemma's structured-output mode produces records
# that pass Pydantic validation on the first try.
_JSON_SCHEMA_PROMPT = """
Return JSON only. No prose before or after. The JSON must have exactly these keys:

{
  "government_relevance": "high" | "medium" | "low" | "none",
  "stance_toward_government": "positive" | "negative" | "neutral" | "mixed",
  "sentiment": "positive" | "negative" | "neutral",
  "target": "string (who the article is about — TVK leadership, Tamil Nadu Government, a minister, a community, etc.)",
  "department": "string (the relevant department or 'general')",
  "district": "string (a TN district name or 'unspecified')",
  "scheme": "string or null (named government scheme only if explicitly mentioned)",
  "topic": "string (3-6 word phrase summarising the story)",
  "issue_category": "string (welfare | concern | listing | out-of-scope | general)",
  "severity": "low" | "medium" | "high" | "critical",
  "summary_original": "string (one factual sentence, original language, <= 22 words)",
  "summary_english": "string (same sentence in English, <= 22 words; empty string if article is not in English and translation would distort)",
  "party_action": "string (what TVK members or leadership did, well or badly; empty string if not about TVK)",
  "people_impact": "string (what ordinary people are facing or benefiting from; empty string if not applicable)",
  "root_cause": "string (the underlying WHY — concrete driver; empty string only if no causal signal)",
  "recommended_step": "string (one concrete action; empty string if no action warranted)",
  "positive_points": ["array of 0-2 short positive bullet phrases"],
  "negative_points": ["array of 0-2 short negative bullet phrases"],
  "evidence_quotes_original": ["array of 1-2 short verbatim quotes from the article"],
  "evidence_quotes_english": ["array of 0-2 English translations of those quotes; empty array if not in English"],
  "confidence": "number between 0.0 and 1.0",
  "needs_human_review": "boolean (true for allegations, sensitive claims, ambiguous stances)"
}
""".strip()


def _build_gemma_prompt(item: NormalizedItem) -> str:
    """Compose the chief-briefing prompt for Gemma. Same lens as the OpenAI
    prompt in ai.py, but with stricter JSON-mode instructions Gemma follows
    more reliably than the OpenAI variant."""
    return f"""
You are preparing a 30-second daily intelligence briefing for the Tamilaga
Vettri Kazhagam (TVK) party leadership office, from a Tamil Nadu newspaper
article. The leader's decision loop is: WHAT happened → WHY → SO WHAT → WHAT NOW.

Required briefing fields:
  1. stance_toward_government — positive | negative | mixed | neutral, toward
     the Tamil Nadu Government in office (DMK-led). Not toward TVK.
  2. party_action — what TVK members / leadership did, well or badly,
     as reported. Empty string if the article is not about TVK.
  3. people_impact — what ordinary people in the relevant area are facing or
     benefiting from. Empty string if not applicable.
  4. root_cause — the underlying WHY: the policy gap, decision, event, or
     structural condition driving what the article reports. Concrete.
  5. recommended_step — one concrete action the leadership office could
     consider (visit, statement, relief, internal review of a member,
     public response, fact-finding, etc.). Empty string if no action is
     clearly warranted by the evidence.

Tone rules:
- One sentence per briefing field. Each field ≤ 22 words.
- Plain language. Avoid corporate jargon, hype, rhetoric.
- Do not invent facts. If the article does not warrant a field, leave it "".
- Preserve any Tamil quotation in evidence_quotes_original verbatim.
- For allegations or sensitive claims, set needs_human_review = true.

Tamil Nadu relevance gate (IMPORTANT):
- This briefing is for the Tamil Nadu party leadership only.
- An article is IN SCOPE only when it materially relates to Tamil Nadu —
  the state, any of its 38 districts, a TN political party or leader,
  a TN public matter, or a national story where TN involvement is central.
- An article is OUT OF SCOPE if it is purely national, Bollywood, cricket
  (without a TN angle), other states (Karnataka / Kerala / Maharashtra /
  Delhi / US / international) and does not concern TN.
- For OUT OF SCOPE articles set government_relevance = "none" and leave
  every briefing field empty. The dashboard hides relevance=none rows.

{_JSON_SCHEMA_PROMPT}

Source: {item.source_name}
URL: {item.source_url}
Language: {item.language}
Title: {item.title or ""}
Article text:
{(item.clean_text_original or "")[:4000]}
""".strip()


class GemmaAnalyzerUnavailable(RuntimeError):
    """Raised when Ollama isn't running or the model isn't pulled yet."""


class GemmaAnalyzer:
    """OpenAI-quality briefing analyser running fully on the operator's machine.

    Talks to a local Ollama daemon. On the first analyse call we ping the
    daemon — if Ollama isn't running OR the model isn't pulled, we raise
    GemmaAnalyzerUnavailable. The cascade catches that and falls through to
    LocalTamilAnalyzer, which still produces a stance-only analysis.

    Args:
        model: the Ollama model tag, e.g. "gemma2:2b", "gemma3:4b",
            "llama3.2:3b". Defaults to gemma2:2b (best CPU sweet spot).
        host: the Ollama HTTP endpoint. Override for remote Ollama instances.
        timeout: per-request timeout in seconds. Gemma 2 2B finishes most
            articles in under 10 s on a modern CPU; we give a generous 90 s
            ceiling for very long articles on slower hardware.
    """

    def __init__(
        self,
        *,
        model: str = _DEFAULT_MODEL,
        host: str = _OLLAMA_HOST,
        timeout: float = 90.0,
    ) -> None:
        # Lazy-imported so the dependency only matters when the analyser
        # is actually used. Keeps test imports + light operations cheap.
        try:
            from ollama import Client  # type: ignore[import-not-found]
        except ImportError as exc:  # noqa: TRY003
            raise GemmaAnalyzerUnavailable(
                "ollama python package is not installed. Run: pip install ollama"
            ) from exc

        self._client = Client(host=host, timeout=timeout)
        self._model = model
        self.model_name = f"ollama/{model}"
        self._daemon_checked = False

    def _ensure_daemon(self) -> None:
        """Ping Ollama once per analyser instance. If the daemon is down or
        the model is missing, raise so the cascade can fall through."""
        if self._daemon_checked:
            return
        try:
            tags = self._client.list()
        except Exception as exc:  # noqa: BLE001 — connection refused, DNS, etc.
            raise GemmaAnalyzerUnavailable(
                f"cannot reach Ollama daemon at {_OLLAMA_HOST}: {exc}"
            ) from exc

        # ollama.Client.list() returns ListResponse(models=[Model(...)])
        available = []
        for entry in getattr(tags, "models", []) or []:
            # Each model has a .model attribute like "gemma2:2b"
            name = getattr(entry, "model", None) or getattr(entry, "name", None)
            if name:
                available.append(name)
        if self._model not in available:
            raise GemmaAnalyzerUnavailable(
                f"model {self._model!r} not found in Ollama. "
                f"Pull it with: ollama pull {self._model}. "
                f"Available models: {available}"
            )
        self._daemon_checked = True

    def analyze(self, item: NormalizedItem) -> AIAnalysis:
        # Same gates as the other analysers — RSS shells and out-of-scope
        # articles never reach the LLM. Saves both Gemma latency and the
        # risk of the model hallucinating a stance on noise.
        title = (item.title or "").strip()
        body = (item.clean_text_original or "").strip()
        if _looks_like_listing_page(title, body):
            evidence = _first_sentence(body) or title
            return _not_relevant_analysis(
                title=title, evidence_quote=evidence, issue_category="listing",
            )
        if not _looks_like_tn_content(title, body):
            evidence = _first_sentence(body) or title
            return _not_relevant_analysis(
                title=title, evidence_quote=evidence, issue_category="out-of-scope",
            )

        self._ensure_daemon()

        prompt = _build_gemma_prompt(item)
        try:
            response = self._client.generate(
                model=self._model,
                prompt=prompt,
                format="json",
                options={
                    "temperature": _TEMPERATURE,
                    "num_predict": _MAX_OUTPUT_TOKENS,
                },
                stream=False,
            )
        except Exception as exc:  # noqa: BLE001
            raise GemmaAnalyzerUnavailable(
                f"Ollama generate failed for {self._model}: {exc}"
            ) from exc

        raw_text = getattr(response, "response", None) or ""
        if not raw_text.strip():
            raise GemmaAnalyzerUnavailable(
                f"Ollama returned empty response for {self._model}"
            )

        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            # Gemma sometimes wraps JSON in ```json ... ``` fences despite
            # format="json". Try one cleanup pass before giving up.
            cleaned = _strip_code_fences(raw_text)
            try:
                payload = json.loads(cleaned)
            except json.JSONDecodeError:
                logger.warning(
                    "Gemma returned malformed JSON for %s: %s",
                    item.source_url,
                    raw_text[:200],
                )
                raise GemmaAnalyzerUnavailable(
                    f"Gemma returned malformed JSON: {exc}"
                ) from exc

        # Gemma occasionally emits null/missing optional fields. Backfill
        # so Pydantic validation passes.
        _backfill_defaults(payload, item)

        try:
            return AIAnalysis.model_validate(payload)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Gemma JSON did not match AIAnalysis schema for %s: %s; payload=%s",
                item.source_url,
                exc,
                {k: v for k, v in payload.items() if k != "evidence_quotes_original"},
            )
            raise GemmaAnalyzerUnavailable(
                f"Gemma output failed AIAnalysis validation: {exc}"
            ) from exc


def _strip_code_fences(text: str) -> str:
    """Strip ```json ... ``` or ``` ... ``` wrappers that Gemma sometimes adds
    despite format="json"."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the first line (``` or ```json)
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:]
        # Drop the trailing closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


def _backfill_defaults(payload: dict[str, Any], item: NormalizedItem) -> None:
    """Fill the few fields Gemma sometimes omits, so AIAnalysis validation
    succeeds. We're permissive about content but strict about structure.

    Three coercion patterns:
      - missing-key   → set a sensible default
      - null-string   → "" (Gemma loves null for "empty")
      - null-list     → []
      - literal "None" / "N/A" in optional fields → ""
    """
    payload.setdefault("government_relevance", "low")
    payload.setdefault("stance_toward_government", "neutral")
    payload.setdefault("sentiment", "neutral")
    payload.setdefault("target", "Public matter")
    payload.setdefault("department", "general")
    payload.setdefault("district", "unspecified")
    payload.setdefault("scheme", None)
    payload.setdefault("topic", _truncate(item.title or "news item", 80))
    payload.setdefault("issue_category", "general")
    payload.setdefault("severity", "low")
    payload.setdefault("summary_original", _first_sentence(item.clean_text_original or "") or "")
    payload.setdefault("summary_english", "")
    payload.setdefault("party_action", "")
    payload.setdefault("people_impact", "")
    payload.setdefault("root_cause", "")
    payload.setdefault("recommended_step", "")
    payload.setdefault("positive_points", [])
    payload.setdefault("negative_points", [])
    payload.setdefault("evidence_quotes_original", [])
    payload.setdefault("evidence_quotes_english", [])
    payload.setdefault("confidence", 0.7)
    payload.setdefault("needs_human_review", False)

    # Some Gemma outputs nullify optional lists; coerce back to empty list.
    for key in (
        "positive_points",
        "negative_points",
        "evidence_quotes_original",
        "evidence_quotes_english",
    ):
        if payload.get(key) is None:
            payload[key] = []

    # Gemma sometimes returns null or the literal string "None"/"N/A" for
    # optional briefing-line fields. Coerce both to empty string so the
    # dashboard hides the line cleanly instead of rendering "None".
    _OPTIONAL_STR_FIELDS = (
        "summary_english",
        "party_action",
        "people_impact",
        "root_cause",
        "recommended_step",
        "target",
        "department",
        "district",
        "topic",
        "issue_category",
        "summary_original",
    )
    _NULL_STRING_MARKERS = {"none", "n/a", "null", "не применимо"}
    for key in _OPTIONAL_STR_FIELDS:
        value = payload.get(key)
        if value is None:
            payload[key] = ""
        elif isinstance(value, str) and value.strip().lower() in _NULL_STRING_MARKERS:
            payload[key] = ""

    # Coerce confidence to float if Gemma returned a string like "0.9"
    if isinstance(payload.get("confidence"), str):
        try:
            payload["confidence"] = float(payload["confidence"])
        except ValueError:
            payload["confidence"] = 0.7

    # scheme is the one optional field that *should* accept None
    scheme_val = payload.get("scheme")
    if isinstance(scheme_val, str) and scheme_val.strip().lower() in _NULL_STRING_MARKERS:
        payload["scheme"] = None
