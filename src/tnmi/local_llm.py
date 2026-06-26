"""LLM analyser running through Ollama — local CPU or cloud GPU.

When OpenAI is unavailable (quota, contract-side restrictions, or simply
"no tokens please") we want a real LLM that:

  * Speaks Tamil and English natively.
  * Generates the full briefing fields the chief expects (summary_original,
    summary_english, party_action, people_impact, root_cause,
    recommended_step) — not just stance labels.
  * Costs zero OpenAI tokens.

Ollama wraps that LLM deployment with a single HTTP API at
http://localhost:11434, regardless of where the model actually runs:

  Local mode (default)
    Model weights live on the operator's machine; inference uses the local
    CPU. Best confidentiality posture (no article text leaves the network)
    but slow — Gemma 2 2B takes 30-60 s per article on CPU.

  Cloud mode (after `ollama signin`)
    Same daemon, same API — but model weights and inference live on Ollama's
    hosted GPUs. The local daemon proxies requests transparently. Much
    faster (typically 2-5 s per article) and unlocks bigger models that
    don't fit on CPU (Gemma 3 27B, Qwen 2.5 72B, GPT-OSS 120B). Article
    text leaves the network, so only use this if newspaper coverage is
    public enough that the OpenAI confidentiality posture would also apply.

Selecting the model:

  Set OLLAMA_MODEL in the environment. Examples:
    OLLAMA_MODEL=gemma2:2b              (default — local CPU)
    OLLAMA_MODEL=gemma3:27b-cloud        (Ollama Cloud, big Gemma)
    OLLAMA_MODEL=qwen2.5:72b-cloud       (Ollama Cloud, big Qwen)

Pipeline:
  1. Apply the listing-page + TN-relevance gates from ai.py — we never
     spend an LLM call on an RSS shell or out-of-scope article.
  2. Build the chief-briefing prompt with an explicit JSON schema, so
     small + large models reliably produce records that pass Pydantic
     validation.
  3. POST to /api/generate with format="json", parse, validate against
     AIAnalysis. On malformed JSON or daemon error we raise — the cascade
     falls through to LocalTamilAnalyzer.

First-time install:

    winget install Ollama.Ollama
    pip install ollama

Then either pull a local model or sign in for cloud:

    ollama pull gemma2:2b               # local mode
    ollama signin                        # cloud mode (opens browser)

The Ollama daemon auto-starts on boot on Windows.
"""

from __future__ import annotations

import json
import logging
import os
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


# Default Ollama endpoint + model. Both are overridable via env vars so the
# operator can switch between local Gemma 2B, Gemma 3 / Qwen 2.5 cloud
# (via Ollama signin), or any other tag without touching code.
#
#   OLLAMA_HOST   — defaults to local daemon at http://localhost:11434.
#                   The same local daemon proxies cloud requests once the
#                   operator runs `ollama signin`, so this rarely changes.
#   OLLAMA_MODEL  — e.g. "gemma2:2b" (local CPU), "gemma3:27b-cloud",
#                   "qwen2.5:72b-cloud", "gpt-oss:120b-cloud". The "-cloud"
#                   suffix tells Ollama to run on its hosted GPUs instead
#                   of the local CPU.
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")

# We want short, scannable briefing lines. Gemma 2 generates a couple of
# hundred tokens in under a few seconds on CPU. The briefing schema has ~30
# fields including several free-text lines and quote lists, so 768 was clipping
# the JSON mid-string (an "unterminated string" parse error). 1536 gives the
# model room to close the object.
_MAX_OUTPUT_TOKENS = 1536

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
  "tvk_relevance": "high" | "medium" | "low" | "none",
  "tvk_portrayal": "positive" | "negative" | "neutral" | "mixed",
  "sentiment": "positive" | "negative" | "neutral",
  "target": "string (who the article is about — TVK leadership, Tamil Nadu Government, a minister, a community, etc.)",
  "political_actors": ["array of explicitly mentioned actors: TVK, Vijay, MLAs, ministers, Chief Minister, departments, parties, officials"],
  "department": "string (the relevant department or 'general')",
  "district": "string (a TN district name or 'unspecified')",
  "scheme": "string or null (named government scheme only if explicitly mentioned)",
  "topic": "string (3-6 word phrase summarising the story)",
  "issue_category": "string (welfare | concern | listing | out-of-scope | general)",
  "people_issue": "boolean (true when ordinary people face or benefit from a public matter)",
  "public_issue": "string (2-6 word issue label; empty string if not a public issue)",
  "severity": "low" | "medium" | "high" | "critical",
  "summary_original": "string (one factual sentence, original language, <= 22 words)",
  "summary_english": "string (same sentence in English, <= 22 words; empty string if article is not in English and translation would distort)",
  "party_action": "string (what TVK members or leadership did, well or badly; empty string if not about TVK)",
  "people_impact": "string (what ordinary people are facing or benefiting from; empty string if not applicable)",
  "root_cause": "string (the underlying WHY — concrete driver; empty string only if no causal signal)",
  "recommended_step": "string (one concrete action; empty string if no action warranted)",
  "action_owner": "string (who should own the follow-up)",
  "action_type": "string (monitor | amplify | field_verification | public_statement | internal_review | legal_review | escalation | policy_research)",
  "action_priority": "low" | "medium" | "high" | "critical",
  "risk_if_ignored": "string (what worsens if the leadership does nothing; empty unless negative or people-issue)",
  "talking_points": ["array of 2-3 short defensible public lines; empty unless negative or people-issue"],
  "verification_checklist": ["array of 2-3 concrete facts to confirm before acting; empty unless negative or people-issue"],
  "draft_statement_original": "string (2-3 careful sentences, original language, conceding nothing unverified; empty unless negative or people-issue)",
  "draft_statement_english": "string (English version of the draft statement; empty unless negative or people-issue)",
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

Who counts as TVK: Vijay, the party organisation and its cadre, AND any TVK
members who hold public office — MLAs, ministers, or the Chief Minister. The
official conduct of a TVK office-holder counts as TVK activity. People from
other parties (DMK, AIADMK, BJP, NTK, PMK) are NOT TVK even when they hold
office; their conduct never sets tvk_portrayal to positive or negative.

Score tvk_portrayal and stance_toward_government as TWO independent axes — when
TVK runs the government they usually agree, but a bureaucratic lapse with no TVK
person named is government-negative while tvk_portrayal stays neutral.

Required briefing fields:
  1. tvk_portrayal — positive | negative | mixed | neutral, the HEADLINE label:
     how the news reflects on TVK, Vijay, the party, or a TVK office-holder.
     positive = good news (scheme delivered, praise, decisive response);
     negative = failure, broken promise, scandal, or a governance lapse on a
     TVK office-holder's watch. neutral if no TVK person is portrayed.
  2. tvk_relevance — high when TVK, Vijay, or a TVK member is directly
     involved; medium when it is a people issue or political opening TVK may
     need to act on; low for background TN context; none for out-of-scope.
  3. stance_toward_government — positive | negative | mixed | neutral toward the
     Tamil Nadu administration in office, judged as an institution.
  4. people_issue — true for welfare, civic services, livelihood, safety,
     health, education, law/order, corruption, farmers, youth, women, etc.
  5. political_actors — explicitly mentioned TVK, Vijay, party members, MLAs,
     ministers, Chief Minister, departments, parties, or officials. No invention.
  6. party_action — what TVK members / leadership did, well or badly,
     as reported. Empty string if the article is not about TVK.
  7. people_impact — what ordinary people in the relevant area are facing or
     benefiting from. Empty string if not applicable.
  8. public_issue — 2-6 word issue label. Empty if there is no people issue.
  9. root_cause — the underlying WHY: the policy gap, decision, event, or
     structural condition driving what the article reports. Concrete.
  10. recommended_step — one concrete action the leadership office could
     consider (visit, statement, relief, internal review of a member,
     public response, fact-finding, etc.). Empty string if no action is
     clearly warranted by the evidence.
  11. action_owner, action_type, action_priority — route the follow-up to a
      concrete owner using only evidence-backed action.
  12. ACTION PLAYBOOK (negative or people-issue articles only) — risk_if_ignored,
      talking_points (2-3 lines), verification_checklist (2-3 facts to confirm),
      and draft_statement_original / draft_statement_english (2-3 careful
      sentences conceding nothing unverified). Leave ALL of these empty for
      positive, neutral, or out-of-scope rows.

Tone rules:
- One sentence per briefing field. Each field ≤ 22 words.
- Plain language. Avoid corporate jargon, hype, rhetoric.
- Do not invent facts. If the article does not warrant a field, leave it "".
- Do not write generic next steps such as "address the concern" or "take
  appropriate action". Name a concrete owner, verification target, and action
  route, or set action_type = "monitor".
- For public harm or civic issues, recommended_step must name what to verify
  locally before any public statement.
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
            is_cloud = self._model.endswith("-cloud") or self._model.endswith(":cloud")
            hint = (
                "Sign in to Ollama Cloud with: ollama signin"
                if is_cloud
                else f"Pull it with: ollama pull {self._model}"
            )
            raise GemmaAnalyzerUnavailable(
                f"model {self._model!r} not found in Ollama. {hint}. "
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
        if not _looks_like_tn_content(title, body, item.source_url):
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
    payload.setdefault("tvk_relevance", payload.get("government_relevance") or "low")
    payload.setdefault("tvk_portrayal", "neutral")
    payload.setdefault("sentiment", "neutral")
    payload.setdefault("target", "Public matter")
    payload.setdefault("political_actors", [])
    payload.setdefault("department", "general")
    payload.setdefault("district", "unspecified")
    payload.setdefault("scheme", None)
    payload.setdefault("topic", _truncate(item.title or "news item", 80))
    payload.setdefault("issue_category", "general")
    payload.setdefault("people_issue", False)
    payload.setdefault("public_issue", "")
    payload.setdefault("severity", "low")
    payload.setdefault("summary_original", _first_sentence(item.clean_text_original or "") or "")
    payload.setdefault("summary_english", "")
    payload.setdefault("party_action", "")
    payload.setdefault("people_impact", "")
    payload.setdefault("root_cause", "")
    payload.setdefault("recommended_step", "")
    payload.setdefault("action_owner", "")
    payload.setdefault("action_type", "monitor")
    payload.setdefault("action_priority", payload.get("severity") or "low")
    payload.setdefault("risk_if_ignored", "")
    payload.setdefault("talking_points", [])
    payload.setdefault("verification_checklist", [])
    payload.setdefault("draft_statement_original", "")
    payload.setdefault("draft_statement_english", "")
    payload.setdefault("positive_points", [])
    payload.setdefault("negative_points", [])
    payload.setdefault("evidence_quotes_original", [])
    payload.setdefault("evidence_quotes_english", [])
    payload.setdefault("confidence", 0.7)
    payload.setdefault("needs_human_review", False)

    # Some Gemma outputs nullify optional lists, or return a single string where
    # a list is expected. Coerce both to a clean list[str].
    for key in (
        "positive_points",
        "negative_points",
        "evidence_quotes_original",
        "evidence_quotes_english",
        "political_actors",
        "talking_points",
        "verification_checklist",
    ):
        value = payload.get(key)
        if value is None or isinstance(value, bool):
            payload[key] = []
        elif isinstance(value, str):
            payload[key] = [value.strip()] if value.strip() else []
        elif isinstance(value, list):
            payload[key] = [str(v).strip() for v in value if v is not None and str(v).strip()]
        else:
            payload[key] = []

    # people_issue / needs_human_review are booleans; Gemma sometimes returns a
    # string ("true"/"yes") or omits them. Coerce to a real bool.
    for key in ("people_issue", "needs_human_review"):
        value = payload.get(key)
        if isinstance(value, str):
            payload[key] = value.strip().lower() in {"true", "yes", "1", "y"}
        elif not isinstance(value, bool):
            payload[key] = bool(value)

    # Gemma sometimes returns null or the literal string "None"/"N/A" for
    # optional briefing-line fields. Coerce both to empty string so the
    # dashboard hides the line cleanly instead of rendering "None".
    _OPTIONAL_STR_FIELDS = (
        "summary_english",
        "party_action",
        "people_impact",
        "root_cause",
        "recommended_step",
        "public_issue",
        "action_owner",
        "action_type",
        "target",
        "department",
        "district",
        "topic",
        "issue_category",
        "summary_original",
        "risk_if_ignored",
        "draft_statement_original",
        "draft_statement_english",
    )
    _NULL_STRING_MARKERS = {"none", "n/a", "null", "не применимо"}
    for key in _OPTIONAL_STR_FIELDS:
        value = payload.get(key)
        # Gemma occasionally types a free-text field as a bool/number/list/object
        # (e.g. people_impact=false). None of those are valid content, so flatten
        # them: bool/list/dict → "", scalar → its string form.
        if value is None or isinstance(value, (bool, list, dict)):
            payload[key] = ""
        elif not isinstance(value, str):
            payload[key] = str(value)
        elif value.strip().lower() in _NULL_STRING_MARKERS:
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

    # Coerce sloppy enum values (a 2B model emits "serious", "mixed signals",
    # "very high"…) to the nearest valid member — otherwise ONE bad word makes
    # Pydantic reject the whole analysis and the article is lost.
    _coerce_enums(payload)


_SEVERITY_VALID = {"low", "medium", "high", "critical"}
_SEVERITY_SYNONYMS = {
    "serious": "high", "severe": "critical", "grave": "critical", "major": "high",
    "significant": "high", "moderate": "medium", "minor": "low", "trivial": "low",
    "urgent": "critical", "emergency": "critical", "very high": "critical",
    "very low": "low", "none": "low", "normal": "low",
}
_RELEVANCE_VALID = {"high", "medium", "low", "none"}
_RELEVANCE_SYNONYMS = {
    "very high": "high", "critical": "high", "moderate": "medium", "minimal": "low",
    "very low": "low", "not relevant": "none", "na": "none", "n/a": "none", "irrelevant": "none",
}
_STANCE_VALID = {"positive", "negative", "neutral", "mixed"}
_STANCE_SYNONYMS = {
    "mixed signals": "mixed", "both": "mixed", "supportive": "positive", "favourable": "positive",
    "favorable": "positive", "critical": "negative", "against": "negative", "unfavourable": "negative",
    "unfavorable": "negative",
}
_SENTIMENT_VALID = {"positive", "negative", "neutral"}


def _coerce_one(value: object, valid: set[str], synonyms: dict[str, str], default: str) -> str:
    text = str(value or "").strip().lower()
    if text in valid:
        return text
    return synonyms.get(text, default)


def _coerce_enums(payload: dict) -> None:
    payload["severity"] = _coerce_one(payload.get("severity"), _SEVERITY_VALID, _SEVERITY_SYNONYMS, "medium")
    payload["action_priority"] = _coerce_one(
        payload.get("action_priority"), _SEVERITY_VALID, _SEVERITY_SYNONYMS, payload["severity"]
    )
    for field in ("government_relevance", "tvk_relevance"):
        if payload.get(field) is not None:
            payload[field] = _coerce_one(payload.get(field), _RELEVANCE_VALID, _RELEVANCE_SYNONYMS, "low")
    for field in ("stance_toward_government", "tvk_portrayal"):
        if payload.get(field) is not None:
            payload[field] = _coerce_one(payload.get(field), _STANCE_VALID, _STANCE_SYNONYMS, "neutral")
    payload["sentiment"] = _coerce_one(payload.get("sentiment"), _SENTIMENT_VALID, _STANCE_SYNONYMS, "neutral")
