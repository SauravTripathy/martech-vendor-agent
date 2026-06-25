"""
Model API layer — cross-provider.

* gather_evidence_for_category -> Anthropic + server-side web_search -> Evidence[]
* score_all -> Anthropic scores the vendor from the gathered evidence
* judge_review -> OpenAI re-scores the SAME evidence and audits the primary scores

All calls force JSON-only output and parse defensively. Usage dicts are tagged
with "provider" so RunMetrics can break tokens down per provider.
"""

from __future__ import annotations

import json
import re
from typing import Any

import config

try:
    from anthropic import Anthropic

    _anthropic = Anthropic()  # reads ANTHROPIC_API_KEY from env
except Exception:  # pragma: no cover - lets the module import without a key
    _anthropic = None

try:
    from openai import OpenAI

    _openai = OpenAI()  # reads OPENAI_API_KEY from env
except Exception:  # pragma: no cover
    _openai = None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _text_and_usage(message: Any) -> tuple[str, dict]:
    """Join text blocks; pull usage including server-tool web-search counts."""
    parts: list[str] = []
    for block in getattr(message, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)

    usage_obj = getattr(message, "usage", None)
    usage = {
        "provider": "anthropic",
        "input_tokens": getattr(usage_obj, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage_obj, "output_tokens", 0) or 0,
        "web_search_requests": 0,
    }
    server_tool = getattr(usage_obj, "server_tool_use", None)
    if server_tool is not None:
        usage["web_search_requests"] = getattr(server_tool, "web_search_requests", 0) or 0

    return "\n".join(parts), usage


def _extract_json(text: str) -> Any:
    """Strip code fences and return the first balanced JSON object/array."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()

    try:
        return json.loads(t)
    except Exception:
        pass

    # balanced-scan fallback for the first { ... } or [ ... ]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = t.find(open_c)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(t)):
            if t[i] == open_c:
                depth += 1
            elif t[i] == close_c:
                depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])

    raise ValueError(f"Could not parse JSON from model output:\n{text[:500]}")


def _call(
    model: str,
    system: str,
    user: str,
    temperature: float,
    web_search: bool = False,
) -> tuple[Any, dict]:
    """Anthropic call: primary evaluation provider."""
    if _anthropic is None:
        raise RuntimeError("Anthropic client not initialised. Set ANTHROPIC_API_KEY in the env.")

    kwargs: dict[str, Any] = dict(
        model=model,
        max_tokens=4096,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if web_search:
        kwargs["tools"] = [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": config.WEB_SEARCH_MAX_USES,
            }
        ]

    message = _anthropic.messages.create(**kwargs)
    text, usage = _text_and_usage(message)
    return _extract_json(text), usage


def _call_openai(model: str, system: str, user: str, temperature: float) -> tuple[Any, dict]:
    """OpenAI call: judge provider. Forces a JSON object response."""
    if _openai is None:
        raise RuntimeError("OpenAI client not initialised. Set OPENAI_API_KEY in the env.")

    resp = _openai.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content or ""
    u = getattr(resp, "usage", None)
    usage = {
        "provider": "openai",
        "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        "web_search_requests": 0,
    }
    return _extract_json(text), usage


# --------------------------------------------------------------------------- #
# Node 1: evidence gathering
# --------------------------------------------------------------------------- #

_EVIDENCE_SYS = """You are a procurement research analyst gathering PUBLIC evidence on a MarTech vendor for a regulated financial-services buyer.
Use web search to find current, specific, verifiable evidence.

For every item, classify the evidence tier:
Tier 1 = vendor-owned marketing/product page (self-asserted, unverified)
Tier 2 = third-party: analyst, press, community, non-vendor docs
Tier 3 = named customer reference / case study / review platform with named users
Tier 4 = verifiable artifact: cert registry, trust portal, security attestation, regulatory filing

Be conservative on tiers: a claim on the vendor's own site is Tier 1 even if it mentions a cert.
Only Tier 4 verifies a certification. If you cannot find evidence for a criterion, omit it.
Never invent sources, URLs, certifications, or customers.

Respond with JSON ONLY, no prose:
{"evidence":[{"criterion_id":"...","claim":"<=30 words","tier":1-4,"source_title":"...","source_url":"https://...","snippet":"<=40 words"}]}
"""


def gather_evidence_for_category(
    vendor: str,
    use_case: str,
    category: str,
    criteria: list,
    documents: str = "",
) -> tuple[list[dict], dict]:
    crit_lines = "\n".join(f" - {c.id}: {c.description}" for c in criteria)
    doc_block = ""
    if documents:
        doc_block = (
            "\n\nUSER-PROVIDED DOCUMENTS — extract any evidence relevant to the criteria above "
            "and tier it honestly. Cite the document name as the source_title:\n"
            + documents
        )

    user = (
        f"Vendor: {vendor}\nBuyer use-case / context: {use_case}\n\n"
        f"Gather evidence for these criteria in category '{category}':\n{crit_lines}\n"
        f"{doc_block}\n\n"
        "Return the JSON object described in the system prompt."
    )

    payload, usage = _call(
        config.PRIMARY_MODEL,
        _EVIDENCE_SYS,
        user,
        config.PRIMARY_TEMPERATURE,
        web_search=True,
    )
    items = payload.get("evidence", []) if isinstance(payload, dict) else []
    return items, usage


# --------------------------------------------------------------------------- #
# Node 2: primary scoring
# --------------------------------------------------------------------------- #

def _scoring_system() -> str:
    rubric = "\n".join(
        f" {k if k is not None else 'blank'} = {v}"
        for k, v in config.SCORING_RUBRIC.items()
    )
    caps = ", ".join(f"tier {t} -> max {cap}" for t, cap in config.EVIDENCE_TIER_CAP.items())

    return f"""You score a MarTech vendor against fixed criteria using ONLY the evidence provided to you.
Do not use outside knowledge. Do not invent evidence.

Scoring rubric (1-5):
{rubric}

EVIDENCE-TIER CAP — a score is limited by the strength of its best supporting evidence:
{caps}
If the only support is a Tier-1 vendor self-asserted claim, the score cannot exceed 3.
Set "capped": true whenever the tier cap held the score down.
If there is NO usable evidence for a criterion, return score=null and is_gap=true.
NEVER score 0 — null means not evaluable and is left blank on the scorecard.

Respond with JSON ONLY:
{{"scores":[{{"criterion_id":"...","score":1-5 or null,"evidence_tier":1-4 or null,"confidence":0.0-1.0,"capped":true/false,"is_gap":true/false,"rationale":"<=40 words"}}]}}
"""


def _evidence_block(evidence: list) -> str:
    lines = []
    for e in evidence:
        lines.append(
            f"[{e.criterion_id} | tier {e.tier}] {e.claim} "
            f"(src: {e.source_title} {e.source_url})"
        )
    return "\n".join(lines) if lines else "(no evidence gathered)"


def _criteria_block(criteria: list) -> str:
    return "\n".join(
        f" - {c.id}: {c.description} (weight {c.weight})" for c in criteria
    )


def _score_user(vendor: str, use_case: str, criteria: list, evidence: list) -> str:
    return (
        f"Vendor: {vendor}\nBuyer use-case / context: {use_case}\n\n"
        f"CRITERIA TO SCORE:\n{_criteria_block(criteria)}\n\n"
        f"EVIDENCE — the only basis for scoring:\n{_evidence_block(evidence)}\n\n"
        "Score every criterion per the system prompt."
    )


def score_all(vendor: str, use_case: str, criteria: list, evidence: list) -> tuple[dict, dict]:
    user = _score_user(vendor, use_case, criteria, evidence)
    return _call(config.PRIMARY_MODEL, _scoring_system(), user, config.PRIMARY_TEMPERATURE)


# --------------------------------------------------------------------------- #
# Node 3: smarter judge / consistency check
# --------------------------------------------------------------------------- #

def _primary_scores_block(primary_scores: list) -> str:
    lines = []
    for s in primary_scores:
        lines.append(
            f"[{s.criterion_id}] primary_score={s.score} tier={s.evidence_tier} "
            f"confidence={s.confidence} capped={s.capped} gap={s.is_gap} "
            f"rationale={s.rationale}"
        )
    return "\n".join(lines) if lines else "(no primary scores supplied)"


def _judge_system() -> str:
    rubric = "\n".join(
        f" {k if k is not None else 'blank'} = {v}"
        for k, v in config.SCORING_RUBRIC.items()
    )
    caps = ", ".join(f"tier {t} -> max {cap}" for t, cap in config.EVIDENCE_TIER_CAP.items())

    return f"""You are a skeptical procurement QA judge.
Use ONLY the supplied evidence and the primary model's scores. Do not use outside knowledge.

Your job has two parts:
1. Re-score every criterion independently using the same rubric and evidence-tier caps.
2. Audit the primary score: identify inflated scores, weak rationale, unsupported assumptions, missing evidence, or material score disagreement.

Scoring rubric:
{rubric}

Evidence-tier caps:
{caps}

Be strict:
- If evidence is vague or vendor-owned only, do not allow a high score.
- If the primary score assumes facts not present in evidence, flag unsupported_claim.
- If there is no usable evidence, score=null and flag missing_evidence.
- If primary and judge scores differ by 2+ points, flag score_disagreement and material=true.
- recommended_score should be your best defensible final score when adjustment is needed; otherwise it may equal judge_score.

Respond with JSON ONLY:
{{
  "scores":[{{"criterion_id":"...","score":1-5 or null,"evidence_tier":1-4 or null,"confidence":0.0-1.0,"capped":true/false,"is_gap":true/false,"rationale":"<=40 words"}}],
  "findings":[{{"criterion_id":"...","issue_type":"inflated_score|missing_evidence|unsupported_claim|weak_rationale|score_disagreement|none","severity":"low|medium|high","primary_score":1-5 or null,"judge_score":1-5 or null,"recommended_score":1-5 or null,"material":true/false,"explanation":"<=55 words"}}],
  "summary":"<=80 words"
}}
"""


def _judge_user(
    vendor: str,
    use_case: str,
    criteria: list,
    evidence: list,
    primary_scores: list,
) -> str:
    return (
        f"Vendor: {vendor}\nBuyer use-case / context: {use_case}\n\n"
        f"CRITERIA:\n{_criteria_block(criteria)}\n\n"
        f"EVIDENCE — the only basis for judging:\n{_evidence_block(evidence)}\n\n"
        f"PRIMARY MODEL SCORES TO AUDIT:\n{_primary_scores_block(primary_scores)}\n\n"
        "Return the JSON object described in the system prompt."
    )


def judge_review(
    vendor: str,
    use_case: str,
    criteria: list,
    evidence: list,
    primary_scores: list,
) -> tuple[dict, dict]:
    """Smarter cross-provider judge: re-scores and audits primary scoring quality."""
    user = _judge_user(vendor, use_case, criteria, evidence, primary_scores)
    return _call_openai(config.JUDGE_MODEL, _judge_system(), user, config.JUDGE_TEMPERATURE)


# Backward-compatible alias for older imports. The graph now calls judge_review.
def judge_rescore(
    vendor: str,
    use_case: str,
    criteria: list,
    evidence: list,
    primary_scores: list | None = None,
) -> tuple[dict, dict]:
    return judge_review(vendor, use_case, criteria, evidence, primary_scores or [])
