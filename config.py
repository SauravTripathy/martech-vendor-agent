"""
Static configuration for the MarTech Vendor Evaluation Agent.

The criteria taxonomy, weights, and category structure are lifted verbatim from
MarTech_Vendor_Scorecard_Single.xlsx so the agent's output maps 1:1 onto the
scorecard. Weights sum to 1.00 (asserted at import time).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Models. These are placeholders — set the env vars to model IDs available on
# your account. The PRIMARY model must support the server-side web_search tool.
# See https://docs.claude.com for current model IDs and tool support.
# --------------------------------------------------------------------------- #
PRIMARY_MODEL = os.getenv("MARTECH_PRIMARY_MODEL", "claude-sonnet-4-5")
JUDGE_MODEL = os.getenv("MARTECH_JUDGE_MODEL", "claude-3-5-haiku-latest")

# Sampling. Scoring is held at low temperature for reproducibility; the judge
# pass is deliberately a *different model* at low temp (the "another model"
# consistency design) so divergence reflects model disagreement, not noise.
PRIMARY_TEMPERATURE = float(os.getenv("MARTECH_PRIMARY_TEMP", "0.0"))
JUDGE_TEMPERATURE = float(os.getenv("MARTECH_JUDGE_TEMP", "0.0"))

WEB_SEARCH_MAX_USES = int(os.getenv("MARTECH_SEARCH_MAX_USES", "4"))


# --------------------------------------------------------------------------- #
# Evidence tiers. The score a claim can earn is capped by the strength of the
# evidence behind it (see EVIDENCE_TIER_CAP). This is what stops a public-source
# agent from awarding a 5 off a vendor marketing page.
# --------------------------------------------------------------------------- #
EVIDENCE_TIERS = {
    1: "Vendor-owned marketing / product page (self-asserted, unverified).",
    2: "Third-party: analyst note, press, community, docs not owned by vendor.",
    3: "Named customer reference, case study, or review-platform evidence with named users.",
    4: "Verifiable artifact: cert registry / trust portal, security attestation, regulatory filing.",
}

# Max score a criterion may receive given the best evidence tier supporting it.
# Tier 1 alone caps at 3 ("adequate") — a self-asserted claim cannot earn
# "strong" or "best-in-class" without independent corroboration.
EVIDENCE_TIER_CAP = {1: 3, 2: 4, 3: 5, 4: 5}


SCORING_RUBRIC = {
    5: "Excellent — best-in-class; exceeds requirement with proven evidence (refs, certs, working POC).",
    4: "Strong — meets requirement well; minor, easily mitigated gaps.",
    3: "Adequate — meets baseline; gaps needing workarounds or roadmap commitments.",
    2: "Weak — partially meets; material gaps, significant effort/risk to close.",
    1: "Poor — does not meet requirement; likely deal-breaker for this criterion.",
    None: "N/A — not applicable or not evaluable from available evidence. Left BLANK, not 0.",
}


@dataclass(frozen=True)
class Criterion:
    id: str
    category: str
    description: str
    weight: float


@dataclass(frozen=True)
class Gate:
    """A must-pass requirement evaluated BEFORE weighted scoring.

    A failed gate eliminates the vendor regardless of weighted total. `condition`
    is the natural-language pass test the model judges against gathered evidence.
    """
    id: str
    criterion_id: str
    label: str
    condition: str


# Category display order and weights (category weight == sum of its criteria).
CATEGORIES = [
    "Functional & Use-Case Fit",
    "Integration & Architecture Fit",
    "AI & Agentic Capability",
    "Data, Security & Compliance (Regulated)",
    "Total Cost of Ownership (3-yr)",
    "Vendor Viability & Roadmap",
    "Switching & Lock-in Risk",
    "Implementation & Support",
]

CRITERIA: list[Criterion] = [
    # Functional & Use-Case Fit (0.20)
    Criterion("core_use_case", CATEGORIES[0], "Core use-case coverage (the jobs you actually need done)", 0.08),
    Criterion("channel_breadth", CATEGORIES[0], "Channel & campaign breadth", 0.05),
    Criterion("personalization_depth", CATEGORIES[0], "Personalization & segmentation depth", 0.04),
    Criterion("reporting_attribution", CATEGORIES[0], "Reporting & attribution quality", 0.03),
    # Integration & Architecture Fit (0.18)
    Criterion("native_connectors", CATEGORIES[1], "Native connectors to existing stack (CDP / CRM / DW)", 0.07),
    Criterion("api_quality", CATEGORIES[1], "API quality, webhooks & extensibility", 0.05),
    Criterion("data_model_identity", CATEGORIES[1], "Data model & identity-resolution fit", 0.04),
    Criterion("realtime_batch", CATEGORIES[1], "Real-time vs. batch fit for use case", 0.02),
    # AI & Agentic Capability (0.15)
    Criterion("model_quality", CATEGORIES[2], "Model quality & relevance to the use case", 0.04),
    Criterion("eval_methodology", CATEGORIES[2], "Eval methodology & output-quality controls", 0.03),
    Criterion("agentic_autonomy", CATEGORIES[2], "Agentic autonomy & human-in-the-loop controls", 0.03),
    Criterion("model_flexibility", CATEGORIES[2], "Model flexibility / no model lock-in (BYO-model)", 0.03),
    Criterion("prompt_governance", CATEGORIES[2], "Prompt & output data governance", 0.02),
    # Data, Security & Compliance (0.16)
    Criterion("data_residency", CATEGORIES[3], "Data residency & sovereignty", 0.04),
    Criterion("security_posture", CATEGORIES[3], "Security posture (SOC 2 / ISO 27001 / pen-test)", 0.04),
    Criterion("regulatory_fit", CATEGORIES[3], "Regulatory fit (GDPR, model-risk mgmt, audit trails)", 0.04),
    Criterion("pii_handling", CATEGORIES[3], "PII handling & data minimization", 0.02),
    Criterion("subprocessor_risk", CATEGORIES[3], "Sub-processor & supply-chain risk", 0.02),
    # Total Cost of Ownership (0.12)
    Criterion("licensing_cost", CATEGORIES[4], "Licensing / consumption cost", 0.05),
    Criterion("implementation_cost", CATEGORIES[4], "Implementation & integration cost", 0.04),
    Criterion("run_ops_cost", CATEGORIES[4], "Internal run / ops cost", 0.03),
    # Vendor Viability & Roadmap (0.10)
    Criterion("financial_stability", CATEGORIES[5], "Financial stability / funding runway", 0.03),
    Criterion("roadmap_alignment", CATEGORIES[5], "Roadmap alignment & sustained AI investment", 0.04),
    Criterion("reference_customers_fs", CATEGORIES[5], "Reference customers in regulated FS", 0.03),
    # Switching & Lock-in Risk (0.05)
    Criterion("data_portability", CATEGORIES[6], "Data portability & exit terms", 0.03),
    Criterion("contractual_flexibility", CATEGORIES[6], "Contractual flexibility (term, pricing, scope)", 0.02),
    # Implementation & Support (0.04)
    Criterion("time_to_value", CATEGORIES[7], "Time-to-value / implementation effort", 0.02),
    Criterion("support_sla", CATEGORIES[7], "Support SLAs & CSM quality", 0.02),
]

# Default must-pass gates for a regulated-FS context. Editable in the UI.
DEFAULT_GATES: list[Gate] = [
    Gate(
        id="gate_security",
        criterion_id="security_posture",
        label="Mandatory security certification",
        condition="Vendor holds a current SOC 2 Type II OR ISO 27001 certification, "
                  "evidenced by something stronger than a marketing claim.",
    ),
    Gate(
        id="gate_residency",
        criterion_id="data_residency",
        label="EU/UK data residency",
        condition="Vendor can contractually guarantee data residency / processing "
                  "in the EU or UK for the buyer's data.",
    ),
]


def criteria_by_category() -> dict[str, list[Criterion]]:
    out: dict[str, list[Criterion]] = {c: [] for c in CATEGORIES}
    for cr in CRITERIA:
        out[cr.category].append(cr)
    return out


def category_weight(category: str) -> float:
    return round(sum(c.weight for c in CRITERIA if c.category == category), 4)


def _validate() -> None:
    total = round(sum(c.weight for c in CRITERIA), 6)
    assert total == 1.0, f"Criteria weights must sum to 1.00, got {total}"
    ids = [c.id for c in CRITERIA]
    assert len(ids) == len(set(ids)), "Duplicate criterion IDs"
    for g in DEFAULT_GATES:
        assert g.criterion_id in ids, f"Gate {g.id} points at unknown criterion"


_validate()
