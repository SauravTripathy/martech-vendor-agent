"""
LangGraph wiring for the MarTech Vendor Evaluation Agent.

 START
  |
  v
 gather_evidence      (Node 1: web search + uploaded docs -> Evidence[])
  |
  v
 evaluate             (Node 2: evidence-tiered scoring)
  |
  v
 consistency_check    (Node 3: smarter judge re-scores + audits primary scores)
  |
  v
 END

Must-have gate elimination has been removed. The workflow always runs the judge
consistency check so users can see whether the score is well-supported.
"""

from __future__ import annotations

import time
from typing import Any

from langgraph.graph import END, START, StateGraph

import config
import llm
from schema import AgentState, ConsistencyReport, CriterionScore, Evidence, RunMetrics

_CRIT_BY_ID = {c.id: c for c in config.CRITERIA}
_ISSUES_THAT_CHALLENGE_SUPPORT = {
    "inflated_score",
    "unsupported_claim",
    "weak_rationale",
    "score_disagreement",
}


# --------------------------------------------------------------------------- #
# Node 1
# --------------------------------------------------------------------------- #

def gather_evidence(state: AgentState) -> dict:
    t0 = time.time()
    metrics: RunMetrics = state["metrics"]
    vendor, use_case = state["vendor_name"], state["use_case"]
    documents = state.get("documents", "")
    evidence: list[Evidence] = []
    errors = list(state.get("errors", []))

    for category, crits in config.criteria_by_category().items():
        try:
            items, usage = llm.gather_evidence_for_category(
                vendor,
                use_case,
                category,
                crits,
                documents,
            )
            metrics.add_usage(usage)

            for it in items:
                cid = it.get("criterion_id")
                if cid not in _CRIT_BY_ID:
                    continue

                tier = _clean_tier(it.get("tier"))
                if tier is None:
                    continue

                evidence.append(
                    Evidence(
                        criterion_id=cid,
                        claim=str(it.get("claim", ""))[:300],
                        tier=tier,
                        source_title=str(it.get("source_title", ""))[:200],
                        source_url=str(it.get("source_url", ""))[:500],
                        snippet=str(it.get("snippet", ""))[:400],
                    )
                )
        except Exception as exc:  # keep going; a category failing is not fatal
            errors.append(f"[gather:{category}] {exc}")

    metrics.node_seconds["gather_evidence"] = round(time.time() - t0, 2)
    return {"evidence": evidence, "errors": errors}


# --------------------------------------------------------------------------- #
# Node 2
# --------------------------------------------------------------------------- #

def _clean_tier(value: Any) -> int | None:
    try:
        tier = int(value)
    except Exception:
        return None
    return tier if tier in (1, 2, 3, 4) else None


def _clean_score(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.lower().strip() in {"", "none", "null", "n/a", "na"}:
        return None
    try:
        score = int(value)
    except Exception:
        return None
    return min(5, max(1, score))


def _clean_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return min(1.0, max(0.0, out))


def _apply_tier_cap(score: int | None, tier: int | None) -> tuple[int | None, bool]:
    """Enforce the evidence-tier ceiling defensively, in code, not just in-prompt."""
    if score is None or tier is None:
        return score, False

    cap = config.EVIDENCE_TIER_CAP.get(int(tier), 5)
    if score > cap:
        return cap, True
    return score, False


def _build_scores(payload: dict, evidence: list[Evidence]) -> list[CriterionScore]:
    raw_scores = payload.get("scores", []) if isinstance(payload, dict) else []
    by_crit = {s.get("criterion_id"): s for s in raw_scores if isinstance(s, dict)}

    ev_by_crit: dict[str, list[Evidence]] = {}
    for e in evidence:
        ev_by_crit.setdefault(e.criterion_id, []).append(e)

    out: list[CriterionScore] = []
    for cr in config.CRITERIA:
        raw = by_crit.get(cr.id, {})
        ev = ev_by_crit.get(cr.id, [])
        best_tier = max((e.tier for e in ev), default=None)

        score = _clean_score(raw.get("score"))
        tier = _clean_tier(raw.get("evidence_tier")) or best_tier
        score, code_capped = _apply_tier_cap(score, tier)
        is_gap = bool(raw.get("is_gap", not ev)) if score is None else False

        out.append(
            CriterionScore(
                criterion_id=cr.id,
                category=cr.category,
                description=cr.description,
                weight=cr.weight,
                score=score,
                evidence_tier=tier,
                confidence=_clean_float(raw.get("confidence", 0.0)),
                rationale=str(raw.get("rationale", ""))[:500],
                sources=[e.source_url for e in ev if e.source_url],
                is_gap=is_gap,
                capped=bool(raw.get("capped", False)) or code_capped,
            )
        )
    return out


def _weighted_total(scores: list[CriterionScore]) -> tuple[float, float]:
    total = round(sum(s.weighted for s in scores), 4)
    evaluable_weight = sum(s.weight for s in scores if s.score is not None)
    normalized = round(total / (5 * evaluable_weight), 4) if evaluable_weight else 0.0
    return total, normalized


def evaluate(state: AgentState) -> dict:
    t0 = time.time()
    metrics: RunMetrics = state["metrics"]
    errors = list(state.get("errors", []))

    try:
        payload, usage = llm.score_all(
            state["vendor_name"],
            state["use_case"],
            config.CRITERIA,
            state["evidence"],
        )
        metrics.add_usage(usage)
    except Exception as exc:
        errors.append(f"[evaluate] {exc}")
        payload = {"scores": []}

    scores = _build_scores(payload, state["evidence"])
    total, normalized = _weighted_total(scores)

    metrics.node_seconds["evaluate"] = round(time.time() - t0, 2)
    return {
        "scores": scores,
        "weighted_total": total,
        "normalized": normalized,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Node 3
# --------------------------------------------------------------------------- #

def _score_delta(a: int | None, b: int | None) -> int | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def _normalize_judge_findings(
    payload: dict,
    primary: list[CriterionScore],
    judge_scores: list[CriterionScore],
) -> list[dict]:
    primary_by_id = {s.criterion_id: s for s in primary}
    judge_by_id = {s.criterion_id: s for s in judge_scores}
    raw_findings = payload.get("findings", []) if isinstance(payload, dict) else []
    findings: list[dict] = []

    seen: set[str] = set()
    for raw in raw_findings:
        if not isinstance(raw, dict):
            continue
        cid = raw.get("criterion_id")
        if cid not in _CRIT_BY_ID:
            continue

        p = primary_by_id.get(cid)
        j = judge_by_id.get(cid)
        primary_score = _clean_score(raw.get("primary_score"))
        judge_score = _clean_score(raw.get("judge_score"))
        recommended_score = _clean_score(raw.get("recommended_score"))

        if primary_score is None and p is not None:
            primary_score = p.score
        if judge_score is None and j is not None:
            judge_score = j.score
        if recommended_score is None:
            recommended_score = judge_score

        issue_type = str(raw.get("issue_type", "none") or "none").strip().lower()
        severity = str(raw.get("severity", "low") or "low").strip().lower()
        if severity not in {"low", "medium", "high"}:
            severity = "low"

        material = bool(raw.get("material", False))
        delta = _score_delta(primary_score, judge_score)
        if delta is not None and delta >= 2:
            issue_type = "score_disagreement" if issue_type == "none" else issue_type
            material = True
            severity = "high" if severity == "low" else severity

        finding = {
            "criterion_id": cid,
            "description": _CRIT_BY_ID[cid].description,
            "issue_type": issue_type,
            "severity": severity,
            "primary_score": primary_score,
            "judge_score": judge_score,
            "recommended_score": recommended_score,
            "delta": delta,
            "material": material,
            "explanation": str(raw.get("explanation", ""))[:500],
        }
        findings.append(finding)
        seen.add(cid)

    # Defensive fallback: if the judge re-score differs materially but the LLM did
    # not emit a finding, create one in code so the consistency report is never silent.
    for p in primary:
        j = judge_by_id.get(p.criterion_id)
        if j is None:
            continue
        delta = _score_delta(p.score, j.score)
        if delta is None or delta < 2 or p.criterion_id in seen:
            continue
        findings.append(
            {
                "criterion_id": p.criterion_id,
                "description": p.description,
                "issue_type": "score_disagreement",
                "severity": "high",
                "primary_score": p.score,
                "judge_score": j.score,
                "recommended_score": j.score,
                "delta": delta,
                "material": True,
                "explanation": "Primary and judge scores diverged by at least 2 points.",
            }
        )

    return findings


def _build_consistency_report(
    payload: dict,
    primary: list[CriterionScore],
    judge_scores: list[CriterionScore],
) -> ConsistencyReport:
    j_by_id = {s.criterion_id: s for s in judge_scores}
    diffs: list[dict] = []
    within1 = 0
    overlap = 0
    abs_diffs: list[int] = []

    for p in primary:
        j = j_by_id.get(p.criterion_id)
        if j is None or p.score is None or j.score is None:
            continue

        overlap += 1
        delta = abs(p.score - j.score)
        abs_diffs.append(delta)
        if delta <= 1:
            within1 += 1
        if delta >= 2:
            diffs.append(
                {
                    "criterion_id": p.criterion_id,
                    "description": p.description,
                    "primary": p.score,
                    "judge": j.score,
                    "delta": delta,
                }
            )

    findings = _normalize_judge_findings(payload, primary, judge_scores)

    # Attach finding explanations to divergence rows when possible.
    finding_by_id = {f["criterion_id"]: f for f in findings}
    for d in diffs:
        f = finding_by_id.get(d["criterion_id"])
        if f:
            d["issue_type"] = f.get("issue_type")
            d["severity"] = f.get("severity")
            d["recommended_score"] = f.get("recommended_score")
            d["explanation"] = f.get("explanation")

    recommended_adjustments = [
        f for f in findings
        if f.get("recommended_score") != f.get("primary_score") or f.get("material")
    ]
    missing_evidence = [f for f in findings if f.get("issue_type") == "missing_evidence"]
    support_issues = [
        f for f in findings
        if f.get("issue_type") in _ISSUES_THAT_CHALLENGE_SUPPORT
    ]
    material_issues_count = sum(1 for f in findings if f.get("material"))

    summary = ""
    if isinstance(payload, dict):
        summary = str(payload.get("summary", ""))[:500]
    if not summary:
        summary = (
            f"{overlap} criteria scored by both providers; "
            f"{len(diffs)} diverged by >=2 points; "
            f"{material_issues_count} material judge findings."
        )

    return ConsistencyReport(
        agreement_within_1=round(within1 / overlap, 3) if overlap else 0.0,
        mean_abs_diff=round(sum(abs_diffs) / overlap, 3) if overlap else 0.0,
        divergences=sorted(diffs, key=lambda d: -d["delta"]),
        judge_findings=sorted(
            findings,
            key=lambda f: (
                0 if f.get("severity") == "high" else 1 if f.get("severity") == "medium" else 2,
                -(f.get("delta") or 0),
            ),
        ),
        recommended_adjustments=recommended_adjustments,
        missing_evidence=missing_evidence,
        support_issues=support_issues,
        material_issues_count=material_issues_count,
        judge_model=config.JUDGE_MODEL,
        note=summary,
    )


def consistency_check(state: AgentState) -> dict:
    t0 = time.time()
    metrics: RunMetrics = state["metrics"]
    errors = list(state.get("errors", []))
    primary: list[CriterionScore] = state["scores"]

    try:
        payload, usage = llm.judge_review(
            state["vendor_name"],
            state["use_case"],
            config.CRITERIA,
            state["evidence"],
            primary,
        )
        metrics.add_usage(usage)
        judge_scores = _build_scores(payload, state["evidence"])
    except Exception as exc:
        errors.append(f"[consistency] {exc}")
        metrics.node_seconds["consistency_check"] = round(time.time() - t0, 2)
        return {"consistency": None, "errors": errors}

    report = _build_consistency_report(payload, primary, judge_scores)
    metrics.node_seconds["consistency_check"] = round(time.time() - t0, 2)
    return {"consistency": report, "judge_scores": judge_scores, "errors": errors}


# --------------------------------------------------------------------------- #
# Graph
# --------------------------------------------------------------------------- #

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("gather_evidence", gather_evidence)
    g.add_node("evaluate", evaluate)
    g.add_node("consistency_check", consistency_check)

    g.add_edge(START, "gather_evidence")
    g.add_edge("gather_evidence", "evaluate")
    g.add_edge("evaluate", "consistency_check")
    g.add_edge("consistency_check", END)

    return g.compile()


def run_evaluation(
    vendor_name: str,
    use_case: str,
    gates_cfg=None,  # kept for backwards compatibility; ignored by design
    documents: str = "",
) -> AgentState:
    metrics = RunMetrics()
    t0 = time.time()
    app = build_graph()
    initial: AgentState = {
        "vendor_name": vendor_name,
        "use_case": use_case,
        "documents": documents,
        "metrics": metrics,
        "errors": [],
    }

    final = app.invoke(initial)
    metrics.total_seconds = round(time.time() - t0, 2)
    final["metrics"] = metrics
    return final
