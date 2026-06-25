"""
LangGraph wiring for the MarTech Vendor Evaluation Agent.

    START
      |
      v
  gather_evidence        (Node 1: web search per category -> Evidence[])
      |
      v
  evaluate               (Node 2: gates first, then evidence-tiered scoring)
      |
      |-- eliminated? --> END        (gate failure short-circuits; no judge spend)
      |
      v
  consistency_check      (Node 3: second model re-scores SAME evidence)
      |
      v
     END
"""
from __future__ import annotations

import time

from langgraph.graph import StateGraph, START, END

import config
import llm
from schema import (
    AgentState, Evidence, CriterionScore, GateResult,
    ConsistencyReport, RunMetrics,
)

_CRIT_BY_ID = {c.id: c for c in config.CRITERIA}


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
                vendor, use_case, category, crits, documents)
            metrics.add_usage(usage)
            for it in items:
                cid = it.get("criterion_id")
                if cid not in _CRIT_BY_ID:
                    continue
                tier = it.get("tier")
                if tier not in (1, 2, 3, 4):
                    continue
                evidence.append(Evidence(
                    criterion_id=cid,
                    claim=str(it.get("claim", ""))[:300],
                    tier=int(tier),
                    source_title=str(it.get("source_title", ""))[:200],
                    source_url=str(it.get("source_url", ""))[:500],
                    snippet=str(it.get("snippet", ""))[:400],
                ))
        except Exception as exc:  # keep going; a category failing is not fatal
            errors.append(f"[gather:{category}] {exc}")

    metrics.node_seconds["gather_evidence"] = round(time.time() - t0, 2)
    return {"evidence": evidence, "errors": errors}


# --------------------------------------------------------------------------- #
# Node 2
# --------------------------------------------------------------------------- #
def _apply_tier_cap(score, tier):
    """Enforce the evidence-tier ceiling defensively, in code, not just in-prompt."""
    if score is None or tier is None:
        return score, False
    cap = config.EVIDENCE_TIER_CAP.get(int(tier), 5)
    if score > cap:
        return cap, True
    return score, False


def _build_scores(payload: dict, evidence: list[Evidence]) -> list[CriterionScore]:
    by_crit = {s.get("criterion_id"): s for s in payload.get("scores", [])}
    ev_by_crit: dict[str, list[Evidence]] = {}
    for e in evidence:
        ev_by_crit.setdefault(e.criterion_id, []).append(e)

    out: list[CriterionScore] = []
    for cr in config.CRITERIA:
        raw = by_crit.get(cr.id, {})
        ev = ev_by_crit.get(cr.id, [])
        best_tier = max((e.tier for e in ev), default=None)
        score = raw.get("score", None)
        if score is not None:
            try:
                score = int(score)
                score = min(5, max(1, score))
            except Exception:
                score = None
        tier = raw.get("evidence_tier", best_tier)
        score, code_capped = _apply_tier_cap(score, tier)
        is_gap = bool(raw.get("is_gap", not ev)) if score is None else False
        out.append(CriterionScore(
            criterion_id=cr.id,
            category=cr.category,
            description=cr.description,
            weight=cr.weight,
            score=score,
            evidence_tier=tier,
            confidence=float(raw.get("confidence", 0.0) or 0.0),
            rationale=str(raw.get("rationale", "")),
            sources=[e.source_url for e in ev if e.source_url],
            is_gap=is_gap,
            capped=bool(raw.get("capped", False)) or code_capped,
        ))
    return out


def _build_gates(payload: dict, gates_cfg: list) -> list[GateResult]:
    by_gate = {g.get("gate_id"): g for g in payload.get("gates", [])}
    results = []
    for g in gates_cfg:
        raw = by_gate.get(g.id, {})
        results.append(GateResult(
            gate_id=g.id,
            criterion_id=g.criterion_id,
            label=g.label,
            passed=raw.get("passed", None),
            rationale=str(raw.get("rationale", "")),
            evidence_tier=raw.get("evidence_tier", None),
        ))
    return results


def _weighted_total(scores: list[CriterionScore]) -> tuple[float, float]:
    total = round(sum(s.weighted for s in scores), 4)
    # normalized = total / max achievable on the criteria that were evaluable
    evaluable_weight = sum(s.weight for s in scores if s.score is not None)
    normalized = round(total / (5 * evaluable_weight), 4) if evaluable_weight else 0.0
    return total, normalized


def evaluate(state: AgentState) -> dict:
    t0 = time.time()
    metrics: RunMetrics = state["metrics"]
    errors = list(state.get("errors", []))
    gates_cfg = state.get("gates_cfg", config.DEFAULT_GATES)

    try:
        payload, usage = llm.score_all(
            state["vendor_name"], state["use_case"],
            config.CRITERIA, gates_cfg, state["evidence"],
        )
        metrics.add_usage(usage)
    except Exception as exc:
        errors.append(f"[evaluate] {exc}")
        payload = {"scores": [], "gates": []}

    scores = _build_scores(payload, state["evidence"])
    gate_results = _build_gates(payload, gates_cfg)
    total, normalized = _weighted_total(scores)

    failed = [g for g in gate_results if g.passed is False]
    eliminated = len(failed) > 0
    reasons = [f"{g.label}: {g.rationale}" for g in failed]

    metrics.node_seconds["evaluate"] = round(time.time() - t0, 2)
    return {
        "scores": scores,
        "gate_results": gate_results,
        "eliminated": eliminated,
        "elimination_reasons": reasons,
        "weighted_total": total,
        "normalized": normalized,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Node 3
# --------------------------------------------------------------------------- #
def consistency_check(state: AgentState) -> dict:
    t0 = time.time()
    metrics: RunMetrics = state["metrics"]
    errors = list(state.get("errors", []))
    gates_cfg = state.get("gates_cfg", config.DEFAULT_GATES)
    primary: list[CriterionScore] = state["scores"]

    try:
        payload, usage = llm.judge_rescore(
            state["vendor_name"], state["use_case"],
            config.CRITERIA, gates_cfg, state["evidence"],
        )
        metrics.add_usage(usage)
        judge_scores = _build_scores(payload, state["evidence"])
    except Exception as exc:
        errors.append(f"[consistency] {exc}")
        metrics.node_seconds["consistency_check"] = round(time.time() - t0, 2)
        return {"consistency": None, "errors": errors}

    j_by_id = {s.criterion_id: s for s in judge_scores}
    diffs, within1, overlap = [], 0, 0
    for p in primary:
        j = j_by_id.get(p.criterion_id)
        if j is None or p.score is None or j.score is None:
            continue
        overlap += 1
        delta = abs(p.score - j.score)
        if delta <= 1:
            within1 += 1
        if delta >= 2:  # material divergence worth surfacing
            diffs.append({
                "criterion_id": p.criterion_id,
                "description": p.description,
                "primary": p.score,
                "judge": j.score,
                "delta": delta,
            })

    report = ConsistencyReport(
        agreement_within_1=round(within1 / overlap, 3) if overlap else 0.0,
        mean_abs_diff=round(
            sum(abs(p.score - j_by_id[p.criterion_id].score)
                for p in primary
                if p.score is not None and p.criterion_id in j_by_id
                and j_by_id[p.criterion_id].score is not None) / overlap, 3
        ) if overlap else 0.0,
        divergences=sorted(diffs, key=lambda d: -d["delta"]),
        judge_model=config.JUDGE_MODEL,
        note=f"{overlap} criteria scored by both providers; "
             f"{len(diffs)} diverged by >=2 points.",
    )
    metrics.node_seconds["consistency_check"] = round(time.time() - t0, 2)
    return {"consistency": report, "errors": errors}


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
def route_after_evaluate(state: AgentState) -> str:
    # Gate failure eliminates the vendor — skip the judge pass to save tokens.
    return END if state.get("eliminated") else "consistency_check"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("gather_evidence", gather_evidence)
    g.add_node("evaluate", evaluate)
    g.add_node("consistency_check", consistency_check)

    g.add_edge(START, "gather_evidence")
    g.add_edge("gather_evidence", "evaluate")
    g.add_conditional_edges("evaluate", route_after_evaluate,
                            {"consistency_check": "consistency_check", END: END})
    g.add_edge("consistency_check", END)
    return g.compile()


def run_evaluation(vendor_name: str, use_case: str, gates_cfg=None,
                   documents: str = "") -> AgentState:
    metrics = RunMetrics()
    t0 = time.time()
    app = build_graph()
    initial: AgentState = {
        "vendor_name": vendor_name,
        "use_case": use_case,
        "documents": documents,
        "gates_cfg": gates_cfg if gates_cfg is not None else config.DEFAULT_GATES,
        "metrics": metrics,
        "errors": [],
    }
    final = app.invoke(initial)
    metrics.total_seconds = round(time.time() - t0, 2)
    final["metrics"] = metrics
    return final
