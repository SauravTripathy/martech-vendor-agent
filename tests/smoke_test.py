"""
Offline smoke test: stubs the llm.* calls so the full graph, gate short-circuit,
evidence-tier cap, scoring math, and xlsx export run with NO API key / network.

Run:  python -m tests.smoke_test
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import llm
import graph as graph_mod


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #
def fake_gather(vendor, use_case, category, criteria):
    # Return one tier-1 (self-asserted) and one tier-3 (named ref) item per category.
    items = []
    for i, c in enumerate(criteria):
        tier = 3 if i % 2 == 0 else 1
        items.append({
            "criterion_id": c.id, "claim": f"{vendor} claim for {c.id}",
            "tier": tier, "source_title": f"src-{c.id}",
            "source_url": f"https://example.com/{c.id}", "snippet": "snippet",
        })
    return items, {"input_tokens": 100, "output_tokens": 50, "web_search_requests": 1}


def _score_payload(gate_pass: bool, high: bool):
    scores = []
    for c in config.CRITERIA:
        scores.append({
            "criterion_id": c.id, "score": 5 if high else 4,
            "evidence_tier": 3, "confidence": 0.8, "capped": False,
            "is_gap": False, "rationale": "stub",
        })
    gates = [{"gate_id": g.id, "passed": gate_pass, "evidence_tier": 4,
              "rationale": "stub gate"} for g in config.DEFAULT_GATES]
    return {"scores": scores, "gates": gates}


def make_score_stub(gate_pass, high):
    def _stub(vendor, use_case, criteria, gates, evidence):
        return _score_payload(gate_pass, high), {
            "input_tokens": 200, "output_tokens": 100, "web_search_requests": 0}
    return _stub


def run_case(name, gate_pass, primary_high, judge_high):
    llm.gather_evidence_for_category = fake_gather
    llm.score_all = make_score_stub(gate_pass, primary_high)
    llm.judge_rescore = make_score_stub(gate_pass, judge_high)
    state = graph_mod.run_evaluation("TestVendor", "FS use-case",
                                     gates_cfg=config.DEFAULT_GATES)
    print(f"\n=== {name} ===")
    print(f"eliminated={state['eliminated']} reasons={state.get('elimination_reasons')}")
    print(f"weighted_total={state['weighted_total']} normalized={state['normalized']}")
    consistency = state.get("consistency")
    print(f"consistency={'skipped' if consistency is None else consistency.agreement_within_1}")
    m = state["metrics"]
    print(f"llm_calls={m.llm_calls} tokens_in={m.input_tokens} latency_s={m.total_seconds}")
    return state


def main():
    # Case A: passes gates, both models agree -> consistency runs, ~100% agreement.
    a = run_case("A passes, agree", gate_pass=True, primary_high=True, judge_high=True)
    assert not a["eliminated"]
    assert a["consistency"] is not None
    assert a["consistency"].agreement_within_1 == 1.0
    # All criteria scored 5 -> weighted total 5.0, normalized 1.0.
    assert abs(a["weighted_total"] - 5.0) < 1e-6, a["weighted_total"]
    assert abs(a["normalized"] - 1.0) < 1e-6, a["normalized"]

    # Case B: fails a gate -> eliminated, consistency SKIPPED (cost short-circuit).
    b = run_case("B fails gate", gate_pass=False, primary_high=True, judge_high=True)
    assert b["eliminated"]
    assert b.get("consistency") is None, "judge pass should be skipped on elimination"
    # Judge call must not have run -> exactly 8 gather + 1 score = 9 calls.
    assert b["metrics"].llm_calls == 9, b["metrics"].llm_calls

    # Case C: models disagree by 1 (5 vs 4) -> within-1 agreement stays 100%,
    # no >=2 divergences surfaced.
    c = run_case("C disagree by 1", gate_pass=True, primary_high=True, judge_high=False)
    assert c["consistency"].agreement_within_1 == 1.0
    assert c["consistency"].mean_abs_diff == 1.0
    assert len(c["consistency"].divergences) == 0

    # xlsx export against the real template if present.
    template = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "MarTech_Vendor_Scorecard_Single.xlsx")
    if os.path.exists(template):
        from scorecard import populate_template
        out = "/tmp/smoke_scorecard.xlsx"
        populate_template(template, out, a)
        assert os.path.exists(out)
        print(f"\nxlsx export OK -> {out}")

    print("\nALL SMOKE ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
