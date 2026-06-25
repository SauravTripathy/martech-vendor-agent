"""
Offline smoke test: stubs llm.* calls so the full graph, evidence-tier cap,
scoring math, smarter judge consistency check, and xlsx export run with NO API
key / network.

Run: python -m tests.smoke_test
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
import graph as graph_mod
import llm


# --------------------------------------------------------------------------- #
# Stubs
# --------------------------------------------------------------------------- #

def fake_gather(vendor, use_case, category, criteria, documents=""):
    # Return one tier-3 and one tier-1 item per category.
    items = []
    for i, c in enumerate(criteria):
        tier = 3 if i % 2 == 0 else 1
        items.append(
            {
                "criterion_id": c.id,
                "claim": f"{vendor} claim for {c.id}",
                "tier": tier,
                "source_title": f"src-{c.id}",
                "source_url": f"https://example.com/{c.id}",
                "snippet": "snippet",
            }
        )
    return items, {
        "provider": "anthropic",
        "input_tokens": 100,
        "output_tokens": 50,
        "web_search_requests": 1,
    }


def _score_payload(score: int):
    scores = []
    for c in config.CRITERIA:
        scores.append(
            {
                "criterion_id": c.id,
                "score": score,
                "evidence_tier": 3,
                "confidence": 0.8,
                "capped": False,
                "is_gap": False,
                "rationale": "stub",
            }
        )
    return {"scores": scores}


def make_score_stub(score: int):
    def _stub(vendor, use_case, criteria, evidence):
        return _score_payload(score), {
            "provider": "anthropic",
            "input_tokens": 200,
            "output_tokens": 100,
            "web_search_requests": 0,
        }

    return _stub


def make_judge_stub(score: int, material: bool = False):
    def _stub(vendor, use_case, criteria, evidence, primary_scores):
        payload = _score_payload(score)
        findings = []
        if material:
            findings.append(
                {
                    "criterion_id": config.CRITERIA[0].id,
                    "issue_type": "inflated_score",
                    "severity": "high",
                    "primary_score": 5,
                    "judge_score": score,
                    "recommended_score": score,
                    "material": True,
                    "explanation": "Primary score is too high for the available evidence.",
                }
            )
        payload["findings"] = findings
        payload["summary"] = "stub judge summary"
        return payload, {
            "provider": "openai",
            "input_tokens": 150,
            "output_tokens": 75,
            "web_search_requests": 0,
        }

    return _stub


def run_case(name: str, primary_score: int, judge_score: int, material: bool = False):
    llm.gather_evidence_for_category = fake_gather
    llm.score_all = make_score_stub(primary_score)
    llm.judge_review = make_judge_stub(judge_score, material=material)

    state = graph_mod.run_evaluation("TestVendor", "FS use-case")

    print(f"\n=== {name} ===")
    print(f"weighted_total={state['weighted_total']} normalized={state['normalized']}")
    consistency = state.get("consistency")
    print(f"consistency={None if consistency is None else consistency.agreement_within_1}")
    print(f"material_findings={None if consistency is None else consistency.material_issues_count}")
    m = state["metrics"]
    print(f"llm_calls={m.llm_calls} tokens_in={m.input_tokens} latency_s={m.total_seconds}")

    return state


def main():
    # Case A: both models agree -> consistency runs, 100% agreement.
    a = run_case("A agree", primary_score=5, judge_score=5)
    assert "eliminated" not in a
    assert a["consistency"] is not None
    assert a["consistency"].agreement_within_1 == 1.0
    assert abs(a["weighted_total"] - 5.0) < 1e-6, a["weighted_total"]
    assert abs(a["normalized"] - 1.0) < 1e-6, a["normalized"]

    # Case B: judge materially disagrees -> consistency still runs and surfaces finding.
    b = run_case("B material disagreement", primary_score=5, judge_score=3, material=True)
    assert b["consistency"] is not None
    assert b["consistency"].agreement_within_1 == 0.0
    assert b["consistency"].mean_abs_diff == 2.0
    assert b["consistency"].material_issues_count >= 1
    assert len(b["consistency"].recommended_adjustments) >= 1

    # Case C: models differ by 1 -> agreement within 1 remains 100%, no material divergence.
    c = run_case("C small disagreement", primary_score=5, judge_score=4)
    assert c["consistency"].agreement_within_1 == 1.0
    assert c["consistency"].mean_abs_diff == 1.0
    assert len(c["consistency"].divergences) == 0

    # xlsx export against the real template if present.
    template = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "MarTech_Vendor_Scorecard_Single.xlsx",
    )
    if os.path.exists(template):
        from scorecard import populate_template

        out = "/tmp/smoke_scorecard.xlsx"
        populate_template(template, out, a)
        assert os.path.exists(out)
        print(f"\nxlsx export OK -> {out}")

    print("\nALL SMOKE ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
