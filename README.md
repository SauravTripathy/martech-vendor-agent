# MarTech Vendor Evaluation Agent

A single-vendor deep-dive agent built with **LangGraph** (orchestration), the
**Anthropic API** (reasoning + web search), and **Gradio** (UI). It gathers public
evidence on a MarTech vendor, scores it against your fixed weighted rubric, runs
must-pass gates, cross-checks the scoring with a second model, and exports a
populated copy of `MarTech_Vendor_Scorecard_Single.xlsx`.

> Initial assessment from publicly available sources. A vendor can have a solid
> product yet score low on evidence or articulation — scores are **capped by
> evidence strength**, and gaps are flagged, not guessed.

## Architecture

```
START
  └─> gather_evidence      Node 1 — web_search per category → tiered Evidence[]
        └─> evaluate        Node 2 — gates FIRST, then evidence-tiered scoring
              ├─ eliminated? ─────────────────────────────> END   (skip judge spend)
              └─> consistency_check   Node 3 — 2nd model re-scores SAME evidence → END
```

Three design choices are deliberate (they were the cheap-now / expensive-later
decisions):

1. **Gates are control flow, not a field.** A failed must-pass gate (e.g. no
   SOC 2, no EU residency) short-circuits the graph in `route_after_evaluate` —
   the vendor is marked *eliminated*, the headline says so loudly, and the
   second-model pass is skipped to avoid spending tokens ranking a disqualified
   vendor.
2. **Scores are capped by evidence tier.** Tier 1 = vendor's own marketing
   (self-asserted) … Tier 4 = verifiable artifact (cert registry, attestation).
   A claim supported only by Tier 1 cannot exceed 3/5 — enforced both in the
   prompt *and* defensively in code (`_apply_tier_cap`). This stops a
   public-source agent from awarding "best-in-class" off a brochure.
3. **"Consistency", not "accuracy".** With no ground truth, Node 3 measures
   model *agreement*: a different model re-scores the **same** evidence at low
   temperature, isolating scoring disagreement from search noise. It reports
   within-1 agreement, mean absolute difference, and surfaces any criterion
   where the two models diverge by ≥2 points for human review.

`null` (not 0) is used for non-evaluable criteria, so the normalized score
divides by the evaluable max and a missing data point can't silently tank a vendor.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
# Model IDs are placeholders — set these to models on YOUR account.
# The PRIMARY model must support the web_search tool. See https://docs.claude.com
export MARTECH_PRIMARY_MODEL=claude-sonnet-4-5
export MARTECH_JUDGE_MODEL=claude-3-5-haiku-latest
python app.py
```

Keep `MarTech_Vendor_Scorecard_Single.xlsx` next to `app.py` (or point
`MARTECH_TEMPLATE` at it) to enable the populated-scorecard download.

## Offline smoke test (no API key needed)

```bash
python -m tests.smoke_test
```

Stubs the LLM calls and asserts: full path runs all 10 LLM calls; a gate failure
eliminates and skips the judge pass (9 calls); a 1-point disagreement stays
within tolerance; and the xlsx export writes scores to col D / sources to col F.

## Files

| File | Role |
|------|------|
| `config.py` | Criteria taxonomy, weights, gates, evidence tiers, rubric |
| `schema.py` | Typed state, evidence, scores, gates, metrics |
| `llm.py` | Anthropic calls: web-search evidence, scoring, judge re-score, JSON parsing |
| `graph.py` | LangGraph nodes, gate routing, scoring math |
| `scorecard.py` | Populate the xlsx template |
| `app.py` | Gradio UI |
| `tests/smoke_test.py` | Offline wiring test |

## Known limits / next steps

- Evidence tiering is model-judged; for production, verify Tier-4 claims against
  the actual source (e.g. fetch the trust portal) rather than trusting the label.
- Consistency is a 2-model, 1-sample check. For a tighter signal, add
  self-consistency (sample N, take the per-criterion median) and track score
  variance as a first-class eval metric.
- This is the single-vendor build. Comparison mode = run N vendors and rank the
  survivors of the gate stage; market mode is a separate aggregation.
- Cost: ~8 search calls + 2 scoring calls per vendor. Tune `WEB_SEARCH_MAX_USES`
  and consider caching evidence per vendor between runs.
