"""
Gradio front-end for the MarTech Vendor Evaluation Agent.

Run:  python app.py   (needs ANTHROPIC_API_KEY in the environment)
"""
from __future__ import annotations

import os
import tempfile

import gradio as gr
import pandas as pd

import config
from graph import run_evaluation
from scorecard import populate_template

TEMPLATE = os.getenv(
    "MARTECH_TEMPLATE",
    os.path.join(os.path.dirname(__file__), "MarTech_Vendor_Scorecard_Single.xlsx"),
)


def _scores_dataframe(state: dict) -> pd.DataFrame:
    rows = []
    cat_order = {c: i for i, c in enumerate(config.CATEGORIES)}
    for s in sorted(state["scores"], key=lambda x: (cat_order[x.category], -x.weight)):
        rows.append({
            "Category": s.category,
            "Criterion": s.description,
            "Weight": s.weight,
            "Score": "" if s.score is None else s.score,
            "Weighted": round(s.weighted, 3),
            "Evid. tier": "" if s.evidence_tier is None else s.evidence_tier,
            "Conf.": round(s.confidence, 2),
            "Flags": " ".join(
                f for f, on in (("capped", s.capped), ("gap", s.is_gap)) if on
            ),
            "Rationale": s.rationale,
        })
    return pd.DataFrame(rows)


def _category_summary(state: dict) -> pd.DataFrame:
    rows = []
    for cat in config.CATEGORIES:
        cs = [s for s in state["scores"] if s.category == cat]
        wsum = round(sum(s.weighted for s in cs), 3)
        rows.append({
            "Category": cat,
            "Cat. weight": config.category_weight(cat),
            "Weighted contribution": wsum,
            "Gaps": sum(1 for s in cs if s.is_gap),
        })
    return pd.DataFrame(rows)


def _evidence_dataframe(state: dict) -> pd.DataFrame:
    desc = {c.id: c.description for c in config.CRITERIA}
    rows = [{
        "Criterion": desc.get(e.criterion_id, e.criterion_id),
        "Tier": e.tier,
        "Claim": e.claim,
        "Source": e.source_title,
        "URL": e.source_url,
    } for e in sorted(state["evidence"], key=lambda x: (x.criterion_id, -x.tier))]
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        [{"Criterion": "—", "Tier": "", "Claim": "No evidence gathered.", "Source": "", "URL": ""}]
    )


def _gates_markdown(state: dict) -> str:
    lines = ["### Must-pass gates"]
    for g in state.get("gate_results", []):
        icon = {True: "✅ PASS", False: "❌ FAIL", None: "⚠️ UNDETERMINED"}[g.passed]
        tier = f" (tier {g.evidence_tier})" if g.evidence_tier else ""
        lines.append(f"- **{g.label}** — {icon}{tier}: {g.rationale}")
    return "\n".join(lines)


def _headline_markdown(state: dict) -> str:
    if state.get("eliminated"):
        reasons = "; ".join(state.get("elimination_reasons", []))
        return (f"## ❌ {state['vendor_name']}: ELIMINATED\n"
                f"Failed must-pass gate(s): {reasons}\n\n"
                f"_Weighted score not used for ranking — a gate failure is disqualifying "
                f"regardless of total. (Indicative total: {state['weighted_total']:.2f}/5.)_")
    return (f"## {state['vendor_name']}\n"
            f"**Weighted score: {state['weighted_total']:.2f} / 5.00**  ·  "
            f"**Normalized: {state['normalized']*100:.1f}%** (of evaluable max)\n\n"
            f"_Initial assessment from public sources. Scores are capped by evidence "
            f"strength; gaps are flagged, not guessed._")


def _consistency_markdown(state: dict) -> str:
    c = state.get("consistency")
    if c is None:
        if state.get("eliminated"):
            return "### Consistency\n_Skipped — vendor eliminated at gate (saves judge-model spend)._"
        return "### Consistency\n_Not available._"
    lines = [
        "### Consistency (second-model cross-check)",
        f"- Judge model: `{c.judge_model}`",
        f"- Agreement within 1 point: **{c.agreement_within_1*100:.0f}%**",
        f"- Mean absolute score difference: **{c.mean_abs_diff:.2f}**",
        f"- {c.note}",
    ]
    if c.divergences:
        lines.append("\n**Criteria diverging ≥2 points (review these):**")
        for d in c.divergences:
            lines.append(f"- {d['description']}: primary {d['primary']} vs judge {d['judge']} "
                         f"(Δ{d['delta']})")
    return "\n".join(lines)


def _metrics_markdown(state: dict) -> str:
    m = state["metrics"]
    node_times = ", ".join(f"{k} {v}s" for k, v in m.node_seconds.items())
    return (
        "### Run metrics\n"
        f"- LLM calls: **{m.llm_calls}**  ·  web searches: **{m.web_search_requests}**\n"
        f"- Tokens: **{m.input_tokens:,}** in / **{m.output_tokens:,}** out\n"
        f"- Latency: **{m.total_seconds}s** total ({node_times})\n"
        + (f"\n**Warnings:** {len(state.get('errors', []))} — "
           + "; ".join(state.get("errors", [])[:3]) if state.get("errors") else "")
    )


def evaluate_vendor(vendor: str, use_case: str, gate_security: bool, gate_residency: bool):
    if not vendor.strip():
        raise gr.Error("Enter a vendor name.")
    gates = []
    if gate_security:
        gates.append(next(g for g in config.DEFAULT_GATES if g.id == "gate_security"))
    if gate_residency:
        gates.append(next(g for g in config.DEFAULT_GATES if g.id == "gate_residency"))

    state = run_evaluation(vendor.strip(), use_case.strip(), gates_cfg=gates)

    xlsx_path = None
    if os.path.exists(TEMPLATE):
        out = os.path.join(tempfile.mkdtemp(),
                           f"{vendor.strip().replace(' ', '_')}_scorecard.xlsx")
        try:
            xlsx_path = populate_template(TEMPLATE, out, state)
        except Exception as exc:
            state.setdefault("errors", []).append(f"[xlsx] {exc}")

    return (
        _headline_markdown(state),
        _gates_markdown(state),
        _scores_dataframe(state),
        _category_summary(state),
        _evidence_dataframe(state),
        _consistency_markdown(state),
        _metrics_markdown(state),
        xlsx_path,
    )


def build_ui():
    with gr.Blocks(title="MarTech Vendor Evaluation Agent",
                   theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown(
            "# MarTech Vendor Evaluation Agent\n"
            "Single-vendor deep-dive. Gathers public evidence, scores against a fixed "
            "weighted rubric (evidence-tier capped), runs must-pass gates, and cross-checks "
            "with a second model. _Decision aid, not an oracle._"
        )
        with gr.Row():
            with gr.Column(scale=1):
                vendor = gr.Textbox(label="Vendor name", placeholder="e.g. Braze")
                use_case = gr.Textbox(
                    label="Buyer use-case / context", lines=4,
                    placeholder="e.g. UK retail bank; real-time cross-channel orchestration "
                                "on Snowflake; must keep data in EU; needs SOC 2.",
                )
                gr.Markdown("**Must-pass gates**")
                gate_security = gr.Checkbox(value=True, label="Mandatory SOC 2 / ISO 27001")
                gate_residency = gr.Checkbox(value=True, label="EU/UK data residency")
                run_btn = gr.Button("Evaluate vendor", variant="primary")
            with gr.Column(scale=2):
                headline = gr.Markdown()
                gates_md = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("Scorecard"):
                scores_df = gr.Dataframe(wrap=True, label="Per-criterion scores")
                cat_df = gr.Dataframe(label="Category summary")
            with gr.Tab("Evidence"):
                evidence_df = gr.Dataframe(wrap=True, label="Gathered evidence (tiered)")
            with gr.Tab("Consistency"):
                consistency_md = gr.Markdown()
            with gr.Tab("Run metrics"):
                metrics_md = gr.Markdown()
        xlsx_file = gr.File(label="Populated scorecard (.xlsx)")

        run_btn.click(
            evaluate_vendor,
            inputs=[vendor, use_case, gate_security, gate_residency],
            outputs=[headline, gates_md, scores_df, cat_df, evidence_df,
                     consistency_md, metrics_md, xlsx_file],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
