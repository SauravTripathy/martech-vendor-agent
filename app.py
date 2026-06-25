"""
Gradio front-end for the MarTech Vendor Evaluation Agent.

Minimal layout, themed to match saurav-tripathy.com (Inter / Manrope, #f8fbff /
#06142f, blue-700 accents). Inputs: vendor, context, uploaded files. Outputs:
run metrics with a live processing timer, and a downloadable Excel scorecard.

Run:  python app.py   (needs ANTHROPIC_API_KEY and OPENAI_API_KEY in the env)
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

import gradio as gr

import config
from graph import run_evaluation
from scorecard import populate_template
from documents import extract_text

TEMPLATE = os.getenv(
    "MARTECH_TEMPLATE",
    os.path.join(os.path.dirname(__file__), "MarTech_Vendor_Scorecard_Single.xlsx"),
)


# --------------------------------------------------------------------------- #
# Output rendering
# --------------------------------------------------------------------------- #
def _processing_md(elapsed: float) -> str:
    return (
        "### Run metrics\n"
        f"⏳ **Processing… {elapsed:0.0f}s**\n\n"
        "_Gathering evidence, scoring against the rubric, and cross-checking._"
    )


def _metrics_md(state: dict) -> str:
    m = state["metrics"]
    node_times = ", ".join(f"{k} {v}s" for k, v in m.node_seconds.items())
    prov_lines = "\n".join(
        f"  - `{prov}`: {d['in']:,} in / {d['out']:,} out ({d['calls']} calls)"
        for prov, d in m.by_provider.items()
    ) or "  - (none)"
    warn = ""
    if state.get("errors"):
        warn = (f"\n\n**Warnings ({len(state['errors'])}):** "
                + "; ".join(state["errors"][:3]))
    return (
        "### Run metrics\n"
        f"✅ **Done in {m.total_seconds}s**\n\n"
        f"- LLM calls: **{m.llm_calls}**  ·  web searches: **{m.web_search_requests}**\n"
        f"- Tokens (all providers): **{m.input_tokens:,}** in / **{m.output_tokens:,}** out\n"
        f"- By provider:\n{prov_lines}\n"
        f"- Node latency: {node_times}"
        f"{warn}"
    )


def _export(state: dict, vendor: str) -> str | None:
    if not os.path.exists(TEMPLATE):
        return None
    out = os.path.join(tempfile.mkdtemp(),
                       f"{vendor.strip().replace(' ', '_')}_scorecard.xlsx")
    try:
        return populate_template(TEMPLATE, out, state)
    except Exception as exc:
        state.setdefault("errors", []).append(f"[xlsx] {exc}")
        return None


# --------------------------------------------------------------------------- #
# Handler — generator so the timer ticks live while the run executes
# --------------------------------------------------------------------------- #
def evaluate_vendor(vendor: str, context: str, files):
    if not (vendor or "").strip():
        raise gr.Error("Please enter a vendor name.")

    paths = []
    if files:
        for f in files:
            paths.append(f if isinstance(f, str) else getattr(f, "name", None))

    holder: dict = {}

    def worker():
        try:
            docs = extract_text(paths)
            holder["state"] = run_evaluation(
                vendor.strip(), (context or "").strip(),
                gates_cfg=[],          # gates not exposed in this layout
                documents=docs,
            )
        except Exception as exc:  # surface to the UI
            holder["error"] = exc

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    start = time.time()
    while t.is_alive():
        yield _processing_md(time.time() - start), None
        time.sleep(1)
    t.join()

    if "error" in holder:
        raise gr.Error(f"Run failed: {holder['error']}")

    state = holder["state"]
    xlsx = _export(state, vendor)
    yield _metrics_md(state), xlsx


# --------------------------------------------------------------------------- #
# Theme + CSS to match saurav-tripathy.com
# --------------------------------------------------------------------------- #
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f8fbff",
    body_text_color="#06142f",
    button_primary_background_fill="#06142f",
    button_primary_background_fill_hover="#1d4ed8",
    button_primary_text_color="#ffffff",
    block_title_text_color="#06142f",
    block_label_text_color="#475569",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@400..800&display=swap');
.gradio-container { background: #f8fbff !important; }
h1, h2, h3, h4, .mt-title { font-family: 'Manrope','Inter',sans-serif !important;
  letter-spacing: -0.03em; color: #06142f; }
.mt-eyebrow { font-size: .8rem; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.2em; color: #1d4ed8; }
.mt-lede { color: #475569; font-size: 1.05rem; }
"""


def build_ui():
    with gr.Blocks(title="MarTech Vendor Evaluation Agent", theme=THEME, css=CSS) as demo:
        gr.HTML(
            "<p class='mt-eyebrow'>AI Projects</p>"
            "<h1 class='mt-title' style='font-size:2.4rem;margin:.3rem 0 .6rem;'>"
            "MarTech Vendor Evaluation Agent</h1>"
            "<p class='mt-lede'>Enter a vendor and your priorities, optionally attach "
            "reports or articles, and the agent gathers public evidence, scores it against "
            "a weighted rubric, and returns a downloadable evaluation scorecard.</p>"
        )
        with gr.Row():
            with gr.Column(scale=1):
                vendor = gr.Textbox(label="Vendor name", placeholder="e.g. Braze")
                context = gr.Textbox(
                    label="Context (priority requirements or other considerations)",
                    lines=4,
                    placeholder="e.g. UK retail bank; real-time orchestration on Snowflake; "
                                "data must stay in the EU; SOC 2 required.",
                )
                files = gr.File(
                    label="Upload files (reports, articles, etc.)",
                    file_count="multiple",
                    file_types=[".pdf", ".docx", ".txt", ".md", ".csv"],
                    type="filepath",
                )
                run_btn = gr.Button("Run evaluation", variant="primary")
            with gr.Column(scale=1):
                metrics_md = gr.Markdown("### Run metrics\n_Run an evaluation to see metrics._")
                xlsx_file = gr.File(label="Download evaluation (.xlsx)")

        run_btn.click(
            evaluate_vendor,
            inputs=[vendor, context, files],
            outputs=[metrics_md, xlsx_file],
        )
    return demo


if __name__ == "__main__":
    build_ui().queue().launch()
