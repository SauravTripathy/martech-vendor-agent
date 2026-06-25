"""
Gradio front-end for the MarTech Vendor Evaluation Agent.

Themed to match saurav-tripathy.com (Inter / Manrope, #f8fbff / #06142f,
blue-700 accent). Inputs: vendor, context, uploaded files. Outputs: a live
processing timer + run metrics, and a downloadable Excel scorecard that turns
green once populated.

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
        f"⏳ **Processing… {elapsed:0.0f}s** — gathering evidence, scoring, "
        f"and cross-checking."
    )


def _metrics_md(state: dict) -> str:
    m = state["metrics"]
    prov_lines = (
        "  ·  ".join(
            f"`{prov}` {d['in']:,}/{d['out']:,} ({d['calls']})"
            for prov, d in m.by_provider.items()
        )
        or "—"
    )
    warn = ""
    if state.get("errors"):
        warn = f"\n\n**Warnings ({len(state['errors'])}):** " + "; ".join(
            state["errors"][:3]
        )
    return (
        f"### Run metrics\n"
        f"✅ **Done in {m.total_seconds}s**  ·  {m.llm_calls} LLM calls  ·  "
        f"{m.web_search_requests} web searches\n\n"
        f"Tokens (in/out, calls) by provider:  {prov_lines}"
        f"{warn}"
    )


def _export(state: dict, vendor: str) -> str | None:
    if not os.path.exists(TEMPLATE):
        return None
    out = os.path.join(
        tempfile.mkdtemp(), f"{vendor.strip().replace(' ', '_')}_scorecard.xlsx"
    )
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
                vendor.strip(),
                (context or "").strip(),
                gates_cfg=[],
                documents=docs,
            )
        except Exception as exc:
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
# Theme + CSS — matches saurav-tripathy.com
# --------------------------------------------------------------------------- #
THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f8fbff",
    body_text_color="#06142f",
    block_title_text_color="#06142f",
    block_label_text_color="#475569",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@400..800&display=swap');

.gradio-container { background:#f8fbff !important; max-width:780px !important; margin:0 auto !important; }

/* 1) Title: bold, large, centered */
#mt-title { text-align:center; font-family:'Manrope','Inter',sans-serif; font-weight:800;
  font-size:2.3rem; letter-spacing:-0.03em; color:#06142f; margin:.4rem 0 .2rem; }
#mt-sub { text-align:center; color:#475569; font-size:1rem; margin:0 0 1.2rem; }

/* 2) White panels around the input boxes — THIN (1px border, small padding) */
.mt-card { background:#ffffff !important;
    box-shadow:0 1px 2px rgba(6,20,47,.04) !important; }

/* 3) Upload box — shorter dropzone */
#mt-upload { min-height:0 !important; }
#mt-upload .wrap, #mt-upload .file-upload, #mt-upload [data-testid="upload-button"],
#mt-upload .center { min-height:96px !important;}

/* Run button: portfolio navy with blue-700 hover */
#mt-run button { background:#06142f !important; color:#fff !important; border:none !important;
  font-weight:700 !important; border-radius:12px !important; }
#mt-run button:hover { background:#1d4ed8 !important; }

/* Tighten vertical gaps so the download sits high */
#mt-metrics { margin:.4rem 0 !important; }

/* 4) Download box: white initially, green once a file is present */
#mt-dl { background:#ffffff; border:1px solid #cbd5e1;
  transition:background .3s ease,border-color .3s ease; }
#mt-dl:has(a[download]), #mt-dl:has(a[href*="file="]), #mt-dl:has(.file-preview) {
  background:#dcfce7; border-color:#16a34a; }

  /* Remove dashed borders from upload and download boxes */
#mt-upload .border-dashed, #mt-upload [class*="dashed"],
#mt-dl .border-dashed, #mt-dl [class*="dashed"] {
    border-style: solid !important; 
    border-color: transparent !important;
    border-width: 0px !important;
}
"""


def build_ui():
    with gr.Blocks(
        title="MarTech Vendor Evaluation Agent", theme=THEME, css=CSS
    ) as demo:
        gr.HTML(
            "<div id='mt-title'>MarTech Vendor Evaluation Agent</div>"
            "<div id='mt-sub'>Enter a vendor and your priorities, attach any reports, "
            "and download a scored evaluation.</div>"
        )

        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes="mt-card"):
                    vendor = gr.Textbox(
                        label="Vendor name", lines=4, placeholder="e.g. Braze"
                    )
            with gr.Column():
                with gr.Group(elem_classes="mt-card"):
                    context = gr.Textbox(
                        label="Additional Context (e.g., key requirements)",
                        lines=4,
                        placeholder="e.g. data must stay in EU, SOC 2 required.",
                    )

        with gr.Group(elem_classes="mt-card"):
            files = gr.File(
                label="Upload files (reports, articles, etc.)",
                file_count="multiple",
                file_types=[".pdf", ".docx", ".txt", ".md", ".csv"],
                type="filepath",
                elem_id="mt-upload",
            )

        run_btn = gr.Button("Run Agent", variant="primary", elem_id="mt-run")

        with gr.Group(elem_id="mt-dl"):
            xlsx_file = gr.File(label="Download Excel")

        metrics_md = gr.Markdown("Run the agent to see metrics.", elem_id="mt-metrics")

        run_btn.click(
            evaluate_vendor,
            inputs=[vendor, context, files],
            outputs=[metrics_md, xlsx_file],
        )
    return demo


if __name__ == "__main__":
    build_ui().queue().launch()
