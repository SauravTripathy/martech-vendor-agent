"""
Gradio front-end for the MarTech Vendor Evaluation Agent.

Portfolio-style layout: centered title, one outer card containing three input cards,
a primary Run Agent button, and a primary Download Excel button.

Run: python app.py (needs ANTHROPIC_API_KEY and OPENAI_API_KEY in the env)
"""

from __future__ import annotations

import os
import tempfile
import threading
import time

import gradio as gr

import config  # noqa: F401 - imported for existing project side effects/config loading
from documents import extract_text
from graph import run_evaluation
from scorecard import populate_template

TEMPLATE = os.getenv(
    "MARTECH_TEMPLATE",
    os.path.join(os.path.dirname(__file__), "MarTech_Vendor_Scorecard_Single.xlsx"),
)


# --------------------------------------------------------------------------- #
# Output rendering
# --------------------------------------------------------------------------- #

def _processing_md(elapsed: float) -> str:
    return f"⏳ **Processing… {elapsed:0.0f}s**"


def _metrics_md(state: dict, xlsx: str | None) -> str:
    m = state["metrics"]
    consistency = state.get("consistency")
    judge_line = ""
    if consistency is not None:
        judge_line = (
            f" Judge agreement within 1 point: {consistency.agreement_within_1:.0%}; "
            f"material judge findings: {consistency.material_issues_count}."
        )

    if xlsx:
        return f"✅ **Done in {m.total_seconds}s. Excel scorecard is ready.**{judge_line}"
    return f"⚠️ **Done in {m.total_seconds}s, but the Excel file could not be generated.**{judge_line}"


def _disabled_download(label: str = "Download Excel"):
    return gr.update(value=None, label=label, interactive=False)


def _enabled_download(path: str):
    return gr.update(value=path, label="Download Excel", interactive=True)


def _export(state: dict, vendor: str) -> str | None:
    if not os.path.exists(TEMPLATE):
        state.setdefault("errors", []).append("[xlsx] scorecard template not found")
        return None

    safe_vendor = vendor.strip().replace(" ", "_").replace("/", "-")
    out = os.path.join(tempfile.mkdtemp(), f"{safe_vendor}_scorecard.xlsx")

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

    paths: list[str] = []
    if files:
        for f in files:
            path = f if isinstance(f, str) else getattr(f, "name", None)
            if path:
                paths.append(path)

    holder: dict = {}

    def worker():
        try:
            docs = extract_text(paths)
            holder["state"] = run_evaluation(
                vendor.strip(),
                (context or "").strip(),
                documents=docs,
            )
        except Exception as exc:  # surface to the UI
            holder["error"] = exc

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    start = time.time()

    while t.is_alive():
        yield _processing_md(time.time() - start), _disabled_download("Preparing Excel…")
        time.sleep(1)

    t.join()

    if "error" in holder:
        raise gr.Error(f"Run failed: {holder['error']}")

    state = holder["state"]
    xlsx = _export(state, vendor)

    if xlsx:
        yield _metrics_md(state, xlsx), _enabled_download(xlsx)
    else:
        yield _metrics_md(state, xlsx), _disabled_download("Excel unavailable")


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
    button_primary_background_fill="#1d4ed8",
    button_primary_background_fill_hover="#1e40af",
    button_primary_text_color="#ffffff",
    block_background_fill="#ffffff",
    block_border_color="#dbeafe",
    block_label_text_color="#475569",
    block_title_text_color="#06142f",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@400..800&display=swap');

:root {
  --portfolio-bg: #f8fbff;
  --portfolio-text: #06142f;
  --portfolio-muted: #475569;
  --portfolio-blue: #1d4ed8;
  --portfolio-blue-dark: #1e40af;
  --portfolio-border: #dbeafe;
  --portfolio-card: #ffffff;
}

.gradio-container {
  background: var(--portfolio-bg) !important;
  color: var(--portfolio-text) !important;
  font-family: 'Inter', ui-sans-serif, system-ui, sans-serif !important;
}

main {
  max-width: 1120px !important;
  margin: 0 auto !important;
  padding: 52px 20px 72px !important;
}

.mt-hero {
  text-align: center;
  margin: 0 auto 30px;
}

.mt-title {
  font-family: 'Manrope', 'Inter', ui-sans-serif, system-ui, sans-serif !important;
  font-size: clamp(2.1rem, 5vw, 4.1rem);
  line-height: 1.02;
  letter-spacing: -0.055em;
  color: var(--portfolio-text);
  margin: 0;
}

.mt-shell {
  max-width: 1040px;
  margin: 0 auto;
  padding: 18px;
  border: 1px solid var(--portfolio-border);
  border-radius: 30px;
  background: rgba(255, 255, 255, 0.88);
  box-shadow: 0 24px 70px rgba(6, 20, 47, 0.09);
}

.mt-input-row {
  gap: 16px !important;
}

.mt-card {
  min-height: 214px;
  padding: 18px !important;
  border: 1px solid #e2e8f0;
  border-radius: 24px;
  background: var(--portfolio-card);
  box-shadow: 0 12px 34px rgba(6, 20, 47, 0.045);
}

.mt-card label,
.mt-card .block-title,
.mt-card .wrap > label {
  color: var(--portfolio-text) !important;
  font-weight: 700 !important;
}

.mt-card textarea,
.mt-card input,
.mt-card .file-preview,
.mt-card .upload-container {
  border-radius: 16px !important;
}

.mt-actions {
  max-width: 1040px;
  margin: 18px auto 0;
}

#run-agent-btn,
#download-excel-btn {
  width: 100% !important;
  border-radius: 18px !important;
  background: var(--portfolio-blue) !important;
  border: 1px solid var(--portfolio-blue) !important;
  color: #ffffff !important;
  font-weight: 800 !important;
  min-height: 56px !important;
  box-shadow: 0 14px 30px rgba(29, 78, 216, 0.22) !important;
}

#run-agent-btn:hover,
#download-excel-btn:hover {
  background: var(--portfolio-blue-dark) !important;
  border-color: var(--portfolio-blue-dark) !important;
}

#download-excel-btn:disabled,
#download-excel-btn[disabled] {
  background: var(--portfolio-blue) !important;
  border-color: var(--portfolio-blue) !important;
  color: #ffffff !important;
  opacity: 0.58 !important;
}

.mt-status {
  min-height: 28px;
  margin: 10px 0 22px;
  text-align: center;
  color: var(--portfolio-muted);
}

.mt-status p {
  margin: 0 !important;
}

.mt-download-gap {
  height: 10px;
}

@media (max-width: 820px) {
  main {
    padding-top: 34px !important;
  }

  .mt-shell {
    padding: 14px;
    border-radius: 24px;
  }

  .mt-card {
    min-height: auto;
  }
}
"""


def build_ui():
    with gr.Blocks(title="MarTech Vendor Evaluation Agent", theme=THEME, css=CSS) as demo:
        gr.HTML(
            "<section class='mt-hero'>"
            "<h1 class='mt-title'>MarTech Vendor Evaluation Agent</h1>"
            "</section>"
        )

        with gr.Group(elem_classes="mt-shell"):
            with gr.Row(equal_height=True, elem_classes="mt-input-row"):
                with gr.Column(scale=1, min_width=240, elem_classes="mt-card"):
                    vendor = gr.Textbox(
                        label="Vendor name",
                        placeholder="e.g. Braze",
                        lines=1,
                    )

                with gr.Column(scale=1, min_width=280, elem_classes="mt-card"):
                    context = gr.Textbox(
                        label="Context",
                        lines=6,
                        placeholder=(
                            "e.g. UK retail bank; real-time orchestration on Snowflake; "
                            "data must stay in the EU; SOC 2 required."
                        ),
                    )

                with gr.Column(scale=1, min_width=260, elem_classes="mt-card"):
                    files = gr.File(
                        label="Upload reports",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".csv"],
                        type="filepath",
                    )

        with gr.Column(elem_classes="mt-actions"):
            run_btn = gr.Button("Run Agent", variant="primary", elem_id="run-agent-btn")
            status_md = gr.Markdown("", elem_classes="mt-status")
            gr.HTML("<div class='mt-download-gap'></div>")
            download_btn = gr.DownloadButton(
                "Download Excel",
                value=None,
                variant="primary",
                elem_id="download-excel-btn",
                interactive=False,
            )

        run_btn.click(
            evaluate_vendor,
            inputs=[vendor, context, files],
            outputs=[status_md, download_btn],
        )

    return demo


if __name__ == "__main__":
    build_ui().queue().launch()
