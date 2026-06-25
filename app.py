"""
Gradio front-end for the MarTech Vendor Evaluation Agent.

Behavior:
- Prominent centered title.
- Simple three-input layout: vendor name, context, uploaded reports.
- No spinner text, no emojis.
- Shows only processing time while the agent runs.
- Disables Run Agent while running and styles it lighter blue.
- Shows run metrics only after completion.
- Shows Download Excel only after the Excel file is ready.

Run: python app.py
"""

from __future__ import annotations

import inspect
import os
import tempfile
import threading
import time
from typing import Any

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
    return f"**Processing time:** {elapsed:0.0f}s\n\n*Note: The agent can take around 10 minutes to generate the output*"


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a dataclass-like object or a dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or 0)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value or 0.0)
    except Exception:
        return default


def _format_seconds(value: Any) -> str:
    return f"{_safe_float(value):0.2f}s"


def _format_int(value: Any) -> str:
    return f"{_safe_int(value):,}"


def _format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:0.0f}%"
    except Exception:
        return "N/A"


def _metrics_md(state: dict, xlsx: str | None, elapsed: float) -> str:
    metrics = state.get("metrics") or {}

    total_seconds = _get_value(metrics, "total_seconds", elapsed) or elapsed
    input_tokens = _safe_int(_get_value(metrics, "input_tokens", 0))
    output_tokens = _safe_int(_get_value(metrics, "output_tokens", 0))
    total_tokens = input_tokens + output_tokens
    llm_calls = _safe_int(_get_value(metrics, "llm_calls", 0))
    web_searches = _safe_int(_get_value(metrics, "web_search_requests", 0))
    node_seconds = _get_value(metrics, "node_seconds", {}) or {}

    rows = [
        ("Time taken", _format_seconds(total_seconds)),
        ("Tokens consumed", _format_int(total_tokens)),
        ("Input tokens", _format_int(input_tokens)),
        ("Output tokens", _format_int(output_tokens)),
        ("LLM calls", _format_int(llm_calls)),
        ("Web searches", _format_int(web_searches)),
        ("Excel status", "Ready" if xlsx else "Not generated"),
    ]

    consistency = state.get("consistency")
    if consistency is not None:
        rows.extend(
            [
                (
                    "Judge agreement within 1 point",
                    _format_pct(_get_value(consistency, "agreement_within_1", None)),
                ),
                (
                    "Mean judge difference",
                    str(_get_value(consistency, "mean_abs_diff", "N/A")),
                ),
                (
                    "Material judge findings",
                    _format_int(_get_value(consistency, "material_issues_count", 0)),
                ),
            ]
        )

    lines = ["### Run metrics", "", "| Metric | Value |", "|---|---:|"]
    lines.extend(f"| {label} | {value} |" for label, value in rows)

    if isinstance(node_seconds, dict) and node_seconds:
        lines.extend(["", "### Step timing", "", "| Step | Time |", "|---|---:|"])
        for node, seconds in node_seconds.items():
            lines.append(f"| {node} | {_format_seconds(seconds)} |")

    errors = state.get("errors") or []
    if errors:
        lines.extend(["", "### Warnings", ""])
        for err in errors[:8]:
            lines.append(f"- {err}")

    return "\n".join(lines)


def _hide_metrics():
    return gr.update(value="", visible=False)


def _show_metrics(markdown: str):
    return gr.update(value=markdown, visible=True)


def _hide_download():
    return gr.update(value=None, visible=False, interactive=False)


def _show_download(path: str):
    return gr.update(
        value=path,
        label="Download Excel",
        visible=True,
        interactive=True,
    )


def _disable_run_button():
    return gr.update(value="Run Agent", interactive=False)


def _enable_run_button():
    return gr.update(value="Run Agent", interactive=True)


def _export(state: dict, vendor: str) -> str | None:
    if not os.path.exists(TEMPLATE):
        state.setdefault("errors", []).append("[xlsx] scorecard template not found")
        return None

    safe_vendor = (
        vendor.strip().replace(" ", "_").replace("/", "-").replace("\\", "-")
        or "vendor"
    )
    out = os.path.join(tempfile.mkdtemp(), f"{safe_vendor}_scorecard.xlsx")

    try:
        generated = populate_template(TEMPLATE, out, state)
    except Exception as exc:
        state.setdefault("errors", []).append(f"[xlsx] {exc}")
        return None

    if generated and os.path.exists(generated):
        return generated

    if os.path.exists(out):
        return out

    state.setdefault("errors", []).append(
        "[xlsx] scorecard export did not return a file path"
    )
    return None


# --------------------------------------------------------------------------- #
# Handler — generator so processing time updates while the run executes
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

    holder: dict[str, Any] = {}

    def worker() -> None:
        try:
            docs = extract_text(paths)
            kwargs = {"documents": docs}

            # Supports both the older graph signature with gates_cfg and the newer
            # no-gates signature without breaking either version.
            if "gates_cfg" in inspect.signature(run_evaluation).parameters:
                kwargs["gates_cfg"] = []

            holder["state"] = run_evaluation(
                vendor.strip(),
                (context or "").strip(),
                **kwargs,
            )
        except Exception as exc:
            holder["error"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    start = time.time()
    thread.start()

    yield (
        _processing_md(0),
        _hide_metrics(),
        _hide_download(),
        _disable_run_button(),
    )

    while thread.is_alive():
        elapsed = time.time() - start
        yield (
            _processing_md(elapsed),
            _hide_metrics(),
            _hide_download(),
            _disable_run_button(),
        )
        time.sleep(1)

    thread.join()
    elapsed = time.time() - start

    if "error" in holder:
        yield (
            f"**Processing time:** {elapsed:0.0f}s\n\nRun failed: {holder['error']}",
            _hide_metrics(),
            _hide_download(),
            _enable_run_button(),
        )
        return

    state = holder.get("state")
    if not state:
        yield (
            f"**Processing time:** {elapsed:0.0f}s\n\nRun failed: no result returned by the agent.",
            _hide_metrics(),
            _hide_download(),
            _enable_run_button(),
        )
        return

    xlsx = _export(state, vendor)
    metrics = _metrics_md(state, xlsx, elapsed)

    yield (
        _processing_md(elapsed),
        _show_metrics(metrics),
        _show_download(xlsx) if xlsx else _hide_download(),
        _enable_run_button(),
    )


# --------------------------------------------------------------------------- #
# Minimal CSS: mostly default Gradio with prominent title and visible metrics
# --------------------------------------------------------------------------- #

CSS = """
#main-title {
  text-align: center !important;
  margin: 12px auto 30px !important;
}

#main-title h1 {
  color: #ffffff !important;
  font-size: clamp(2.0rem, 4.5vw, 3.5rem) !important; /
  line-height: 1.02 !important;
  letter-spacing: -0.05em !important;
  font-weight: 900 !important;
  margin: 0 !important;
}

.mt-shell,
.mt-card {
  border: 0 !important;
  box-shadow: none !important;
  padding: 0 !important;
  background: transparent !important;
}

.mt-input-row {
  gap: 12px !important;
}

.mt-actions {
  margin-top: 18px !important;
}

#run-agent-btn,
#run-agent-btn button {
  width: 100% !important;
  min-height: 54px !important;
  border-radius: 12px !important;
  background: #2563eb !important;
  border-color: #2563eb !important;
  color: #ffffff !important;
  font-weight: 800 !important;
}

#run-agent-btn:hover,
#run-agent-btn button:hover {
  background: #1d4ed8 !important;
  border-color: #1d4ed8 !important;
}

#run-agent-btn:disabled,
#run-agent-btn[disabled],
#run-agent-btn button:disabled,
#run-agent-btn button[disabled] {
  background: #93c5fd !important;
  border-color: #93c5fd !important;
  color: #ffffff !important;
  opacity: 1 !important;
  cursor: not-allowed !important;
}

.mt-status {
  margin: 12px 0 10px !important;
  min-height: 26px !important;
  text-align: center !important;
}

.mt-status p {
  margin: 0 !important;
}

.mt-metrics {
  margin-top: 10px !important;
  background: #050b18 !important;
  color: #f8fafc !important;
  border: 0 !important;
  border-radius: 12px !important;
  padding: 14px 16px !important;
}

.mt-metrics * {
  color: #f8fafc !important;
}

.mt-metrics table {
  width: 100% !important;
}

.mt-metrics th,
.mt-metrics td {
  border-color: rgba(248, 250, 252, 0.25) !important;
}

#download-excel-btn,
#download-excel-btn button {
  width: 100% !important;
  min-height: 54px !important;
  margin-top: 14px !important;
  border-radius: 12px !important;
  background: #16a34a !important;
  border-color: #16a34a !important;
  color: #ffffff !important;
  font-weight: 800 !important;
}

#download-excel-btn:hover,
#download-excel-btn button:hover {
  background: #15803d !important;
  border-color: #15803d !important;
}

/* Make input labels inside mt-card larger and bold */
.mt-card label span,
.mt-card label p,
.mt-card [data-testid="block-info"] {
  font-size: 1.5rem !important;
  font-weight: 800 !important;
  color: #ffffff !important; 
}
"""


def build_ui():
    with gr.Blocks(title="MarTech Vendor Evaluation Agent", css=CSS) as demo:
        gr.HTML("<div id='main-title'><h1>MarTech Vendor Evaluation Agent</h1></div>")

        with gr.Group(elem_classes="mt-shell"):
            with gr.Row(equal_height=True, elem_classes="mt-input-row"):
                with gr.Column(scale=1, min_width=240, elem_classes="mt-card"):
                    vendor = gr.Textbox(
                        label="Enter Martech vendor name",
                        placeholder="e.g. Braze",
                        lines=6,
                    )

                with gr.Column(scale=1, min_width=280, elem_classes="mt-card"):
                    context = gr.Textbox(
                        label="Enter any additional context",
                        lines=6,
                        placeholder=("e.g. data must stay in the EU, SOC 2 required"),
                    )

                with gr.Column(scale=1, min_width=260, elem_classes="mt-card"):
                    files = gr.File(
                        label="Upload any existing reports on the MarTech vendor",
                        file_count="multiple",
                        file_types=[".pdf", ".docx", ".txt", ".md", ".csv"],
                        type="filepath",
                    )

        # Updated Actions Section: Centered, narrowed, and reordered
        with gr.Column(elem_classes="mt-actions"):
            with gr.Row():
                # Left spacer
                gr.Column(scale=1, min_width=0)

                # Center column holding the actions
                with gr.Column(scale=2, min_width=300):
                    run_btn = gr.Button(
                        "Run Agent", variant="primary", elem_id="run-agent-btn"
                    )
                    status_md = gr.Markdown("", elem_classes="mt-status")

                    # Moved DownloadButton above metrics_md
                    download_btn = gr.DownloadButton(
                        "Download Excel",
                        value=None,
                        elem_id="download-excel-btn",
                        visible=False,
                        interactive=False,
                    )

                    metrics_md = gr.Markdown(
                        "", elem_classes="mt-metrics", visible=False
                    )

                # Right spacer
                gr.Column(scale=1, min_width=0)

        # The outputs array order remains unchanged to match the generator yields
        run_btn.click(
            evaluate_vendor,
            inputs=[vendor, context, files],
            outputs=[status_md, metrics_md, download_btn, run_btn],
            show_progress="hidden",
        )

    return demo


if __name__ == "__main__":
    build_ui().queue().launch()
