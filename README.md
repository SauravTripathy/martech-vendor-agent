---
title: MarTech Vendor Evaluation Agent
emoji: 🧮
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: false
short_description: LangGraph agent that evaluates MarTech vendors
---

# MarTech Vendor Evaluation Agent

A single-vendor deep-dive built with **LangGraph**, the **Anthropic API**, and
**Gradio**. It gathers public evidence on a MarTech vendor, scores it against a
weighted rubric (with scores capped by evidence strength), runs must-pass gates,
and cross-checks the scoring with a second model.

**To run this Space you must set two Space secrets:** `ANTHROPIC_API_KEY`
(evaluation / web search) and `OPENAI_API_KEY` (cross-provider judge).
Optionally set `MARTECH_PRIMARY_MODEL` (an Anthropic model that supports web
search) and `MARTECH_JUDGE_MODEL` (an OpenAI model).

> Initial assessment from publicly available sources. Output is a decision aid,
> not an oracle — must-pass gates and category subtotals are designed to keep a
> high overall score from hiding a disqualifying gap.

Code repository: https://github.com/SauravTripathy/martech-vendor-agent
