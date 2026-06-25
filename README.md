---
title: MarTech Vendor Evaluation Agent
emoji:
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: "6.19.0"
app_file: app.py
pinned: false
short_description: LangGraph agent that evaluates MarTech vendors
---

# MarTech Vendor Evaluation Agent

A single-vendor deep-dive built with **LangGraph**, the **Anthropic API**, **OpenAI**, and **Gradio**.

The agent gathers public and uploaded-document evidence on a MarTech vendor, scores it against a weighted rubric, caps scores by evidence strength, and then runs a smarter cross-provider judge review. The judge re-scores the same evidence, challenges unsupported primary scores, flags missing evidence, and recommends score adjustments when the primary score is not defensible.

**To run this Space you must set two Space secrets:** `ANTHROPIC_API_KEY` for evidence gathering / primary scoring and `OPENAI_API_KEY` for the cross-provider judge. Optionally set `MARTECH_PRIMARY_MODEL` and `MARTECH_JUDGE_MODEL`.

> Initial assessment from publicly available and uploaded sources. Output is a decision aid, not an oracle. The consistency check is designed to show where the score is strong, weak, disputed, or under-supported.

Code repository: https://github.com/SauravTripathy/martech-vendor-agent
