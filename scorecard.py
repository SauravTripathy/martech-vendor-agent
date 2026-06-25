"""
Write agent results back into the single-vendor scorecard template, matching its
layout (category rows, criterion rows, weights, score col D, source col F).
"""
from __future__ import annotations

import os
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

import config

YELLOW = PatternFill("solid", start_color="FFFFF2CC", end_color="FFFFF2CC")


def populate_template(template_path: str, out_path: str, state: dict) -> str:
    wb = load_workbook(template_path)
    ws = wb["Scorecard"]

    # Vendor name lives just under the title band (row 4 in the template).
    for row in ws.iter_rows(min_row=1, max_row=6):
        for cell in row:
            if cell.value == "Vendor Name":
                cell.value = f"Vendor Name: {state['vendor_name']}"
                break

    scores = {s.criterion_id: s for s in state["scores"]}
    desc_to_id = {c.description: c.id for c in config.CRITERIA}

    # Walk rows; criterion rows have a description in column B.
    for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
        desc_cell = row[1]  # column B
        cid = desc_to_id.get(desc_cell.value)
        if not cid:
            continue
        s = scores.get(cid)
        if not s:
            continue
        score_cell = row[3]   # column D — Score (1-5)
        src_cell = row[5]     # column F — Source
        score_cell.value = s.score if s.score is not None else None  # blank, not 0
        note = []
        if s.capped:
            note.append("[tier-capped]")
        if s.is_gap:
            note.append("[no evidence]")
        src = "; ".join(s.sources[:2])
        src_cell.value = (" ".join(note) + " " + src).strip() or ("[no evidence]" if s.is_gap else "")

    # Headline note for gate elimination.
    if state.get("eliminated"):
        ws["A2"] = ("ELIMINATED — failed must-pass gate(s): "
                    + "; ".join(state.get("elimination_reasons", [])))
        ws["A2"].font = Font(bold=True, color="FF9C0006")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    wb.save(out_path)
    return out_path
