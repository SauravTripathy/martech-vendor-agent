"""Extract text from user-uploaded reports/articles to feed the agent as evidence."""
from __future__ import annotations

import os

import config


def _read_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:
        return f"[could not read PDF {os.path.basename(path)}: {exc}]"


def _read_docx(path: str) -> str:
    try:
        import docx
        d = docx.Document(path)
        return "\n".join(p.text for p in d.paragraphs)
    except Exception as exc:
        return f"[could not read DOCX {os.path.basename(path)}: {exc}]"


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except Exception as exc:
        return f"[could not read {os.path.basename(path)}: {exc}]"


def extract_text(paths) -> str:
    """Concatenate extracted text from a list of file paths, capped at a budget."""
    if not paths:
        return ""
    chunks: list[str] = []
    for p in paths:
        if not p:
            continue
        name = os.path.basename(p)
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            body = _read_pdf(p)
        elif ext in (".docx",):
            body = _read_docx(p)
        elif ext in (".txt", ".md", ".csv"):
            body = _read_text(p)
        else:
            body = f"[unsupported file type skipped: {name}]"
        chunks.append(f"### Document: {name}\n{body.strip()}")

    combined = "\n\n".join(chunks)
    budget = config.DOC_CHAR_BUDGET
    if len(combined) > budget:
        combined = combined[:budget] + "\n[...truncated...]"
    return combined
