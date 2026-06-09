"""Scan manuscript text for claim-boundary terms that require review.

This script is intentionally dependency-free so it can scan both Markdown/text
files and DOCX files using only the Python standard library.
"""

from __future__ import annotations

import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


TERMS = [
    ("tensor", "review: current 2D implementation is scalar, not tensor-valued"),
    ("anisotropic fluid", "review: avoid fluid-dynamic overclaiming"),
    ("fluid-dynamic", "review: avoid broad physical-fluid claim"),
    ("clinical", "review: avoid clinical-grade or clinical-utility claims"),
    ("LLM", "review: LLM/UI should not be central evidence"),
    ("false-positive", "review: avoid promotional false-positive language"),
    ("crusher", "review: remove promotional phrasing"),
    ("guarantee", "review: avoid guarantee/convergence overclaim"),
    ("universal", "review: avoid universal superiority claim"),
    ("superior", "review: require exact benchmark context"),
    ("3D physical", "review: avoid unsupported 3D tensor framing"),
    ("nuclei density", "review: hematoxylin is proxy unless segmentation exists"),
    ("div(", "review: implemented PDE is not divergence form"),
    ("divergence", "review: do not describe implemented PDE as divergence-form"),
]


def read_docx(path: Path) -> str:
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml")
    root = ET.fromstring(xml)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras = []
    for para in root.findall(".//w:p", ns):
        text = "".join(node.text or "" for node in para.findall(".//w:t", ns))
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            paras.append(text)
    return "\n".join(paras)


def read_text(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return read_docx(path)
    return path.read_text(encoding="utf-8", errors="replace")


def iter_hits(path: Path, text: str):
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        for term, note in TERMS:
            if re.search(re.escape(term), line, flags=re.IGNORECASE):
                yield {
                    "file": str(path),
                    "line": lineno,
                    "term": term,
                    "note": note,
                    "context": line[:500],
                }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("inputs", nargs="+", type=Path)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        if not path.exists():
            rows.append(
                {
                    "file": str(path),
                    "line": "",
                    "term": "MISSING_FILE",
                    "note": "input path does not exist",
                    "context": "",
                }
            )
            continue
        text = read_text(path)
        rows.extend(iter_hits(path, text))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "line", "term", "note", "context"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} QC hits to {args.output}")


if __name__ == "__main__":
    main()
