"""Render selected PDF pages to PNG for local visual review."""

from __future__ import annotations

import os

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
VENDOR_DIR = PROJECT_ROOT / "codexAnalysis" / "pdf_review" / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

import pypdfium2 as pdfium  # noqa: E402


def page_range(page_count: int, *, last: int | None, start: int | None, end: int | None) -> list[int]:
    if last is not None:
        first = max(page_count - last, 0)
        return list(range(first, page_count))
    if start is None:
        start = 1
    if end is None:
        end = start
    start_idx = max(start - 1, 0)
    end_idx = min(end, page_count)
    return list(range(start_idx, end_idx))


def render_pages(pdf_path: Path, out_dir: Path, pages: list[int], scale: float) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    paths: list[Path] = []
    for idx in pages:
        page = doc[idx]
        bitmap = page.render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        out_path = out_dir / f"{pdf_path.stem}_page_{idx + 1:03d}.png"
        image.save(out_path)
        paths.append(out_path)
    return paths


def make_contact_sheet(paths: list[Path], out_path: Path, thumb_width: int = 360) -> None:
    thumbs: list[Image.Image] = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        ratio = thumb_width / img.width
        thumb = img.resize((thumb_width, int(img.height * ratio)), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb.width, thumb.height + 34), "white")
        canvas.paste(thumb, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb.height + 8), path.stem, fill=(30, 40, 50))
        thumbs.append(canvas)
    if not thumbs:
        return
    cols = min(3, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
    for i, thumb in enumerate(thumbs):
        x = (i % cols) * cell_w
        y = (i // cols) * cell_h
        sheet.paste(thumb, (x, y))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "codexAnalysis" / "pdf_review" / "renders")
    parser.add_argument("--last", type=int)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--scale", type=float, default=1.8)
    args = parser.parse_args()

    pdf_path = args.pdf
    doc = pdfium.PdfDocument(str(pdf_path))
    count = len(doc)
    pages = page_range(count, last=args.last, start=args.start, end=args.end)
    stem_dir = args.out_dir / pdf_path.stem
    paths = render_pages(pdf_path, stem_dir, pages, args.scale)
    contact = stem_dir / f"{pdf_path.stem}_contact_sheet.png"
    make_contact_sheet(paths, contact)
    print(f"{pdf_path.name}: {count} pages")
    print(f"Rendered pages: {', '.join(str(p + 1) for p in pages)}")
    print(f"Contact sheet: {contact}")


if __name__ == "__main__":
    main()

