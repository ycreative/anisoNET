"""Create a working supplementary-figure folder from existing assets.

This script only copies/tiles already-generated PNG assets. It does not rerun
model fitting or recompute metrics.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("ANISONET_ANALYSIS_ROOT", PROJECT_ROOT / "codexAnalysis"))
OUT_DIR = ROOT / "manuscript_figures" / "supplementary_working"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf" if bold else "C:/Windows/Fonts/calibri.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


TITLE_FONT = font(44, bold=True)
SUBTITLE_FONT = font(26, bold=False)
LABEL_FONT = font(24, bold=True)
SMALL_FONT = font(18, bold=False)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def glob_one(pattern: str) -> Path | None:
    matches = sorted(PROJECT_ROOT.glob(pattern))
    return matches[0] if matches else None


def glob_many(pattern: str, limit: int | None = None) -> list[Path]:
    matches = sorted(PROJECT_ROOT.glob(pattern))
    return matches[:limit] if limit else matches


def fit_image(path: Path, max_w: int, max_h: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img = trim_whitespace(img)
    img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (max_w, max_h), "white")
    x = (max_w - img.width) // 2
    y = (max_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def trim_whitespace(img: Image.Image, margin: int = 18) -> Image.Image:
    bg = Image.new("RGB", img.size, "white")
    diff = ImageChops.difference(img, bg).convert("L")
    bbox = diff.point(lambda p: 255 if p > 8 else 0).getbbox()
    if not bbox:
        return img
    left, top, right, bottom = bbox
    left = max(0, left - margin)
    top = max(0, top - margin)
    right = min(img.width, right + margin)
    bottom = min(img.height, bottom + margin)
    if right - left < img.width * 0.15 or bottom - top < img.height * 0.15:
        return img
    return img.crop((left, top, right, bottom))


def wrap_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > max_chars and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def tile_page(
    stem: str,
    title: str,
    subtitle: str,
    items: list[tuple[str, Path]],
    *,
    cols: int = 2,
    tile_w: int = 1280,
    tile_h: int = 760,
) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = max(1, (len(items) + cols - 1) // cols)
    margin = 70
    gap = 42
    header_h = 145
    label_h = 46
    page_w = margin * 2 + cols * tile_w + (cols - 1) * gap
    page_h = margin * 2 + header_h + rows * (label_h + tile_h) + (rows - 1) * gap
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    draw.text((margin, margin - 10), title, fill="#111111", font=TITLE_FONT)
    draw.text((margin, margin + 52), subtitle, fill="#39424e", font=SUBTITLE_FONT)

    for i, (label, path) in enumerate(items):
        row, col = divmod(i, cols)
        x = margin + col * (tile_w + gap)
        y = margin + header_h + row * (tile_h + label_h + gap)
        draw.rounded_rectangle((x, y, x + tile_w, y + label_h + tile_h), radius=6, outline="#d8d8d8", width=2, fill="#fbfbfb")
        draw.text((x + 18, y + 10), label, fill="#222222", font=LABEL_FONT)
        img = fit_image(path, tile_w - 24, tile_h - 20)
        page.paste(img, (x + 12, y + label_h + 10))

    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    page.save(png_path, dpi=(220, 220))
    page.save(pdf_path, "PDF", resolution=220.0)
    return png_path


def build_overview(figures: list[tuple[str, str, Path]]) -> Path:
    cols = 3
    tile_w = 900
    tile_h = 620
    label_h = 92
    margin = 58
    gap = 34
    header_h = 120
    rows = (len(figures) + cols - 1) // cols
    page_w = margin * 2 + cols * tile_w + (cols - 1) * gap
    page_h = margin * 2 + header_h + rows * (tile_h + label_h) + (rows - 1) * gap
    page = Image.new("RGB", (page_w, page_h), "white")
    draw = ImageDraw.Draw(page)
    draw.text((margin, margin - 8), "Working Supplementary Figure Overview", fill="#111111", font=TITLE_FONT)
    draw.text((margin, margin + 52), "Generated from existing assets; numbering and inclusion are still provisional.", fill="#39424e", font=SUBTITLE_FONT)
    for i, (code, title, path) in enumerate(figures):
        row, col = divmod(i, cols)
        x = margin + col * (tile_w + gap)
        y = margin + header_h + row * (tile_h + label_h + gap)
        draw.rounded_rectangle((x, y, x + tile_w, y + tile_h + label_h), radius=6, outline="#d8d8d8", width=2, fill="#fbfbfb")
        draw.text((x + 16, y + 12), code, fill="#111111", font=LABEL_FONT)
        for line_i, line in enumerate(wrap_text(title, 52)[:2]):
            draw.text((x + 16, y + 42 + line_i * 24), line, fill="#39424e", font=SMALL_FONT)
        img = fit_image(path, tile_w - 22, tile_h - 16)
        page.paste(img, (x + 11, y + label_h + 8))
    png_path = OUT_DIR / "Supplementary_Figure_Working_Overview.png"
    pdf_path = OUT_DIR / "Supplementary_Figure_Working_Overview.pdf"
    page.save(png_path, dpi=(220, 220))
    page.save(pdf_path, "PDF", resolution=220.0)
    return png_path


def main() -> None:
    brain_samples = [
        "GSM5773453_Young_mouse_brain_A1-1",
        "GSM5773454_Young_mouse_brain_B1-1",
        "GSM5773455_Young_mouse_brain_C1-1",
        "GSM5773456_Young_mouse_brain_D1-1",
        "GSM5773457_Old_mouse_brain_A1-2",
        "GSM5773458_Old_mouse_brain_B1-2",
        "GSM5773459_Old_mouse_brain_C1-2",
        "GSM5773460_Old_mouse_brain_D1-2",
    ]

    figure_specs: list[dict[str, object]] = [
        {
            "code": "FigS01",
            "stem": "FigS01_primary_brain_preflight_qc_working",
            "title": "Supplementary Figure S1. Primary brain preprocessing QC",
            "subtitle": "Apoe/CNS-myelin preflight fields across the eight GSE193107 sections.",
            "cols": 2,
            "items": [
                (sample.replace("GSM577", "G"), ROOT / "preflight" / "brain_aging_gse193107" / sample / "Apoe_CNS_Myelin" / "preflight_fields.png")
                for sample in brain_samples
            ],
        },
        {
            "code": "FigS02",
            "stem": "FigS02_primary_brain_full_fields_working",
            "title": "Supplementary Figure S2. Full Apoe and Gfap primary fields",
            "subtitle": "Complete eight-section spatial field montages supporting the primary brain application.",
            "cols": 1,
            "items": [
                ("Apoe eight-section field montage", ROOT / "manuscript_figures" / "Figure2_gse193107_primary_application" / "Fig2C_Apoe_8section_fields.png"),
                ("Gfap eight-section field montage", ROOT / "manuscript_figures" / "Figure2_gse193107_primary_application" / "Fig2D_Gfap_8section_fields.png"),
            ],
        },
        {
            "code": "FigS03",
            "stem": "FigS03_heldout_benchmark_details_working",
            "title": "Supplementary Figure S3. Held-out benchmark details",
            "subtitle": "Conventional interpolation endpoint; keep separate from fitted-source QC claims.",
            "cols": 1,
            "items": [
                ("Held-out test Pearson summary", ROOT / "heldout" / "brain_aging_gse193107" / "summary" / "heldout_test_pearson_summary.png"),
                ("Current main benchmark panel for context", ROOT / "manuscript_figures" / "Figure3_benchmark_and_synthetic_validation" / "Fig3A_generic_heldout_benchmark.png"),
            ],
        },
        {
            "code": "FigS04",
            "stem": "FigS04_synthetic_replicates_working",
            "title": "Supplementary Figure S4. Synthetic barrier benchmark replicates",
            "subtitle": "Seed-level synthetic fields and aggregate synthetic-barrier metrics.",
            "cols": 2,
            "items": [
                ("Aggregate synthetic summary", ROOT / "synthetic_barrier" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin" / "synthetic_barrier_summary.png"),
                ("Seed 0, balanced split", ROOT / "synthetic_barrier" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin" / "seed0" / "synthetic_barrier_fields.png"),
                ("Seed 0, train20/test80", ROOT / "synthetic_barrier" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin" / "seed0_train20_test80" / "synthetic_barrier_fields.png"),
                ("Seed 1, train20/test80", ROOT / "synthetic_barrier" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin" / "seed1_train20_test80" / "synthetic_barrier_fields.png"),
            ],
        },
        {
            "code": "FigS05",
            "stem": "FigS05_histology_prior_sensitivity_working",
            "title": "Supplementary Figure S5. Histology-prior sensitivity",
            "subtitle": "Brightness versus hematoxylin structural-prior checks.",
            "cols": 1,
            "items": [
                ("Preflight-level histology-prior comparison", ROOT / "histology_prior" / "brain_aging_gse193107" / "histology_prior_preflight_summary.png"),
                ("PINN-level histology-prior comparison", ROOT / "histology_prior" / "brain_aging_gse193107" / "histology_prior_pinn_summary.png"),
            ],
        },
        {
            "code": "FigS06",
            "stem": "FigS06_profile_and_loss_sensitivity_working",
            "title": "Supplementary Figure S6. Profile and loss-weight sensitivity",
            "subtitle": "Low-PDE all-section sensitivity plus representative loss-weight probe.",
            "cols": 1,
            "items": [
                ("Low-PDE profile validation across sections", ROOT / "loss_weight_sensitivity" / "brain_aging_gse193107" / "multi_section_low_pde" / "low_pde_profile_validation_summary.png"),
                ("Representative loss-weight sensitivity", ROOT / "loss_weight_sensitivity" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "loss_weight_sensitivity_summary.png"),
            ],
        },
        {
            "code": "FigS07",
            "stem": "FigS07_parameter_stability_working",
            "title": "Supplementary Figure S7. Parameter and stochastic stability",
            "subtitle": "Source clipping, alpha sensitivity, and seed-level low-PDE stability.",
            "cols": 1,
            "items": [
                ("Source-clipping sensitivity", ROOT / "source_clipping_sensitivity" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "source_clipping_sensitivity_summary.png"),
                ("Alpha sensitivity", ROOT / "alpha_sensitivity" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "alpha_sensitivity_summary.png"),
                ("Low-PDE seed stability", ROOT / "loss_weight_sensitivity" / "brain_aging_gse193107" / "low_pde_seed_stability" / "low_pde_seed_stability_summary.png"),
            ],
        },
        {
            "code": "FigS08",
            "stem": "FigS08_runtime_memory_convergence_working",
            "title": "Supplementary Figure S8. Runtime, memory, and convergence",
            "subtitle": "Computational reproducibility evidence; not an active main Figure 4G claim.",
            "cols": 2,
            "items": [
                ("Convergence summary", ROOT / "convergence_diagnostics" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "pinn_convergence_summary.png"),
                ("Representative convergence traces", ROOT / "convergence_diagnostics" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "pinn_convergence_representative.png"),
                ("Default resource profile", ROOT / "resource_profile" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "anisonet_resource_profile_summary.png"),
                ("Low-PDE resource profile", ROOT / "resource_profile" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "anisonet_resource_profile_summary_low_pde_candidate.png"),
            ],
        },
        {
            "code": "FigS09",
            "stem": "FigS09_kidney_evidence_boundary_working",
            "title": "Supplementary Figure S9. Kidney evidence-boundary analysis",
            "subtitle": "Resistance-aware baselines and prior-hybrid probes; conservative optimization evidence.",
            "cols": 2,
            "items": [
                ("Kidney barrier split summary", ROOT / "barrier_split_anisonet" / "mouse_kidney_10x" / "V1_Mouse_Kidney" / "summary_with_prior" / "generic_barrier_split_summary.png"),
                ("Grid-geodesic prior summary", ROOT / "barrier_split_anisonet" / "mouse_kidney_10x" / "V1_Mouse_Kidney" / "grid_geodesic_prior" / "grid_geodesic_prior_summary.png"),
                ("Prior-hybrid blend summary", ROOT / "barrier_split_anisonet" / "mouse_kidney_10x" / "V1_Mouse_Kidney" / "prior_hybrid_blend_analysis" / "kidney_prior_hybrid_blend_summary.png"),
                ("Umod profile probe", ROOT / "barrier_split_anisonet" / "mouse_kidney_10x" / "V1_Mouse_Kidney" / "Umod_profile_probe" / "kidney_barrier_split_profile_probe_summary.png"),
            ],
        },
        {
            "code": "FigS10",
            "stem": "FigS10_kidney_marker_and_targeted_extension_working",
            "title": "Supplementary Figure S10. Kidney marker screen and targeted extension",
            "subtitle": "Kidney marker context and selected targeted-extension outputs.",
            "cols": 2,
            "items": [
                ("Slc34a1 kidney preflight", ROOT / "cross_tissue" / "mouse_kidney_10x" / "V1_Mouse_Kidney" / "preflight" / "Slc34a1_TAL_CD_barrier" / "preflight_fields.png"),
                ("Umod kidney preflight", ROOT / "cross_tissue" / "mouse_kidney_10x" / "V1_Mouse_Kidney" / "preflight" / "Umod_proximal_barrier" / "preflight_fields.png"),
                ("Kidney targeted field contact sheet", ROOT / "targeted_gene_extension" / "field_contact_sheets" / "mouse_kidney_10x_full_field_contact_sheet.png"),
                ("Kidney marker extension panel candidate", ROOT / "manuscript_figures" / "Figure5_cross_tissue_boundary" / "Fig5G_kidney_marker_extension.png"),
            ],
        },
        {
            "code": "FigS11",
            "stem": "FigS11_sagittal_brain_negative_control_working",
            "title": "Supplementary Figure S11. Sagittal brain portability control",
            "subtitle": "Healthy sagittal brain preflight plus current negative-control context.",
            "cols": 2,
            "items": [
                ("Sagittal S1 Apoe preflight", ROOT / "cross_tissue" / "mouse_brain_sagittal_10x" / "V1_Mouse_Brain_Sagittal_Anterior_Section1" / "preflight" / "Apoe_CNS_Myelin" / "preflight_fields.png"),
                ("Sagittal S1 Gfap preflight", ROOT / "cross_tissue" / "mouse_brain_sagittal_10x" / "V1_Mouse_Brain_Sagittal_Anterior_Section1" / "preflight" / "Gfap_CNS_Myelin" / "preflight_fields.png"),
                ("Sagittal targeted field contact sheet", ROOT / "targeted_gene_extension" / "field_contact_sheets" / "mouse_brain_sagittal_10x_full_field_contact_sheet.png"),
                ("Sagittal negative-control main candidate", ROOT / "manuscript_figures" / "Figure5_cross_tissue_boundary" / "Fig5D_sagittal_spatial_negative_control_context.png"),
            ],
        },
        {
            "code": "FigS12",
            "stem": "FigS12_liver_apap_task_design_working",
            "title": "Supplementary Figure S12. Liver/APAP task-design analysis",
            "subtitle": "Preflight, annotation, and resistance-vs-Euclidean boundary checks.",
            "cols": 2,
            "items": [
                ("Liver/APAP preflight summary", ROOT / "cross_tissue" / "mouse_liver_apap_gse280515" / "summary" / "liver_apap_preflight_summary.png"),
                ("Central marker resistance-IDW check", ROOT / "barrier_aware_interpolation" / "mouse_liver_apap_gse280515" / "summary_cyp2e1_glul" / "liver_apap_cyp2e1_glul_resistance_vs_euclidean_idw_summary.png"),
                ("Annotation-boundary benchmark", ROOT / "annotation_boundary_benchmark" / "mouse_liver_apap_gse280515" / "summary" / "liver_annotation_boundary_resistance_vs_euclidean_idw_summary.png"),
                ("Liver targeted field contact sheet", ROOT / "targeted_gene_extension" / "field_contact_sheets" / "mouse_liver_apap_gse280515_full_field_contact_sheet.png"),
            ],
        },
        {
            "code": "FigS13",
            "stem": "FigS13_targeted_extension_contact_sheets_working",
            "title": "Supplementary Figure S13. Targeted multi-gene extension contact sheets",
            "subtitle": "Full-profile targeted fields across primary brain and cross-tissue exploratory datasets.",
            "cols": 2,
            "items": [
                ("Primary brain targeted fields", ROOT / "targeted_gene_extension" / "field_contact_sheets" / "brain_aging_gse193107_full_field_contact_sheet.png"),
                ("Sagittal brain targeted fields", ROOT / "targeted_gene_extension" / "field_contact_sheets" / "mouse_brain_sagittal_10x_full_field_contact_sheet.png"),
                ("Kidney targeted fields", ROOT / "targeted_gene_extension" / "field_contact_sheets" / "mouse_kidney_10x_full_field_contact_sheet.png"),
                ("Liver/APAP targeted fields", ROOT / "targeted_gene_extension" / "field_contact_sheets" / "mouse_liver_apap_gse280515_full_field_contact_sheet.png"),
            ],
        },
        {
            "code": "FigS14",
            "stem": "FigS14_marker_module_and_spot_diagnostics_working",
            "title": "Supplementary Figure S14. Marker-module and spot-level diagnostics",
            "subtitle": "Representative CNS-myelin leave-one-marker-out and barrier-edge spot diagnostics.",
            "cols": 2,
            "items": [
                ("Leave-one-marker-out summary", ROOT / "leave_one_marker_out" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "leave_one_marker_out_summary.png"),
                ("Leave-one-marker-out Pearson heatmap", ROOT / "leave_one_marker_out" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "leave_one_marker_out_pearson_delta_heatmap.png"),
                ("Apoe barrier-edge discordant spots", ROOT / "spot_diagnostics" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin" / "gaussian_vs_anisonet_default" / "barrier_edge_discordant_spots.png"),
                ("Gfap barrier-edge discordant spots", ROOT / "spot_diagnostics" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Gfap_CNS_Myelin" / "gaussian_vs_anisonet_default" / "barrier_edge_discordant_spots.png"),
            ],
        },
    ]

    source_rows: list[dict[str, str]] = []
    built: list[tuple[str, str, Path]] = []
    for spec in figure_specs:
        raw_items = spec["items"]  # type: ignore[index]
        items = [(label, path) for label, path in raw_items if Path(path).exists()]  # type: ignore[union-attr]
        for label, path in raw_items:  # type: ignore[union-attr]
            source_rows.append(
                {
                    "figure": str(spec["code"]),
                    "label": label,
                    "exists": str(Path(path).exists()),
                    "source_path": rel(Path(path)),
                }
            )
        if not items:
            continue
        png = tile_page(
            str(spec["stem"]),
            str(spec["title"]),
            str(spec["subtitle"]),
            items,
            cols=int(spec["cols"]),
            tile_w=1320 if int(spec["cols"]) == 2 else 1840,
            tile_h=780 if int(spec["cols"]) == 2 else 900,
        )
        built.append((str(spec["code"]), str(spec["title"]), png))

    overview = build_overview(built)

    with (OUT_DIR / "supplementary_working_source_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["figure", "label", "exists", "source_path"])
        writer.writeheader()
        writer.writerows(source_rows)

    lines = [
        "# Working Supplementary Figure Set",
        "",
        "Generated from existing assets only. Numbering and inclusion are provisional until final journal assembly.",
        "",
        f"- Overview: `{rel(overview)}`",
        "- Source map: `codexAnalysis/manuscript_figures/supplementary_working/supplementary_working_source_map.csv`",
        "",
        "## Figures",
        "",
    ]
    for code, title, png in built:
        lines.append(f"- `{code}`: `{rel(png)}`")
        lines.append(f"  - `{rel(png.with_suffix('.pdf'))}`")
    (OUT_DIR / "CURRENT_SUPPLEMENTARY_FIGURE_SET.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {len(built)} working supplementary figures to {OUT_DIR}")
    print(f"Overview: {overview}")


if __name__ == "__main__":
    main()
