"""Sync GitHub staging files and build a compact reviewer cache archive.

This script performs packaging only. It does not rerun analyses or model fitting.
"""

from __future__ import annotations

import shutil
import zipfile
from datetime import date
from pathlib import Path

from docx import Document


ROOT = Path(__file__).resolve().parents[2]
TODAY = date(2026, 6, 12).isoformat()
GITHUB_DIR = ROOT / "github_repo" / "anisoNET"
REVIEWER_CACHE_DIR = ROOT / "codexAnalysis" / "reviewer_cache"
REVIEWER_CACHE_STAGING = REVIEWER_CACHE_DIR / "anisoNET_GPB_reviewer_cache_v20260612"
REVIEWER_CACHE_ZIP = REVIEWER_CACHE_DIR / "anisoNET_GPB_reviewer_cache_v20260612.zip"
SUBMISSION_DIR = ROOT / "codexAnalysis" / "submission_package"


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def ensure_inside(path: Path, parent: Path) -> None:
    path_resolved = path.resolve()
    parent_resolved = parent.resolve()
    if parent_resolved not in path_resolved.parents and path_resolved != parent_resolved:
        raise ValueError(f"Refusing to operate outside {parent}: {path}")


def clear_dir(path: Path) -> None:
    ensure_inside(path, ROOT)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    ensure_inside(dst, ROOT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree_filtered(src: Path, dst: Path, suffixes: set[str] | None = None) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    ensure_inside(dst, ROOT)
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        if path.is_dir():
            if path.name == "__pycache__":
                continue
            continue
        if "__pycache__" in path.parts:
            continue
        if suffixes is not None and path.suffix.lower() not in suffixes:
            continue
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)


def write_text(path: Path, text: str) -> None:
    ensure_inside(path, ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def sync_github_staging() -> None:
    GITHUB_DIR.mkdir(parents=True, exist_ok=True)

    copy_file(ROOT / "codexAnalysis" / "reproducibility_package" / "README.md", GITHUB_DIR / "README.md")
    copy_file(ROOT / "codexAnalysis" / "reproducibility_package" / "environment.yml", GITHUB_DIR / "environment.yml")
    copy_file(ROOT / "codexAnalysis" / "reproducibility_package" / "requirements.txt", GITHUB_DIR / "requirements.txt")

    for name in [
        "data_and_code_availability.md",
        "repository_manifest.md",
        "reproduce_figures.md",
        "zenodo_release_checklist.md",
        "submission_release_decision.md",
    ]:
        copy_file(ROOT / "codexAnalysis" / "reproducibility_package" / name, GITHUB_DIR / "reproducibility" / name)

    docs = {
        ROOT / "codexAnalysis" / "main_figure_provenance.md": GITHUB_DIR / "docs" / "main_figure_provenance.md",
        ROOT / "codexAnalysis" / "final_main_figure_manifest.md": GITHUB_DIR / "docs" / "final_main_figure_manifest.md",
        ROOT / "codexAnalysis" / "reproducibility_script_index.md": GITHUB_DIR / "docs" / "reproducibility_script_index.md",
        ROOT / "codexAnalysis" / "reproducibility_package_plan.md": GITHUB_DIR / "docs" / "reproducibility_package_plan.md",
        ROOT / "codexAnalysis" / "barrier_metric_spec.md": GITHUB_DIR / "docs" / "barrier_metric_spec.md",
        ROOT / "codexAnalysis" / "supplementary_materials_map.md": GITHUB_DIR / "docs" / "supplementary_materials_map.md",
        ROOT / "codexAnalysis" / "references" / "anisoNET_GPB_references_v8.md": GITHUB_DIR / "docs" / "references" / "anisoNET_GPB_references_v8.md",
        ROOT / "codexAnalysis" / "references" / "anisoNET_GPB_references_v8.bib": GITHUB_DIR / "docs" / "references" / "anisoNET_GPB_references_v8.bib",
        ROOT / "codexAnalysis" / "references" / "anisoNET_GPB_references_v8.ris": GITHUB_DIR / "docs" / "references" / "anisoNET_GPB_references_v8.ris",
    }
    for src, dst in docs.items():
        copy_file(src, dst)

    copy_tree_filtered(ROOT / "Script" / "anisonet", GITHUB_DIR / "Script" / "anisonet", {".py"})
    copy_tree_filtered(ROOT / "Script" / "configs", GITHUB_DIR / "Script" / "configs", {".json"})

    workflow_names = [
        "generate_figure1_panel_assets.py",
        "generate_figure2_panel_assets.py",
        "generate_figure3_panel_assets.py",
        "generate_figure4_panel_assets.py",
        "generate_figure5_panel_assets.py",
        "organize_supplementary_figures.py",
        "prepare_gpb_submission_admin_package.py",
        "prepare_github_staging_and_reviewer_cache.py",
        "manuscript_claim_qc.py",
        "run_anisonet_preflight.py",
        "run_anisonet_pinn.py",
        "export_supplementary_tables_xlsx.py",
        "compute_barrier_field_metrics.py",
        "heldout_gse193107_benchmark.py",
        "synthetic_barrier_benchmark.py",
        "cross_tissue_evidence_summary.py",
    ]
    workflow_dst = GITHUB_DIR / "Script" / "workflows"
    if workflow_dst.exists():
        shutil.rmtree(workflow_dst)
    workflow_dst.mkdir(parents=True, exist_ok=True)
    for name in workflow_names:
        src = ROOT / "Script" / "workflows" / name
        if src.exists():
            copy_file(src, workflow_dst / name)

    write_text(
        GITHUB_DIR / "GITHUB_UPLOAD_STEPS.md",
        """
# GitHub Upload Steps

Current local staging folder:

```text
K:\\YC\\experiment\\STagent\\github_repo\\anisoNET
```

Configured remote in this staging repository:

```text
https://github.com/ycreative/anisoNET.git
```

`git` is not currently available in this PowerShell PATH, so upload can be done either by:

- GitHub web UI: open the repository, choose upload files, and upload the contents of this folder.
- Git command line on a machine/session where Git is available.

Recommended visibility:

- Start private for reviewer/package checks.
- Make public only after final author, license, CITATION, and path-cleanup checks.

Do not upload raw public datasets or full local analysis output trees to GitHub.
""",
    )


def reviewer_cache_readme() -> str:
    return f"""
# anisoNET GPB Reviewer Cache

Date: {TODAY}

This cache is a compact reviewer-facing package for auditing the current anisoNET GPB submission materials without rerunning the full analysis from raw public data.

## Intended Use

- Inspect final Figure 1-5 files and active panel mappings.
- Inspect Supplementary Figures S1-S14 and figure legends.
- Inspect supplementary tables and metric/provenance files.
- Inspect selected representative preflight/PINN outputs for one GSE193107 brain-aging section.

## Not Included

- Raw public Visium datasets.
- Full standardized matrices.
- Full preflight/PINN output trees.
- Broad parameter sweeps and large intermediate arrays.

## Public Data Sources

Raw data should be obtained from public sources listed in the manuscript and supplementary dataset inventory, including GSE193107, 10x public mouse kidney/sagittal brain resources, and GSE280515 where applicable.

## Availability Placeholders

- GitHub repository URL: [AUTHOR TO COMPLETE]
- Zenodo DOI: [AUTHOR TO COMPLETE]
- Journal portal reviewer-cache filename: anisoNET_GPB_reviewer_cache_v20260612.zip
"""


def cache_add_file(src: Path, dst_rel: str, manifest: list[tuple[str, str]]) -> None:
    if not src.exists():
        manifest.append((dst_rel, "MISSING"))
        return
    dst = REVIEWER_CACHE_STAGING / dst_rel
    copy_file(src, dst)
    manifest.append((dst_rel, rel(src)))


def build_reviewer_cache() -> None:
    clear_dir(REVIEWER_CACHE_STAGING)
    REVIEWER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[tuple[str, str]] = []

    write_text(REVIEWER_CACHE_STAGING / "README.md", reviewer_cache_readme())

    main_figs = [
        ("Figure1_method_overview", "Figure1"),
        ("Figure2_gse193107_primary_application", "Figure2"),
        ("Figure3_benchmark_and_synthetic_validation", "Figure3"),
        ("Figure4_robustness_reproducibility", "Figure4"),
        ("Figure5_cross_tissue_boundary", "Figure5"),
    ]
    for folder, stem in main_figs:
        src_dir = ROOT / "codexAnalysis" / "manuscript_figures" / folder
        for ext in [".pdf", ".png"]:
            cache_add_file(src_dir / f"{stem}{ext}", f"main_figures/{stem}{ext}", manifest)
        cache_add_file(src_dir / "CURRENT_PANEL_SET.md", f"main_figures/{stem}_CURRENT_PANEL_SET.md", manifest)

    supp_dir = ROOT / "codexAnalysis" / "manuscript_figures" / "supplementary_final_candidate"
    for path in sorted(supp_dir.glob("*")):
        if path.is_file():
            cache_add_file(path, f"supplementary_figures/{path.name}", manifest)

    for path in sorted((ROOT / "codexAnalysis" / "supplementary_tables").glob("*")):
        if path.is_file():
            cache_add_file(path, f"supplementary_tables/{path.name}", manifest)
    cache_add_file(
        ROOT / "draft" / "revised" / "anisoNET_GPB_supplementary_tables_submission_candidate.xlsx",
        "supplementary_tables/anisoNET_GPB_supplementary_tables_submission_candidate.xlsx",
        manifest,
    )

    docs = [
        ROOT / "codexAnalysis" / "final_main_figure_manifest.md",
        ROOT / "codexAnalysis" / "main_figure_provenance.md",
        ROOT / "codexAnalysis" / "supplementary_materials_map.md",
        ROOT / "codexAnalysis" / "reproducibility_script_index.md",
        ROOT / "codexAnalysis" / "barrier_metric_spec.md",
        ROOT / "codexAnalysis" / "reproducibility_package" / "README.md",
        ROOT / "codexAnalysis" / "reproducibility_package" / "reproduce_figures.md",
        ROOT / "codexAnalysis" / "reproducibility_package" / "data_and_code_availability.md",
        ROOT / "codexAnalysis" / "reproducibility_package" / "submission_release_decision.md",
        ROOT / "codexAnalysis" / "references" / "anisoNET_GPB_references_v8.md",
        ROOT / "codexAnalysis" / "references" / "anisoNET_GPB_references_v8.ris",
        ROOT / "codexAnalysis" / "references" / "anisoNET_GPB_references_v8.bib",
    ]
    for path in docs:
        cache_add_file(path, f"documentation/{path.name}", manifest)

    qc_files = [
        ROOT / "codexAnalysis" / "manuscript_qc" / "reference_qc_integrated_v8_20260612.csv",
        ROOT / "codexAnalysis" / "manuscript_qc" / "claim_qc_hits_integrated_v8_20260612.csv",
        ROOT / "codexAnalysis" / "manuscript_qc" / "claim_qc_hits_cover_letter_v1_20260612.csv",
        ROOT / "codexAnalysis" / "manuscript_qc" / "claim_qc_hits_response_v2_20260612.csv",
    ]
    for path in qc_files:
        cache_add_file(path, f"qc/{path.name}", manifest)

    for path in sorted((ROOT / "codexAnalysis" / "barrier_field_metrics").glob("*")):
        if path.is_file():
            cache_add_file(path, f"metric_tables/{path.name}", manifest)

    representative_files = [
        ROOT / "codexAnalysis" / "preflight" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin" / "preflight_fields.png",
        ROOT / "codexAnalysis" / "preflight" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin" / "preflight_metrics.json",
        ROOT / "codexAnalysis" / "pinn" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin" / "fourier_refined_16g_gauss07_batch" / "pinn_fields.png",
        ROOT / "codexAnalysis" / "pinn" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin" / "fourier_refined_16g_gauss07_batch" / "pinn_metrics.json",
        ROOT / "codexAnalysis" / "pinn" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin" / "fourier_refined_16g_gauss07_batch" / "pinn_history.json",
    ]
    for path in representative_files:
        cache_add_file(path, f"representative_outputs/{path.name}", manifest)

    manifest_path = REVIEWER_CACHE_STAGING / "MANIFEST.tsv"
    write_text(manifest_path, "cache_path\tsource_path\n" + "\n".join(f"{a}\t{b}" for a, b in manifest))

    if REVIEWER_CACHE_ZIP.exists():
        REVIEWER_CACHE_ZIP.unlink()
    with zipfile.ZipFile(REVIEWER_CACHE_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in REVIEWER_CACHE_STAGING.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(REVIEWER_CACHE_STAGING.parent))

    write_text(
        REVIEWER_CACHE_DIR / "REVIEWER_CACHE_STATUS.md",
        f"""
# Reviewer Cache Status

Date: {TODAY}

Archive:

- `{rel(REVIEWER_CACHE_ZIP)}`

Staging folder:

- `{rel(REVIEWER_CACHE_STAGING)}`

Current use:

- Upload this zip to Zenodo as part of the derived/reviewer materials, or upload it directly to the GPB/OUP portal if a reviewer-cache file is requested.
- The public link/DOI is not available until the user uploads or publishes the archive.
""",
    )


def potential_reviewers_assessment() -> None:
    source = Path(r"K:\YC\download\Potential Reviewers.docx")
    paragraphs: list[str] = []
    if source.exists():
        doc = Document(source)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    source_text = "\n".join(paragraphs)

    assessment = f"""
# Potential Reviewers Assessment

Date: {TODAY}

Source file:

- `K:/YC/download/Potential Reviewers.docx`

## Verdict

The previous potential-reviewer file is usable only as a starting point. It should not be submitted unchanged because it uses the old manuscript title and old high-risk framing around physical tensors, clinical translation, and semantic biomarkers.

## Current Reviewer Fit

1. Dr. Lu Lu

- Fit: potentially suitable as a scientific machine learning/PINN reviewer.
- Keep only if the email and current affiliation are verified before submission.
- Rewrite the reason around scalar PINN methodology, numerical formulation, and validation boundaries.

2. Dr. Ye Yuan

- Fit: possible but lower-confidence from the current local information.
- Keep only if current affiliation, email, and topic fit are verified.
- Rewrite the reason around physics-guided machine learning or scientific ML, not physical tensors in biological tissue.

3. Dr. Clemens Schmitt

- Fit: biologically relevant for senescence/SASP context, but less central for spatial transcriptomics field inference and PINN methods.
- Use as an optional biology-side reviewer only if GPB asks for broad biological reviewers.
- Remove clinical-translational and diagnostic-biomarker language.

## Conservative Revised Wording

Suggested reviewer category 1: scientific machine learning / PINN reviewer.

Reason template: The reviewer has expertise in physics-informed neural networks and scientific machine learning, and could evaluate the scalar reaction-diffusion formulation, loss design, numerical assumptions, and validation boundaries of anisoNET.

Suggested reviewer category 2: spatial transcriptomics computational reviewer.

Reason template: The reviewer has expertise in spatial transcriptomics analysis and could evaluate whether the barrier-aware field-inference task, benchmarks, and cross-tissue claim boundaries are appropriate for spatial omics data.

Suggested reviewer category 3: brain aging / senescence biology reviewer.

Reason template: The reviewer has expertise in aging-associated tissue microenvironments and could evaluate the biological plausibility of the Apoe/Gfap/CNS-myelin primary brain application while recognizing that the manuscript's main contribution is computational.

## Extracted Original Text

```text
{source_text}
```
"""
    write_text(SUBMISSION_DIR / "potential_reviewers_assessment.md", assessment)


def main() -> None:
    sync_github_staging()
    build_reviewer_cache()
    potential_reviewers_assessment()
    print(f"Updated GitHub staging: {GITHUB_DIR}")
    print(f"Reviewer cache zip: {REVIEWER_CACHE_ZIP}")
    print(f"Reviewer assessment: {SUBMISSION_DIR / 'potential_reviewers_assessment.md'}")


if __name__ == "__main__":
    main()
