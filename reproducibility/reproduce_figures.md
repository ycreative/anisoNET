# Reproduce Current Figure Panel Assets

This document records the current reviewer-facing reproduction route for Figure 1-5 panel assets. It assumes the analysis outputs already exist under `codexAnalysis/`. It does not rerun all upstream benchmarks by default.

## Environment

Validated local interpreter:

```powershell
K:\software\miniconda\envs\scvi_env\python.exe
```

Public release target:

```bash
conda env create -f environment.yml
conda activate anisonet-gpb
```

## Path Configuration

The current Figure 1-5 panel scripts default to the repository root inferred from their file location. They also support environment-variable overrides:

```powershell
$env:ANISONET_PROJECT_ROOT = "K:\YC\experiment\STagent"
$env:ANISONET_ANALYSIS_ROOT = "K:\YC\experiment\STagent\codexAnalysis"
```

Some upstream batch scripts still contain local development defaults. Treat the first GitHub upload as a private reviewer/release-candidate repository until those are cleaned.

## Main Figure Panel Commands

Run from the repository root:

```powershell
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\generate_figure1_panel_assets.py
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\generate_figure2_panel_assets.py
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\generate_figure3_panel_assets.py
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\generate_figure4_panel_assets.py
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\generate_figure5_panel_assets.py
```

Expected outputs:

```text
codexAnalysis/manuscript_figures/Figure1_method_overview
codexAnalysis/manuscript_figures/Figure2_gse193107_primary_application
codexAnalysis/manuscript_figures/Figure3_benchmark_and_synthetic_validation
codexAnalysis/manuscript_figures/Figure4_robustness_reproducibility
codexAnalysis/manuscript_figures/Figure5_cross_tissue_boundary
```

Each directory contains a `CURRENT_PANEL_SET.md` file. Use that file, not older historical panel files in the same folder, to identify the current active panel set.

## Supplementary Figure Candidate Set

This command assembles Supplementary Figures S1-S14 from existing PNG assets. It does not rerun model fitting or recompute metrics. The working folder is provenance; the current submission-candidate copies are stored under `codexAnalysis/manuscript_figures/supplementary_final_candidate`.

```powershell
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\organize_supplementary_figures.py
```

Expected outputs:

```text
codexAnalysis/manuscript_figures/supplementary_working
codexAnalysis/manuscript_figures/supplementary_final_candidate
```

The final-candidate folder was prepared with:

```powershell
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\prepare_gpb_submission_admin_package.py
```

## Selected Single-Sample Model Rerun

This route checks the core workflow without rerunning the full manuscript.

```powershell
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\run_anisonet_preflight.py `
  --sample-dir codexAnalysis\processed_visium\brain_aging_gse193107\GSM5773457_Old_mouse_brain_A1-2 `
  --target-gene Apoe `
  --barrier-genes Mbp Plp1 Mobp `
  --output-dir codexAnalysis\reviewer_rerun\preflight\GSM5773457_Old_mouse_brain_A1-2\Apoe_CNS_Myelin

K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\run_anisonet_pinn.py `
  --sample-dir codexAnalysis\processed_visium\brain_aging_gse193107\GSM5773457_Old_mouse_brain_A1-2 `
  --preflight-dir codexAnalysis\reviewer_rerun\preflight\GSM5773457_Old_mouse_brain_A1-2\Apoe_CNS_Myelin `
  --target-gene Apoe `
  --profile fourier_refined_16g `
  --postprocess-sigma 0.7 `
  --output-dir codexAnalysis\reviewer_rerun\pinn\GSM5773457_Old_mouse_brain_A1-2\Apoe_CNS_Myelin\fourier_refined_16g_gauss07
```

## Supplementary Tables And Manuscript Utilities

```powershell
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\export_supplementary_tables_xlsx.py
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\build_integrated_manuscript_docx.py
K:\software\miniconda\envs\scvi_env\python.exe Script\workflows\manuscript_claim_qc.py --help
```

## Full Analysis Rerun Policy

The manuscript package was generated incrementally. A full rerun of all upstream analyses is not recommended as the default reviewer route. Prefer:

1. audit the code and provenance;
2. regenerate figure panels from existing output tables and arrays;
3. rerun one representative preflight/PINN task;
4. rerun larger batches only if a specific result is questioned.
