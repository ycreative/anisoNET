# GitHub, Zenodo, And Reviewer Reproducibility Package Plan

Date: 2026-06-09

Purpose:

- Define what should be made public for the anisoNET GPB revision.
- Separate source code, lightweight derived tables, figure assets, large generated intermediates, and public raw data.
- Provide a practical route to a reviewer package without rerunning the full analysis from scratch.

## Recommended Release Structure

Public GitHub repository:

```text
anisoNET/
  README.md
  LICENSE
  CITATION.cff
  environment.yml
  requirements.txt
  Script/
    anisonet/
    workflows/
    configs/
  reproducibility/
    reproduce_figures.md
    data_and_code_availability.md
    repository_manifest.md
    zenodo_release_checklist.md
  docs/
    main_figure_provenance.md
    reproducibility_script_index.md
    barrier_metric_spec.md
    supplementary_materials_map.md
  examples/
    synthetic_smoke_test/
```

Zenodo record:

```text
anisoNET-code-vX.Y.Z.zip
anisoNET-derived-tables-vX.Y.Z.zip
anisoNET-main-figure-panel-assets-vX.Y.Z.zip
anisoNET-minimal-reviewer-cache-vX.Y.Z.zip
```

Reviewer private package, if GPB allows confidential files:

```text
reviewer_cache/
  manuscript_figures/Figure1_method_overview/
  manuscript_figures/Figure2_gse193107_primary_application/
  manuscript_figures/Figure3_benchmark_and_synthetic_validation/
  manuscript_figures/Figure4_robustness_reproducibility/
  manuscript_figures/Figure5_cross_tissue_boundary/
  supplementary_tables/
  barrier_field_metrics/
  selected_preflight_and_pinn_outputs/
```

## Public Code

Publish:

- `Script/anisonet/`
  - lightweight Visium reader;
  - preprocessing;
  - scalar diffusion/resistance construction;
  - native PyTorch PINN solver;
  - postprocessing;
  - metrics.
- `Script/workflows/`
  - dataset standardization scripts;
  - preflight construction;
  - PINN inference;
  - benchmarks and summaries;
  - figure panel generation;
  - manuscript/supplementary table export and claim QC utilities.
- `Script/configs/`
  - publish a sanitized example config only.
  - do not publish local Windows paths or garbled-path entries from the current local config.

Do not publish as active workflow code:

- historical exploratory scripts under `Script/agorithmdebugging/`, `Script/figure1_3/`, `Script/Figure4/`, `Script/figure5/`, and `Script/sampleHandling/` unless moved to an `archive/` folder with clear "not used for final figures" labeling.
- older scripts with promotional names or obsolete tensor/divergence-form claims.

Cleanup targets before public GitHub release:

- Current Figure 1-5 panel-generation scripts now support repository-root inference and `ANISONET_PROJECT_ROOT` / `ANISONET_ANALYSIS_ROOT` overrides.
- Several upstream batch and summary scripts still contain local development defaults and should be parameterized before a public release.
- Add a sanitized `Script/configs/anisonet_datasets.example.json`.
- Add a tiny synthetic smoke test so installation can be verified without downloading Visium data.

## Public Derived Data

Publish in GitHub or Zenodo:

- `codexAnalysis/barrier_field_metrics/*.csv`
- `codexAnalysis/profile_definitions/*.csv` or equivalent profile tables.
- `codexAnalysis/supplementary_tables/*.csv`
- `draft/revised/anisoNET_GPB_supplementary_tables_submission_candidate.xlsx`
- `codexAnalysis/main_figure_provenance.md`
- `codexAnalysis/reproducibility_script_index.md`
- `codexAnalysis/supplementary_materials_map.md`
- `codexAnalysis/barrier_metric_spec.md`
- current `CURRENT_PANEL_SET.md` files for Figures 1-5.

Publish in Zenodo, not necessarily GitHub:

- final per-panel PNG/PDF assets for Figures 1-5;
- final assembled figure PDFs after Inkscape layout;
- selected reviewer-cache NPY/JSON outputs needed to regenerate figure panels quickly.

## Outputs To Keep Out Of The Public Repository

Do not include large generated intermediates in GitHub:

- full `codexAnalysis/pinn/` trees;
- full `codexAnalysis/preflight/` trees;
- full `codexAnalysis/processed_visium/` standardized matrices and images;
- broad sweep outputs under `codexAnalysis/sweeps/`;
- temporary workflow tests;
- old rough assemblies not in `CURRENT_PANEL_SET.md`;
- local PDF review renders;
- Word temporary files such as `~$*.docx`.

These may be supplied selectively in Zenodo or reviewer cache if they are required for exact panel regeneration.

## Raw Data Policy

Use public downloads rather than redistributing raw public datasets when license and size make that preferable.

Primary public data:

- GSE193107 mouse brain aging Visium data.
- 10x Genomics public mouse kidney Visium data.
- 10x Genomics public mouse sagittal brain Visium data.
- GSE280515 mouse liver/APAP data.

Repository documentation should provide:

- accession or vendor source;
- exact sample identifiers;
- expected input file names;
- command to standardize each dataset into a 10x-like layout;
- statement that derived outputs can be regenerated from public data and released code.

## Reviewer Reproducibility Levels

Level 1, fastest audit:

- inspect source code;
- inspect `main_figure_provenance.md`;
- inspect supplementary table workbook;
- inspect current panel assets.

Level 2, panel regeneration:

- use Zenodo reviewer cache with selected precomputed arrays and summary CSVs;
- run `generate_figure1_panel_assets.py` through `generate_figure5_panel_assets.py`.

Level 3, selected model rerun:

- download one GSE193107 section;
- standardize it;
- run preflight;
- run `fourier_refined_16g` PINN for one target;
- evaluate the field.

Level 4, full analysis rerun:

- download all public datasets;
- run all batch benchmarks and summaries.
- This is the most expensive route and should not be the default reviewer path.

## Current Package Status

Ready:

- core scalar anisoNET modules;
- workflow scripts;
- figure panel asset scripts;
- figure provenance;
- reproducibility script index;
- supplementary table candidate workbook.

Partly ready:

- environment files;
- public README;
- exact figure reproduction commands;
- data/code availability text.

Needs cleanup before public release:

- remove or archive obsolete exploratory script trees;
- parameterize hard-coded local paths in remaining upstream workflow scripts;
- replace local dataset config with a sanitized example;
- add license and citation metadata;
- add a small smoke test.
