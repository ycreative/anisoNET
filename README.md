# anisoNET Reproducibility Package Draft

Date: 2026-06-12

This folder contains draft public-release materials for the anisoNET GPB submission package. It is intended to be copied or adapted into the root of a clean GitHub repository after final path cleanup and repository metadata are locked.

anisoNET is framed as a barrier-aware, tissue-constrained scalar spatial field inference workflow for spatial transcriptomics. The released implementation should be described as a scalar Laplacian PINN approximation, not as tensor-valued diffusion or full divergence-form diffusion.

## Contents

- `environment.yml`: conda environment specification for the analysis and figure workflows.
- `requirements.txt`: pip-style dependency list for non-conda installs.
- `reproduce_figures.md`: commands and input expectations for regenerating the current Figure 1-5 panel assets and supplementary figure previews from existing outputs.
- `repository_manifest.md`: what belongs in GitHub, Zenodo, and the reviewer cache.
- `data_and_code_availability.md`: manuscript-ready data and code availability draft text with URL/DOI placeholders.
- `zenodo_release_checklist.md`: release checklist for code, derived tables, panel assets, and reviewer cache.
- `submission_release_decision.md`: current decision on GitHub, Zenodo, and reviewer-cache submission routes.

## Minimal Reviewer Workflow

From the repository root, create the environment:

```bash
conda env create -f environment.yml
conda activate anisonet-gpb
```

For the current local workspace, the validated Python interpreter is:

```text
K:\software\miniconda\envs\scvi_env\python.exe
```

After public release, scripts should be run from a clean repository root with user-provided `ANISONET_PROJECT_ROOT` or explicit command-line paths. The current Figure 1-5 panel scripts support `ANISONET_PROJECT_ROOT` and `ANISONET_ANALYSIS_ROOT`; some upstream batch scripts still contain local development defaults and should be cleaned before a public release.

## Current Best Submission Files

- Manuscript base: `draft/revised/anisoNET_GPB_integrated_manuscript_v8.docx`
- Cover letter: `draft/revised/anisoNET_GPB_cover_letter_conservative_v1.docx`
- Response template: `draft/revised/anisoNET_GPB_response_to_reviewers_v2_template.docx`
- Supplementary figure legends: `draft/revised/anisoNET_GPB_supplementary_figure_legends_final_candidate.docx`
- Supplementary figures: `codexAnalysis/manuscript_figures/supplementary_final_candidate`
- Supplementary tables: `draft/revised/anisoNET_GPB_supplementary_tables_submission_candidate.xlsx`
- Figure provenance: `codexAnalysis/main_figure_provenance.md`
- Script index: `codexAnalysis/reproducibility_script_index.md`
- References: `codexAnalysis/references/anisoNET_GPB_references_v8.ris`
