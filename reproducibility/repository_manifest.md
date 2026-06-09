# Repository And Archive Manifest

## Include In GitHub

Source code:

- `Script/anisonet/`
- `Script/workflows/`
- sanitized `Script/configs/anisonet_datasets.example.json`

Documentation:

- `README.md`
- `environment.yml`
- `requirements.txt`
- `reproducibility/reproduce_figures.md`
- `reproducibility/data_and_code_availability.md`
- `docs/main_figure_provenance.md`
- `docs/reproducibility_script_index.md`
- `docs/barrier_metric_spec.md`
- `docs/supplementary_materials_map.md`

Lightweight derived outputs:

- supplementary table CSV files;
- metric summary CSV files used by final figures;
- `CURRENT_PANEL_SET.md` files for Figures 1-5.

## Include In Zenodo

- GitHub release source archive.
- Final supplementary table workbook.
- Final Figure 1-5 panel assets and assembled PDFs.
- Lightweight reviewer cache for exact figure-panel regeneration.
- Optional selected NPY/JSON intermediate outputs for one representative rerun.

## Exclude From GitHub

- raw public datasets;
- full standardized Visium matrices and image folders;
- full PINN output trees;
- full sweep directories;
- temporary workflow tests;
- local manuscript Word temp files;
- local absolute-path dataset config;
- old exploratory script trees unless clearly archived.

## Suggested `.gitignore`

```gitignore
codexAnalysis/processed_visium/
codexAnalysis/preflight/
codexAnalysis/pinn/
codexAnalysis/sweeps/
codexAnalysis/workflow_tests/
codexAnalysis/pdf_review/renders/
draft/**/*.docx
draft/**/*.xlsx
~$*.docx
*.tmp
__pycache__/
*.pyc
```

Keep final manuscript and table files in Zenodo or journal submission folders rather than the public code repository unless the authors want full manuscript-version tracking.
