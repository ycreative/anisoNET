# Zenodo Release Checklist

## Before Creating The Release

- [ ] Parameterize hard-coded local paths in Figure 1-5 panel-generation scripts.
- [ ] Replace local dataset config with `anisonet_datasets.example.json`.
- [ ] Add license file.
- [ ] Add `CITATION.cff`.
- [ ] Add package-level `README.md`.
- [ ] Add synthetic smoke test.
- [ ] Run `py_compile` on `Script/anisonet` and `Script/workflows`.
- [ ] Regenerate final Figure 1-5 panel assets after final manuscript edits.
- [ ] Confirm final `CURRENT_PANEL_SET.md` files match submitted figures.
- [ ] Re-export supplementary table workbook.
- [ ] Run final manuscript/response claim-boundary QC.

## Archive Components

- [ ] Code source archive.
- [ ] Derived metric summary tables.
- [ ] Supplementary table workbook.
- [ ] Main figure panel assets.
- [ ] Final assembled figure PDFs.
- [ ] Reviewer cache with selected arrays and JSON metadata.
- [ ] README describing which files are needed for fast audit versus full rerun.

## Metadata

- [ ] Title uses `anisoNET`.
- [ ] Authors match manuscript.
- [ ] Related identifiers include GitHub release and GPB manuscript if available.
- [ ] Keywords include spatial transcriptomics, physics-informed neural network, anatomical barriers, Visium, reproducibility.
- [ ] Version tag matches GitHub release.
- [ ] License is consistent across GitHub and Zenodo.
