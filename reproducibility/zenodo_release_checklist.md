# Zenodo Release Checklist

Date: 2026-06-12

## Before Creating The Release

- [x] Active submission manuscript recorded as v8.
- [x] Conservative cover letter prepared.
- [x] Response-to-reviewers template updated; exact reviewer comments still require author insertion.
- [x] Final-candidate Supplementary Figure S1-S14 set prepared from existing assets.
- [x] Supplementary table workbook exists.
- [x] Figure 1-5 final PDFs exist in figure folders.
- [x] Figure 1-5 panel-generation scripts support repository-root inference and environment-variable overrides.
- [x] Sanitized dataset example config exists.
- [ ] Replace author/contact/funding/declaration placeholders.
- [x] Create or confirm public GitHub repository URL: `https://github.com/ycreative/anisoNET`.
- [x] Create Zenodo draft record and DOI: `10.5281/zenodo.20663845`.
- [ ] Push current `github_repo/anisoNET` staging content to GitHub and confirm the public page shows the current v8/v20260612 README.
- [ ] Add license file.
- [ ] Add `CITATION.cff`.
- [ ] Add package-level public `README.md`.
- [ ] Add synthetic smoke test.
- [ ] Run `py_compile` on selected public `Script/anisonet` and cleaned `Script/workflows` files.
- [ ] Confirm final `CURRENT_PANEL_SET.md` files match submitted figures.
- [ ] Re-export supplementary table workbook after any final edits.
- [ ] Run final manuscript/response/cover claim-boundary QC.

## Archive Components

- [ ] Code source archive.
- [ ] Derived metric summary tables.
- [x] Supplementary table workbook candidate.
- [x] Main figure panel assets and final assembled PDFs exist locally.
- [x] Supplementary Figure S1-S14 candidate assets exist locally.
- [x] Reviewer cache with selected arrays and JSON metadata exists locally as `anisoNET_GPB_reviewer_cache_v20260612.zip`.
- [ ] README describing fast audit versus full rerun routes.

## Metadata

- [ ] Title uses `anisoNET`.
- [ ] Authors match manuscript.
- [ ] Related identifiers include GitHub release and GPB manuscript if available.
- [ ] Keywords include spatial transcriptomics, physics-informed neural network, anatomical barriers, Visium, reproducibility.
- [ ] Version tag matches GitHub release.
- [ ] License is consistent across GitHub and Zenodo.
- [ ] Confirm Zenodo record is published or reviewer-accessible; reserved DOI alone is not enough for final availability text.
