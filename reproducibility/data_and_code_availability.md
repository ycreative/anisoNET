# Draft Data And Code Availability Text

Date: 2026-06-12

## Recommended Submission Text

The source code for anisoNET, including Visium preprocessing utilities, scalar diffusion/resistance field construction, PyTorch-based scalar PINN inference, postprocessing, benchmarking scripts, metric summaries, and figure-panel generation workflows, is prepared for release at https://github.com/ycreative/anisoNET. Reviewer-facing reproducibility materials are archived or reserved under Zenodo DOI https://doi.org/10.5281/zenodo.20663845. A compact reviewer cache containing selected precomputed arrays, metric tables, final panel assets, Supplementary Figure S1-S14 files, and the supplementary table workbook is provided as `anisoNET_GPB_reviewer_cache_v20260612.zip` for peer-review auditing.

The current implementation should be cited as a scalar, tissue-constrained, barrier-aware spatial field inference workflow. It does not implement tensor-valued diffusion or the full divergence-form operator `div(D grad C)`.

## Data Availability

The primary mouse brain aging spatial transcriptomics analysis uses public Visium data from GSE193107. Cross-tissue validation and claim-boundary analyses use public mouse kidney, mouse sagittal brain, and mouse liver/APAP spatial transcriptomics datasets from 10x Genomics public resources and GEO, including GSE280515 where applicable. Accession identifiers, sample identifiers, target genes, barrier markers, and analysis roles are listed in the supplementary dataset inventory table.

Raw public sequencing matrices and histology images are not redistributed in the GitHub repository. Scripts are provided to standardize downloaded public Visium files into the layout expected by the workflow. Lightweight derived metric tables, figure provenance files, current Supplementary Figure S1-S14 assets, and supplementary tables are intended for GitHub/Zenodo release. Larger derived arrays and panel-generation caches should be archived on Zenodo or provided to reviewers as a compact cache when needed for exact figure regeneration.

## Current Decision

Use GitHub + Zenodo as the primary public availability route, and keep `anisoNET_GPB_reviewer_cache_v20260612.zip` as the fast-audit route for large or selected precomputed intermediates. Before formal submission, confirm that the GitHub repository contains the current staged files and that the Zenodo DOI is published or reviewer-accessible.
