# Draft Data And Code Availability Text

## Code Availability

The source code for anisoNET, including Visium preprocessing utilities, scalar diffusion/resistance field construction, PyTorch-based scalar PINN inference, postprocessing, benchmarking scripts, metric summaries, and figure-panel generation workflows, will be released in a public GitHub repository and archived on Zenodo upon acceptance or submission, according to journal policy. The repository will include an environment file, a script index, figure provenance notes, and commands for regenerating the main figure panel assets from public data or from the reviewer cache.

The current implementation should be cited as a scalar, tissue-constrained, barrier-aware spatial field inference workflow. It does not implement tensor-valued diffusion or the full divergence-form operator `div(D grad C)`.

## Data Availability

The primary mouse brain aging spatial transcriptomics analysis uses public Visium data from GSE193107. Cross-tissue validation and claim-boundary analyses use public mouse kidney, mouse sagittal brain, and mouse liver/APAP spatial transcriptomics datasets from 10x Genomics public resources and GEO, including GSE280515 where applicable. Accession identifiers, sample identifiers, target genes, barrier markers, and analysis roles are listed in the supplementary dataset inventory table.

Raw public sequencing matrices and histology images are not redistributed in the GitHub repository. Scripts are provided to standardize downloaded public Visium files into the layout expected by the workflow. Lightweight derived metric tables, figure provenance files, and current supplementary tables will be included with the release. Larger derived arrays and panel-generation caches will be archived on Zenodo or provided to reviewers as a separate cache when needed for exact figure regeneration.

## Reviewer Cache

For peer review, we recommend providing a compact reviewer cache containing the current Figure 1-5 panel assets, metric summary tables, supplementary table workbook, and selected preflight/PINN outputs for one representative GSE193107 section. This allows reviewers to audit and regenerate the figures without running the entire analysis from raw public data.
