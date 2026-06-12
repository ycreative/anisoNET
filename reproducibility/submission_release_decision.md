# Submission Release Decision

Date: 2026-06-12

Recommended GPB submission strategy:

- Primary code route: public GitHub repository URL.
- Archival route: Zenodo DOI linked to the GitHub release and selected derived materials.
- Reviewer audit route: compact reviewer cache containing final Figure 1-5 panel assets, Supplementary Figure S1-S14 PDFs/PNGs, supplementary table workbook, metric summary tables, and selected representative preflight/PINN arrays.

Use the reviewer cache for bulky intermediate outputs rather than placing full `preflight`, `pinn`, `processed_visium`, or sweep folders in GitHub.

Current links and files to insert:

- GitHub repository: `https://github.com/ycreative/anisoNET`
- Zenodo DOI: `https://doi.org/10.5281/zenodo.20663845`
- Reviewer cache: `anisoNET_GPB_reviewer_cache_v20260612.zip`

Current decision:

- Target a GitHub + Zenodo pair for submission if the repository can be cleaned in time.
- Keep a reviewer cache as a fallback and as a fast-audit supplement.
- Do not promise unrestricted redistribution of raw public datasets; provide accessions and standardization scripts instead.
- Before final submission, confirm that the current `github_repo/anisoNET` staging content has been pushed and that the Zenodo record is published or otherwise accessible to reviewers.
