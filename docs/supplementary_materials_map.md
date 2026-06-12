# Supplementary Materials Map For GPB Revision

Date: 2026-06-08

This map assigns existing outputs to the current GPB supplementary materials package. Supplementary Figures S1-S14 are now locked as the final-candidate submission set unless the journal portal imposes a file-size limit or the authors decide to split files during upload.

## Current Final-Candidate Supplementary Figure Folder

Use this folder for submission-package assembly:

- Directory: `codexAnalysis/manuscript_figures/supplementary_final_candidate`
- Current set: `codexAnalysis/manuscript_figures/supplementary_final_candidate/CURRENT_SUPPLEMENTARY_FIGURE_SET.md`
- Combined PDF: `codexAnalysis/manuscript_figures/supplementary_final_candidate/Supplementary_Figures_S1-S14_final_candidate.pdf`
- Legends: `codexAnalysis/manuscript_figures/supplementary_final_candidate/supplementary_figure_legends_final_candidate.md`
- Source map: `codexAnalysis/manuscript_figures/supplementary_final_candidate/supplementary_figure_source_map_final.csv`
- Word legends: `draft/revised/anisoNET_GPB_supplementary_figure_legends_final_candidate.docx`

The older working folder is retained only as provenance:

- Directory: `codexAnalysis/manuscript_figures/supplementary_working`
- Overview: `codexAnalysis/manuscript_figures/supplementary_working/Supplementary_Figure_Working_Overview.png`
- Current set: `codexAnalysis/manuscript_figures/supplementary_working/CURRENT_SUPPLEMENTARY_FIGURE_SET.md`
- Source map: `codexAnalysis/manuscript_figures/supplementary_working/supplementary_working_source_map.csv`
- Script: `Script/workflows/organize_supplementary_figures.py`

The working overview is not part of the final candidate set.

## Main Figures

Main figure provenance:

- `codexAnalysis/main_figure_provenance.md`

Figure 1: Method overview.

- Draft legend: `codexAnalysis/manuscript_text_drafts/figure1_method_overview_legend.md`
- Methods alignment: `codexAnalysis/manuscript_text_drafts/methods_definitions_scalar_model.md`
- Assets: `codexAnalysis/manuscript_figures/Figure1_method_overview`
- Panel list: `codexAnalysis/manuscript_figures/Figure1_method_overview/CURRENT_PANEL_SET.md`

Figure 2: Primary GSE193107 brain-aging multi-gene application.

- Assets: `codexAnalysis/manuscript_figures/Figure2_gse193107_primary_application`
- Panel list: `codexAnalysis/manuscript_figures/Figure2_gse193107_primary_application/CURRENT_PANEL_SET.md`
- Draft legend: `codexAnalysis/manuscript_text_drafts/figure2_5_legends_spatial_first.md`

Figure 3: Benchmark and synthetic validation.

- Assets: `codexAnalysis/manuscript_figures/Figure3_benchmark_and_synthetic_validation`
- Panel list: `codexAnalysis/manuscript_figures/Figure3_benchmark_and_synthetic_validation/CURRENT_PANEL_SET.md`
- Metric definitions: `codexAnalysis/barrier_metric_spec.md`

Figure 4: Robustness, sensitivity, and targeted extension.

- Assets: `codexAnalysis/manuscript_figures/Figure4_robustness_reproducibility`
- Panel list: `codexAnalysis/manuscript_figures/Figure4_robustness_reproducibility/CURRENT_PANEL_SET.md`
- Text draft: `codexAnalysis/manuscript_text_drafts/profile_sensitivity_methods_results.md`

Figure 5: Cross-tissue portability and claim boundary.

- Assets: `codexAnalysis/manuscript_figures/Figure5_cross_tissue_boundary`
- Panel list: `codexAnalysis/manuscript_figures/Figure5_cross_tissue_boundary/CURRENT_PANEL_SET.md`
- Text drafts:
  - `codexAnalysis/manuscript_text_drafts/cross_tissue_validation_boundary_results.md`
  - `codexAnalysis/manuscript_text_drafts/kidney_cross_tissue_benchmark_results.md`

## Final-Candidate Supplementary Figures

Supplementary Figure S1: Full GSE193107 preprocessing QC.

- Suggested content: H&E, tissue mask, source, barrier, diffusion, and resistance grids for all sections.
- Role: demonstrate standardized field construction.

Supplementary Figure S2: Full Apoe and Gfap field montage.

- Suggested content: all eight sections, measured source expression, raw output, tissue-masked output, optional smoothed output.
- Role: support Figure 2 with complete section-level visual evidence.

Supplementary Figure S3: Generic held-out benchmark details.

- Suggested content: per-target and per-section held-out metrics, baseline comparison, high-barrier/edge split if available.
- Role: document the interpolation claim boundary.

Supplementary Figure S4: Synthetic benchmark replicates.

- Suggested content: synthetic source, barrier, ground truth, prediction, error, leakage maps across replicate settings.
- Role: support Figure 3 mechanism with replicate visual evidence.

Supplementary Figure S5: Histology-prior sensitivity.

- Suggested content: brightness versus hematoxylin structural priors across representative sections and metrics.
- Role: support Figure 4A/C.

Supplementary Figure S6: Low-PDE profile all-section sensitivity.

- Suggested content: default versus low-PDE fields, source Pearson deltas, MSE deltas, roughness deltas across all eight sections and both targets.
- Role: support Figure 4B/D/H.

Supplementary Figure S7: Source clipping and random seed stability.

- Suggested content: clipping percentile sensitivity and seed-level low-PDE deltas.
- Role: support Figure 4E/F.

Supplementary Figure S8: Runtime, memory, and convergence diagnostics.

- Suggested content: runtime per target, CUDA peak memory, representative loss curves if available.
- Role: support computational reproducibility; runtime is not an active main Figure 4 claim.

Supplementary Figure S9: Kidney evidence-boundary analysis.

- Suggested content: Slc34a1 and Umod method comparison, resistance-IDW baseline, line-prior, grid-geodesic, prior-hybrid readouts.
- Role: support Figure 5B.

Supplementary Figure S10: Kidney systematic marker screen.

- Suggested content: compartment marker screen, selected candidate tasks, follow-up high-barrier/edge metrics.
- Role: support Figure 5G.

Supplementary Figure S11: Sagittal brain negative-control analysis.

- Suggested content: healthy sagittal brain source/barrier construction and high-barrier/edge held-out metrics.
- Role: show that resistance-aware distances are not automatically positive.

Supplementary Figure S12: Liver/APAP task-design analysis.

- Suggested content: annotations, injury markers, central/periportal markers, annotation-aware endpoints.
- Role: support Figure 5C/E/F.

Supplementary Figure S13: Targeted multi-gene extension contact sheets.

- Suggested content: full-profile targeted fields across primary brain, sagittal brain, kidney, and liver/APAP exploratory datasets.
- Role: provide full visual provenance for targeted-extension summaries in Figures 3-5 and supplementary claim-boundary review.

Supplementary Figure S14: Marker-module and spot-level diagnostics.

- Suggested content: representative CNS-myelin leave-one-marker-out analysis and barrier-edge discordant spot examples.
- Role: support marker-module robustness and real-spot diagnostic checks; keep as representative supplementary evidence.

## Candidate Supplementary Tables

Supplementary Table S1: Dataset inventory.

- Source: `codexAnalysis/validation_dataset_inventory.md`
- Columns: dataset, accession/source, tissue, sample count, role, target genes, barrier markers, analysis status.

Supplementary Table S2: Parameter profiles.

- Source: `codexAnalysis/manuscript_text_drafts/methods_definitions_scalar_model.md`
- Columns: profile, Fourier features, width, depth, iterations, loss weights, intended use.

Supplementary Table S3: Metric definitions.

- Source: `codexAnalysis/barrier_metric_spec.md`
- Columns: metric, formula or operational definition, interpretation, caveat.

Supplementary Table S4: GSE193107 fitted-source and leakage summary.

- Source: Figure 2 and Figure 4 output tables.
- Columns: section, age group, target, profile, source Pearson, source MSE, roughness p95, leakage ratio.

Supplementary Table S5: Synthetic benchmark metrics.

- Source: Figure 3 output tables.
- Columns: simulation, method, barrier setting, MSE, Pearson, leakage, attenuation index.

Supplementary Table S6: Cross-tissue evidence summary.

- Source: `codexAnalysis/cross_tissue/evidence_summary/cross_tissue_evidence_summary_table.csv`
- Columns: tissue, task, baseline, method, Pearson delta, MSE delta, evidence role, interpretation.

Supplementary Table S7: Kidney marker screen.

- Source: `codexAnalysis/cross_tissue/mouse_kidney_10x/V1_Mouse_Kidney/marker_task_screen`
- Columns: target, barrier panel, compartment, expression score, specificity score, overlap/leakage scores, follow-up result.

Supplementary Table S8: Runtime and hardware.

- Source: runtime profiling logs and targeted-extension run-status provenance.
- Columns: hardware, PyTorch/CUDA version, profile, target, runtime, peak memory.

## Immediate Gaps To Fill Before Submission

- Confirm exact source CSV/TSV file paths for each plotted metric panel and list them in figure-specific provenance notes.
- Use the final-candidate Supplementary Figure S1-S14 folder unless upload-file size forces a split.
- Add full Supplementary Table S1-S8 files or explicitly mark unavailable tables as not included.
- Run a manuscript text QC pass using `codexAnalysis/manuscript_text_drafts/submission_claim_boundary_qc.md`.
