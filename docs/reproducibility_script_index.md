# Reproducibility Script Index For GPB Revision

Date: 2026-06-08

Python environment:

- Use `K:\software\miniconda\envs\scvi_env\python.exe`.
- Do not rely on the default `python`, which may not include required scientific packages.

Purpose:

- List the scripts that generate the current manuscript analysis assets.
- Provide a practical audit trail for final Methods, Code availability, and Supplementary Materials.
- Avoid rerunning the full project from scratch unless a specific output needs regeneration.

## Dataset Standardization And Preflight Construction

GSE193107 brain aging:

- `Script/workflows/standardize_gse193107_visium.py`
- `Script/workflows/batch_histology_prior_preflight.py`
- `Script/workflows/run_anisonet_preflight.py`

Public 10x and prefixed GEO Visium datasets:

- `Script/workflows/standardize_public_10x_visium.py`
- `Script/workflows/standardize_prefixed_geo_visium.py`

Gene/task surveys:

- `Script/workflows/survey_gse193107_genes.py`
- `Script/workflows/survey_visium_genes.py`
- `Script/workflows/find_barrier_edge_discordant_spots.py`

## Primary anisoNET Runs And Postprocessing

Core fitting:

- `Script/workflows/run_anisonet_pinn.py`
- `Script/workflows/batch_gse193107_anisonet.py`
- `Script/workflows/sweep_anisonet_fourier.py`

Postprocessing and field evaluation:

- `Script/workflows/postprocess_anisonet_field.py`
- `Script/workflows/smooth_anisonet_field.py`
- `Script/workflows/evaluate_anisonet_field.py`
- `Script/workflows/compute_barrier_field_metrics.py`
- `Script/workflows/collect_anisonet_metrics.py`
- `Script/workflows/summarize_anisonet_batch.py`
- `Script/workflows/compare_anisonet_target_batches.py`

## Benchmarks And Mechanism Tests

Generic held-out and barrier split benchmarks:

- `Script/workflows/heldout_gse193107_benchmark.py`
- `Script/workflows/summarize_heldout_benchmark.py`
- `Script/workflows/generic_barrier_split_anisonet_benchmark.py`
- `Script/workflows/summarize_generic_barrier_split_benchmark.py`
- `Script/workflows/barrier_aware_spot_interpolation_benchmark.py`
- `Script/workflows/build_anisonet_baselines.py`

Synthetic and perturbation tests:

- `Script/workflows/synthetic_barrier_benchmark.py`
- `Script/workflows/summarize_synthetic_barrier.py`
- `Script/workflows/barrier_perturbation_benchmark.py`
- `Script/workflows/plot_barrier_perturbation.py`
- `Script/workflows/alpha_sensitivity_benchmark.py`
- `Script/workflows/summarize_alpha_sensitivity.py`
- `Script/workflows/loss_weight_sensitivity_benchmark.py`
- `Script/workflows/summarize_loss_weight_sensitivity.py`
- `Script/workflows/leave_one_marker_out_benchmark.py`
- `Script/workflows/summarize_leave_one_marker_out.py`

## Robustness, Sensitivity, And Resource Profiling

Histology prior comparison:

- `Script/workflows/compare_histology_priors.py`
- `Script/workflows/batch_histology_prior_pinn.py`
- `Script/workflows/summarize_histology_prior_preflight.py`
- `Script/workflows/summarize_histology_prior_pinn.py`

Low-PDE and clipping sensitivity:

- `Script/workflows/batch_low_pde_profile_validation.py`
- `Script/workflows/summarize_low_pde_profile_validation.py`
- `Script/workflows/low_pde_seed_stability_benchmark.py`
- `Script/workflows/summarize_low_pde_seed_stability.py`
- `Script/workflows/source_clipping_sensitivity_benchmark.py`
- `Script/workflows/summarize_source_clipping_sensitivity.py`

Compute profile:

- `Script/workflows/profile_anisonet_resources.py`
- `Script/workflows/summarize_resource_profile.py`
- `Script/workflows/compare_resource_profiles.py`
- `Script/workflows/summarize_pinn_convergence.py`

## Cross-Tissue And Claim-Boundary Analyses

Kidney:

- `Script/workflows/kidney_barrier_split_profile_probe.py`
- `Script/workflows/kidney_evidence_boundary_diagnostics.py`
- `Script/workflows/kidney_grid_geodesic_prior_benchmark.py`
- `Script/workflows/kidney_prior_hybrid_blend_analysis.py`
- `Script/workflows/kidney_prior_hybrid_sweep.py`
- `Script/workflows/kidney_marker_task_screen.py`

Liver/APAP:

- `Script/workflows/liver_annotation_boundary_benchmark.py`
- `Script/workflows/liver_apap_annotation_field_summary.py`

Cross-tissue summary:

- `Script/workflows/cross_tissue_evidence_summary.py`

## Targeted Multi-Gene Extension

Added 2026-06-10:

- `Script/workflows/select_target_gene_extension.py`
- `Script/workflows/run_targeted_gene_extension.py`
- `Script/workflows/summarize_targeted_gene_extension.py`
- `Script/workflows/visualize_targeted_gene_extension_fields.py`

Primary outputs:

- `codexAnalysis/targeted_gene_extension/gene_selection_rules.md`
- `codexAnalysis/targeted_gene_extension/selected_gene_manifest.csv`
- `codexAnalysis/targeted_gene_extension/run_manifest.csv`
- `codexAnalysis/targeted_gene_extension/full_metrics_summary.csv`
- `codexAnalysis/targeted_gene_extension/full_metrics_by_dataset_gene.csv`
- `codexAnalysis/targeted_gene_extension/full_run_interpretation.md`
- `codexAnalysis/targeted_gene_extension/field_contact_sheets`

Notes:

- The extension is a targeted a priori screen, not an all-gene discovery analysis.
- Use full-profile `fourier_refined_low_pde_16g` outputs for figure decisions.
- Treat smoke outputs as execution checks only.
- Interpret source-fidelity metrics as fitted-source agreement, not held-out prediction.

## Figure Asset Generation

Current manuscript panel workflows:

- `Script/workflows/generate_figure1_panel_assets.py`
- `Script/workflows/generate_figure2_panel_assets.py`
- `Script/workflows/generate_figure3_panel_assets.py`
- `Script/workflows/generate_figure4_panel_assets.py`
- `Script/workflows/generate_figure5_panel_assets.py`

Current panel directories:

- `codexAnalysis/manuscript_figures/Figure1_method_overview`
- `codexAnalysis/manuscript_figures/Figure2_gse193107_primary_application`
- `codexAnalysis/manuscript_figures/Figure3_benchmark_and_synthetic_validation`
- `codexAnalysis/manuscript_figures/Figure4_robustness_reproducibility`
- `codexAnalysis/manuscript_figures/Figure5_cross_tissue_boundary`

Assembly and legacy helpers:

- `Script/workflows/assemble_main_figures.py`
- `Script/workflows/assemble_figure1_method_overview_gpb.py`
- `Script/workflows/draw_figure1_method_overview.py`
- `Script/workflows/render_pdf_pages.py`

## Manuscript, Supplementary Tables, And QC

Manuscript and response export:

- `Script/workflows/build_integrated_manuscript_docx.py`
- `Script/workflows/export_manuscript_drafts_to_docx.py`

Supplementary table export:

- `Script/workflows/export_supplementary_tables_xlsx.py`

Claim-boundary QC:

- `Script/workflows/manuscript_claim_qc.py`

Current manuscript and table outputs:

- `draft/revised/anisoNET_GPB_integrated_manuscript_v8.docx`
- `draft/revised/anisoNET_GPB_cover_letter_conservative_v1.docx`
- `draft/revised/anisoNET_GPB_response_to_reviewers_v2_template.docx`
- `draft/revised/anisoNET_GPB_supplementary_figure_legends_final_candidate.docx`
- `draft/revised/anisoNET_GPB_supplementary_tables_submission_candidate.xlsx`
- `draft/revised/response_to_reviewers_working_draft.docx`
- `codexAnalysis/manuscript_figures/supplementary_final_candidate`

## Reproduction Notes

- The current working package was generated incrementally; do not rerun every upstream benchmark by default.
- Regenerate individual figure panels only when the underlying corresponding analysis output changes.
- Before submission, rerun only:
  - the relevant figure-generation script after final data changes;
  - `export_supplementary_tables_xlsx.py` after table edits;
  - `build_integrated_manuscript_docx.py` after Markdown text edits;
  - `manuscript_claim_qc.py` after final Word edits.
- Final Word formatting, references, and Inkscape figure assembly remain manual submission steps.

## Public Reproducibility Package Draft

Added 2026-06-09:

- `codexAnalysis/reproducibility_package_plan.md`
- `codexAnalysis/reproducibility_package/README.md`
- `codexAnalysis/reproducibility_package/environment.yml`
- `codexAnalysis/reproducibility_package/requirements.txt`
- `codexAnalysis/reproducibility_package/reproduce_figures.md`
- `codexAnalysis/reproducibility_package/repository_manifest.md`
- `codexAnalysis/reproducibility_package/data_and_code_availability.md`
- `codexAnalysis/reproducibility_package/zenodo_release_checklist.md`
- `codexAnalysis/reproducibility_package/submission_release_decision.md`
- `Script/configs/anisonet_datasets.example.json`

Current release cleanup target:

- Figure 1-5 panel scripts now support repository-root inference and `ANISONET_PROJECT_ROOT` / `ANISONET_ANALYSIS_ROOT` overrides.
- Several upstream batch and summary scripts still contain local development defaults and should be parameterized before public GitHub release.
- The local dataset config should remain private; publish the sanitized example config instead.
