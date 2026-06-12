# Final Main Figure Manifest

Date: 2026-06-12

Purpose:

- Record the final assembled Figure 1-5 files exported from Inkscape/manual assembly.
- Map each final figure to the active panel assets that the manuscript text and legends should describe.
- Prevent older provenance panels from being mistaken for active main-figure panels.

## Final Assembled Figure Files

Use the following PDFs for the current main-figure package:

| Figure | Final PDF | Final PNG / assembly source | Notes |
|---|---|---|---|
| Figure 1 | `codexAnalysis/manuscript_figures/Figure1_method_overview/Figure1.pdf` | `codexAnalysis/manuscript_figures/Figure1_method_overview/Figure1.png` copied from `text1.png` | White-background flattened 600 dpi PDF. |
| Figure 2 | `codexAnalysis/manuscript_figures/Figure2_gse193107_primary_application/Figure2.pdf` | `codexAnalysis/manuscript_figures/Figure2_gse193107_primary_application/Figure2.png` copied from `text1.png` | Uses `v20260611_03` panel set. |
| Figure 3 | `codexAnalysis/manuscript_figures/Figure3_benchmark_and_synthetic_validation/Figure3.pdf` | `codexAnalysis/manuscript_figures/Figure3_benchmark_and_synthetic_validation/Figure3.png` copied from `text1.png` | White-background flattened 600 dpi PDF. |
| Figure 4 | `codexAnalysis/manuscript_figures/Figure4_robustness_reproducibility/Figure4.pdf` | `codexAnalysis/manuscript_figures/Figure4_robustness_reproducibility/Figure4.png` copied from the current manual export `figure5.png` in the Figure 4 folder | Source filename was historically misleading; the folder and content are Figure 4. |
| Figure 5 | `codexAnalysis/manuscript_figures/Figure5_cross_tissue_boundary/Figure5.pdf` | `codexAnalysis/manuscript_figures/Figure5_cross_tissue_boundary/figure5.png` | Windows treats `figure5.png` and `Figure5.png` as the same filename; the PDF uses standard `Figure5.pdf` naming. |

## Active Panel Mapping

### Figure 1: Method Overview And Technical Definition

- A: `Fig1A_data_and_task_definition`
- B: `Fig1B_field_construction`
- C: `Fig1C_scalar_pinn_architecture`
- D: `Fig1D_barrier_mechanism_explanation`
- E: `Fig1E_output_and_local_zoom`

Text implications:

- Figure 1 has panels A-E only.
- The method must be described as scalar barrier-aware field inference, not tensor-valued diffusion or a full divergence-form variable-coefficient PDE.

### Figure 2: Primary GSE193107 Brain Application

- A: `Fig2_v20260611_03_A_dataset_design`
- B: `Fig2_v20260611_03_B_representative_input_stack`
- C: `Fig2_v20260611_03_C_source_field_delta_barrier_overlay`
- D: `Fig2_v20260611_03_D_gene_section_qc_heatmap`
- E: `Fig2_v20260611_03_E_tissue_support_leakage`
- F: `Fig2_v20260611_03_F_barrier_context_summary`

Text implications:

- Figure 2 has panels A-F only.
- Figure 2E is raw background field mass plus a raw-field/tissue-mask/masked-field spatial example.
- Figure 2F is the high/low barrier-context lollipop/dot-interval summary.
- Fitted-source Pearson is source-fit QC, not held-out prediction.

### Figure 3: Benchmark And Synthetic Barrier Validation

- A: `Fig3A_generic_heldout_benchmark`
- B: `Fig3B_synthetic_design`
- C: `Fig3C_synthetic_prediction_maps`
- D: `Fig3D_synthetic_error_and_leakage_maps`
- E: `Fig3E_synthetic_metric_summary`
- F: `Fig3F_paired_barrier_ablation`
- G: `Fig3G_synthetic_seed_split_robustness`

Text implications:

- Figure 3A defines the generic interpolation claim boundary.
- Figure 3C shows the stricter train 20% / test 80% stress case as a single representative synthetic map row.
- Split- and seed-level robustness should be described through Figure 3E and Figure 3G.

### Figure 4: Robustness, Sensitivity, And Targeted Extension

- A: `Fig4A_spatial_histology_prior_comparison`
- B: `Fig4B_spatial_low_pde_profile_sensitivity`
- C: `Fig4C_targeted_extension_multigene_spatial_summary`
- D: `Fig4D_low_pde_profile_delta`
- E: `Fig4E_source_clipping_sensitivity`
- F: `Fig4F_seed_stability`
- G: `Fig4G_targeted_extension_spatial_metrics`

Text implications:

- Figure 4 has panels A-G only in the final assembled main figure.
- Runtime is not Figure 4G.
- Figure 4E is the tall/narrow source-clipping sensitivity panel.
- Figure 4G is primary brain targeted-extension fitted-source fidelity and roughness QC.
- Low-PDE should be framed as a source-fit-optimized sensitivity profile, not a universal replacement for the conservative default.
- `Fig4H_profile_decision_table` remains a provenance/optional supplementary asset only; it is not in the final main Figure 4.

### Figure 5: Cross-Tissue Portability And Claim Boundary

- A: `Fig5A_kidney_spatial_gene_context`
- B: `Fig5B_kidney_main_task_boundary`
- C: `Fig5C_liver_spatial_annotation_context`
- D: `Fig5D_sagittal_spatial_negative_control_context`
- E: `Fig5E_cross_tissue_task_matrix`
- F: `Fig5F_cross_tissue_transfer_delta_lollipop`
- G: `Fig5G_kidney_marker_extension`
- H: `Fig5H_cross_tissue_evidence_role_matrix`

Text implications:

- Figure 5G is the targeted kidney marker-extension summary using full-profile fitted-source agreement, not the older marker-screen follow-up panel.
- Figure 5H is the graphical evidence-role matrix.
- Figure 5F clips the strongest sagittal-brain `Section2 Gfap` negative-control point for display only and labels its true value.

## Deprecated Or Provenance-Only Main-Figure Assets

- Do not use the older Figure 2 `Fig2F_tissue_support_leakage` or `Fig2G_barrier_context_summary` files in final assembly.
- Do not use older Figure 4 runtime/resource panels as Figure 4G.
- Do not describe fitted-source Pearson as held-out prediction.
- Do not add panel letters inside regenerated panel assets; final assembled figures may carry panel letters at the assembly layer.
