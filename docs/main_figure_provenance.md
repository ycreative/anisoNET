# Main Figure Provenance For GPB Revision

Date: 2026-06-08

Purpose:

- Provide a single audit trail for the current Figure 1-5 panel assets.
- Help final Inkscape assembly use the current panel set rather than older draft files that remain in the asset folders.
- Record which scripts and analysis outputs support each figure.

## Figure 1: Method Overview And Technical Definition

Current assets:

- Directory: `codexAnalysis/manuscript_figures/Figure1_method_overview`
- Panel list: `codexAnalysis/manuscript_figures/Figure1_method_overview/CURRENT_PANEL_SET.md`
- Script: `Script/workflows/generate_figure1_panel_assets.py`

Current panels:

- `Fig1A_data_and_task_definition`
- `Fig1B_field_construction`
- `Fig1C_scalar_pinn_architecture`
- `Fig1D_barrier_mechanism_explanation`
- `Fig1E_output_and_local_zoom`

Primary inputs:

- `codexAnalysis/preflight/brain_aging_gse193107/GSM5773457_Old_mouse_brain_A1-2/Apoe_CNS_Myelin`
- `codexAnalysis/pinn/brain_aging_gse193107/GSM5773457_Old_mouse_brain_A1-2/Apoe_CNS_Myelin/fourier_refined_16g_gauss07_batch`
- `codexAnalysis/processed_visium/brain_aging_gse193107/GSM5773457_Old_mouse_brain_A1-2/spatial`
- `codexAnalysis/manuscript_figures/Figure1_method_overview/source_assets/Fig1D_barrier_constrained_schematic_v2.png`

Evidence role:

- Define the biological task, scalar model implementation, barrier mechanism, and representative output.
- Prevent overclaiming by explicitly describing the current implementation as scalar rather than tensor-valued or divergence-form.
- Panel letters are intentionally not embedded in generated panel assets; add final A-E labels manually during vector assembly.

## Figure 2: Primary GSE193107 Brain Application

Current assets:

- Directory: `codexAnalysis/manuscript_figures/Figure2_gse193107_primary_application`
- Panel list: `codexAnalysis/manuscript_figures/Figure2_gse193107_primary_application/CURRENT_PANEL_SET.md`
- Script: `Script/workflows/generate_figure2_panel_assets.py`

Current panels:

- `Fig2A_dataset_design`
- `Fig2B_representative_input_stack`
- `Fig2C_Apoe_8section_fields`
- `Fig2D_Gfap_8section_fields`
- `Fig2E_source_fidelity_and_roughness`
- `Fig2F_tissue_support_leakage`
- `Fig2G_barrier_context_summary`

Primary inputs:

- `codexAnalysis/batch/brain_aging_gse193107`
- `codexAnalysis/preflight/brain_aging_gse193107`
- `codexAnalysis/pinn/brain_aging_gse193107`
- `codexAnalysis/processed_visium/brain_aging_gse193107`

Evidence role:

- Main biological application evidence for Apoe and Gfap fields across eight GSE193107 brain sections.
- Supports tissue-restricted fitted field reconstruction, not generic superiority over all interpolation methods.

## Figure 3: Benchmark And Synthetic Barrier Validation

Current assets:

- Directory: `codexAnalysis/manuscript_figures/Figure3_benchmark_and_synthetic_validation`
- Panel list: `codexAnalysis/manuscript_figures/Figure3_benchmark_and_synthetic_validation/CURRENT_PANEL_SET.md`
- Script: `Script/workflows/generate_figure3_panel_assets.py`

Current panels:

- `Fig3A_generic_heldout_benchmark`
- `Fig3B_synthetic_design`
- `Fig3C_synthetic_prediction_maps`
- `Fig3D_synthetic_error_and_leakage_maps`
- `Fig3E_synthetic_metric_summary`
- `Fig3F_paired_barrier_ablation`
- `Fig3G_barrier_attenuation_index`

Primary inputs:

- `codexAnalysis/heldout_benchmark`
- `codexAnalysis/generic_barrier_split_benchmark`
- `codexAnalysis/synthetic_barrier`
- `codexAnalysis/barrier_metric_spec.md`

Evidence role:

- Separate generic held-out interpolation from controlled barrier-leakage validation.
- Synthetic panels provide the clearest mechanism evidence because source, barrier, and ground truth are controlled.

## Figure 4: Robustness, Sensitivity, And Computational Profile

Current assets:

- Directory: `codexAnalysis/manuscript_figures/Figure4_robustness_reproducibility`
- Panel list: `codexAnalysis/manuscript_figures/Figure4_robustness_reproducibility/CURRENT_PANEL_SET.md`
- Script: `Script/workflows/generate_figure4_panel_assets.py`

Current panels:

- `Fig4A_spatial_histology_prior_comparison`
- `Fig4B_spatial_low_pde_profile_sensitivity`
- `Fig4C_histology_prior_metric_summary`
- `Fig4D_low_pde_profile_delta`
- `Fig4E_source_clipping_sensitivity`
- `Fig4F_seed_stability`
- `Fig4G_resource_profile`
- `Fig4H_profile_decision_table`

Primary inputs:

- `codexAnalysis/histology_prior_comparison`
- `codexAnalysis/low_pde_profile_validation`
- `codexAnalysis/source_clipping_sensitivity`
- `codexAnalysis/low_pde_seed_stability`
- `codexAnalysis/resource_profile`
- `codexAnalysis/batch/brain_aging_gse193107`

Evidence role:

- Show sensitivity to histology priors, loss-weight profiles, source clipping, seed, and compute resources.
- Frame low-PDE as a source-fit-optimized sensitivity profile rather than a universal replacement for the conservative default.

## Figure 5: Cross-Tissue Portability And Claim Boundary

Current assets:

- Directory: `codexAnalysis/manuscript_figures/Figure5_cross_tissue_boundary`
- Panel list: `codexAnalysis/manuscript_figures/Figure5_cross_tissue_boundary/CURRENT_PANEL_SET.md`
- Script: `Script/workflows/generate_figure5_panel_assets.py`

Current panels:

- `Fig5A_kidney_spatial_gene_context`
- `Fig5B_kidney_main_task_boundary`
- `Fig5C_liver_spatial_annotation_context`
- `Fig5D_sagittal_spatial_negative_control_context`
- `Fig5E_cross_tissue_task_matrix`
- `Fig5F_cross_tissue_transfer_delta_lollipop`
- `Fig5G_kidney_marker_screen_followup`
- `Fig5H_cross_tissue_evidence_role_matrix`

Primary inputs:

- `codexAnalysis/cross_tissue/mouse_kidney_10x`
- `codexAnalysis/cross_tissue/liver_apap`
- `codexAnalysis/cross_tissue/sagittal_brain_10x`
- `codexAnalysis/cross_tissue/evidence_summary`
- `codexAnalysis/barrier_split_anisonet/mouse_kidney_10x`

Evidence role:

- Demonstrate that preprocessing and barrier-prior construction are portable across tissues.
- Define the claim boundary: kidney supports benchmark development and Umod prior-development evidence; sagittal brain and liver/APAP emphasize task specificity and negative/mismatched controls.

## Final Assembly Notes

- Use PDF assets for vector-friendly Inkscape assembly whenever possible.
- Use PNG assets only for quick visual checking or when raster display is preferable.
- Rough assemblies are orientation drafts, not final submission figures.
- Older panel files may remain in output folders for history; `CURRENT_PANEL_SET.md` defines the recommended active panel list.
