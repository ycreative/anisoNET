# Main Figure Provenance For GPB Revision

Date: 2026-06-12

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
- `codexAnalysis/manuscript_figures/Figure1_method_overview/source_assets/Fig1D_continuous_resistance_field_schematic_v6.png`

Evidence role:

- Define the biological task, spot-level field-estimation step, scalar model implementation, continuous resistance mechanism, and representative output.
- Prevent overclaiming by explicitly describing the current implementation as scalar rather than tensor-valued or divergence-form.
- Panel letters are intentionally not embedded in generated panel assets; add final A-E labels manually during vector assembly.

## Figure 2: Primary GSE193107 Brain Application

Current assets:

- Directory: `codexAnalysis/manuscript_figures/Figure2_gse193107_primary_application`
- Panel list: `codexAnalysis/manuscript_figures/Figure2_gse193107_primary_application/CURRENT_PANEL_SET.md`
- Script: `Script/workflows/generate_figure2_panel_assets.py`

Current panels:

- `Fig2_v20260611_03_A_dataset_design`
- `Fig2_v20260611_03_B_representative_input_stack`
- `Fig2_v20260611_03_C_source_field_delta_barrier_overlay`
- `Fig2_v20260611_03_D_gene_section_qc_heatmap`
- `Fig2_v20260611_03_E_tissue_support_leakage`
- `Fig2_v20260611_03_F_barrier_context_summary`

Primary inputs:

- `codexAnalysis/batch/brain_aging_gse193107`
- `codexAnalysis/preflight/brain_aging_gse193107`
- `codexAnalysis/pinn/brain_aging_gse193107`
- `codexAnalysis/processed_visium/brain_aging_gse193107`
- `codexAnalysis/targeted_gene_extension/full_metrics_summary.csv`
- `codexAnalysis/targeted_gene_extension/full_metrics_by_dataset_gene.csv`
- `codexAnalysis/targeted_gene_extension/preflight/brain_aging_gse193107`
- `codexAnalysis/targeted_gene_extension/pinn/brain_aging_gse193107`

Evidence role:

- Main biological application evidence for the GSE193107 brain-aging workflow, with Apoe/Gfap anchors used to show source-to-field changes and CNS-myelin barrier contour context in a representative section.
- The targeted eight-gene brain panel is summarized as gene x section fitted-source/source-field QC across Apoe, Gfap, C1qa, Trem2, Tyrobp, Aif1, Cst3, and Lpl using existing targeted-extension outputs.
- The current tissue-support panel uses raw background field mass as a horizontal dot-interval summary plus a representative raw-field/tissue-mask/masked-field spatial example; the previous all-1 leakage-reduction index is no longer recommended as a main-panel display.
- The current barrier-context summary uses a horizontal lollipop/dot-interval display anchored at the 1.0 high/low barrier ratio reference line, avoiding the visually awkward compact four-column bar layout.
- Supports tissue-restricted fitted field reconstruction and fitted-source QC, not held-out prediction or generic superiority over all interpolation methods.
- Larger multi-gene spatial mosaics remain available as optional/supplementary Figure 2 assets but are no longer the recommended main Figure 2 core.
- The Figure 2 panel PDFs were regenerated with 600 dpi raster embedding and display-only upsampling/smoothing for spatial rasters so that Inkscape assembly does not inherit low-resolution embedded images.
- Use the `v20260611_03` versioned aliases for final Figure 2 assembly. The older Chinese-named manual SVG in the Figure 2 folder is a partial/stale layout with embedded A-G letters and older multi-gene spatial content, not the current final assembly source.

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
- `Fig3G_synthetic_seed_split_robustness`

Primary inputs:

- `codexAnalysis/heldout_benchmark`
- `codexAnalysis/generic_barrier_split_benchmark`
- `codexAnalysis/synthetic_barrier`
- `codexAnalysis/barrier_metric_spec.md`
- `codexAnalysis/targeted_gene_extension/full_metrics_by_dataset_gene.csv`

Evidence role:

- Separate generic held-out interpolation from controlled barrier-leakage validation.
- Synthetic panels provide the clearest mechanism evidence because source, barrier, and ground truth are controlled; synthetic map panels use display-only masked smoothing and upsampling to suppress Visium-grid artifacts without changing source arrays or metrics.
- Figure 3B/3C/3D masked rasters are pre-composited as opaque RGB on a white background before PDF export to prevent transparency-edge halo artifacts during Inkscape assembly/export.
- Figure 3B's synthetic truth display uses a hard-edged white-background render to avoid boundary interpolation artifacts; source arrays and synthetic benchmark metrics are unchanged.
- Figure 3C now focuses the representative synthetic map display on the stricter train 20% / test 80% stress case: synthetic truth, aNET with barrier, aNET without transcriptomic barrier, no-barrier excess, and high-barrier prior. The train 80% / test 20% versus train 20% / test 80% comparison remains in the quantitative Figure 3E and Figure 3G summaries rather than as visually near-duplicative map rows.
- Figure 3E now uses a method-by-metric heatmap; color encodes relative performance with MSE and leakage score-reversed while cell text preserves original metric values.
- Summary panels use the revised muted manuscript palette, with Figure 3F shown as a compact paired point summary and Figure 3G showing seed- and split-level barrier-vs-no-barrier advantages for truth Pearson, truth MSE, and leakage ratio.

## Figure 4: Robustness, Sensitivity, And Targeted Extension

Current assets:

- Directory: `codexAnalysis/manuscript_figures/Figure4_robustness_reproducibility`
- Panel list: `codexAnalysis/manuscript_figures/Figure4_robustness_reproducibility/CURRENT_PANEL_SET.md`
- Script: `Script/workflows/generate_figure4_panel_assets.py`

Current panels:

- `Fig4A_spatial_histology_prior_comparison`
- `Fig4B_spatial_low_pde_profile_sensitivity`
- `Fig4C_targeted_extension_multigene_spatial_summary`
- `Fig4D_low_pde_profile_delta`
- `Fig4E_source_clipping_sensitivity`
- `Fig4F_seed_stability`
- `Fig4G_targeted_extension_spatial_metrics`

Primary inputs:

- `codexAnalysis/histology_prior_comparison`
- `codexAnalysis/low_pde_profile_validation`
- `codexAnalysis/source_clipping_sensitivity`
- `codexAnalysis/low_pde_seed_stability`
- `codexAnalysis/batch/brain_aging_gse193107`
- `codexAnalysis/targeted_gene_extension/full_metrics_by_dataset.csv`
- `codexAnalysis/targeted_gene_extension/full_metrics_by_dataset_gene.csv`
- `codexAnalysis/targeted_gene_extension/preflight`
- `codexAnalysis/targeted_gene_extension/pinn`

Evidence role:

- Show sensitivity to histology priors, loss-weight profiles, source clipping, seed, and primary brain targeted-extension behavior.
- Figures 4A and 4B show source/barrier inputs as spot-level overlays to avoid misreading rasterized Visium grid artifacts as tissue texture.
- Figure 4C shows representative added-gene targeted-extension spatial fields and gene-level fitted-source QC for the primary GSE193107 brain-aging 8-gene set across eight sections.
- Figure 4G summarizes primary brain targeted-extension behavior at the section level, with aligned representative fields from one GSE193107 old A1 section and per-section fitted-source fidelity and roughness QC across eight genes.
- Runtime summaries from `codexAnalysis/resource_profile` are kept as provenance/supplementary material rather than an active main-figure claim.
- Frame low-PDE as a source-fit-optimized sensitivity profile rather than a universal replacement for the conservative default.
- Figure 4 panel PDFs were regenerated with explicit 600 dpi raster embedding after soft PDF imports were observed during manual assembly; this changes only export quality, not metrics or source arrays.
- Figure 4E is formatted as a tall, narrow two-row source-clipping sensitivity plot so it can be placed as a side-column panel during manual Inkscape assembly; source-clipping values are unchanged.
- The final assembled main Figure 4 contains panels A-G only. `Fig4H_profile_decision_table` remains a provenance/optional supplementary asset, not an active main-figure panel.

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
- `Fig5G_kidney_marker_extension`
- `Fig5H_cross_tissue_evidence_role_matrix`

Primary inputs:

- `codexAnalysis/cross_tissue/mouse_kidney_10x`
- `codexAnalysis/cross_tissue/liver_apap`
- `codexAnalysis/cross_tissue/sagittal_brain_10x`
- `codexAnalysis/cross_tissue/evidence_summary`
- `codexAnalysis/barrier_split_anisonet/mouse_kidney_10x`
- `codexAnalysis/targeted_gene_extension/full_metrics_by_dataset_gene.csv`

Evidence role:

- Demonstrate that preprocessing and barrier-prior construction are portable across tissues.
- Define the claim boundary: kidney supports benchmark development and Umod prior-development evidence; sagittal brain and liver/APAP emphasize task specificity and negative/mismatched controls.
- Figure 5G now uses the targeted kidney marker extension full-profile outputs for a compact fitted-source fidelity lollipop; the older marker-screen follow-up delta panel remains historical/provenance material rather than the recommended active panel.
- Figure 5H is now a graphical evidence-role matrix summarizing effect direction and evidence-row count across method-development, diagnostic, supplementary, claim-boundary, and negative-control roles; detailed row-level values remain in Supplementary Tables S6-S7 rather than the main panel.
- Spatial RGB context panels include high-expression marker-boundary contours on RGB composites only; single-gene maps remain uncluttered and all metrics are unchanged.
- Figure 5F uses a left-clipped display for the strongest healthy sagittal-brain negative-control effect (`Section2 Gfap`, Pearson delta `-0.081`) so smaller cross-tissue effects remain legible in the main figure; the true value is labeled on the panel and remains unchanged in the evidence table.
- Figure 5 panel PDFs were checked and regenerated with explicit 600 dpi raster embedding; no comparable low-dpi PDF issue was found in the active Figure 5 panel exporter.

## Final Assembly Notes

- Use PDF assets for vector-friendly Inkscape assembly whenever possible.
- Use PNG assets only for quick visual checking or when raster display is preferable.
- Rough assemblies are orientation drafts, not final submission figures.
- Current final-check assembly drafts are in `codexAnalysis/manuscript_figures/final_assembly_check`; these are review layouts built from the active panel PNGs, not a substitute for final journal vector polishing.
- Older panel files may remain in output folders for history; `CURRENT_PANEL_SET.md` defines the recommended active panel list.
