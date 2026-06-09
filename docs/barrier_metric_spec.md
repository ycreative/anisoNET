# Barrier-Aware Spatial Field Metric Specification

Date: 2026-06-08

Purpose:

- Define reproducible metrics for the revised anisoNET task: barrier-aware spatial field inference.
- Separate generic source-field fidelity from barrier-control behavior.
- Prevent metric shopping by specifying endpoint families, directionality, inputs, and manuscript role before adding new figure panels.

## Task Definition

Barrier-aware spatial field inference estimates a tissue-constrained field `C(x,y)` from a source signal `S(x,y)`, tissue support `T(x,y)`, histology-derived structural resistance, and transcriptomic barrier prior `B(x,y)`.

The task is not identical to generic held-out spot interpolation. A useful model should:

- preserve source-field fidelity where the source is biologically supported;
- reduce inappropriate propagation across high-resistance or high-barrier regions;
- stay constrained to tissue support;
- remain robust to reasonable parameter and seed changes.

## Endpoint Families

### Primary Endpoint Family 1: Source-Field Fidelity

Question:

- Does the predicted field remain consistent with the observed or synthetic source signal?

Metrics:

- `source_pearson`: Pearson correlation between predicted spot-level field and source expression.
- `source_mse`: mean squared error between predicted spot-level field and source expression.
- `grid_truth_pearson`: Pearson correlation between predicted grid field and synthetic truth.
- `grid_truth_mse`: grid-level MSE against synthetic truth.

Directionality:

- Pearson: higher is better.
- MSE: lower is better.

Primary sources:

- `codexAnalysis/batch/brain_aging_gse193107/*/batch_metrics_summary.csv`
- `codexAnalysis/heldout/brain_aging_gse193107/summary/heldout_group_summary.csv`
- `codexAnalysis/synthetic_barrier/.../synthetic_barrier_summary.csv`

Manuscript role:

- Use to show that anisoNET does not simply suppress signal to reduce leakage.
- Use generic held-out fidelity results as claim-boundary evidence when Gaussian/IDW baselines win.

### Primary Endpoint Family 2: Barrier-Control Performance

Question:

- Does the predicted field reduce inappropriate propagation across barrier-enriched regions?

Metrics:

- `barrier_leakage_ratio`: mean predicted field in high-barrier regions divided by mean predicted field in low-barrier/source-accessible regions.
- `barrier_attenuation_index`: fractional leakage reduction relative to a no-barrier, wrong-barrier, or shuffled-barrier model.
- `paired_barrier_delta`: paired difference between full-barrier and no-barrier/shuffled-barrier models for leakage, MSE, or Pearson.
- `tissue_support_leakage`: field mass outside tissue support divided by total field mass.

Suggested definitions:

```text
barrier_leakage_ratio = mean(C in high-barrier region) / mean(C in low-barrier or source-accessible region)
barrier_attenuation_index = (leakage_reference - leakage_barrier_model) / leakage_reference
tissue_support_leakage = sum(C outside tissue mask) / sum(C everywhere)
```

Directionality:

- `barrier_leakage_ratio`: lower is better when the task is attenuation across high-barrier regions.
- `barrier_attenuation_index`: higher is better.
- `paired_barrier_delta`: direction depends on metric; lower leakage/MSE and higher Pearson are favorable.
- `tissue_support_leakage`: lower is better.

Primary sources:

- `codexAnalysis/synthetic_barrier/.../synthetic_barrier_summary.csv`
- `codexAnalysis/synthetic_barrier/.../synthetic_barrier_paired_ablation_stats.csv`
- grid arrays under `codexAnalysis/preflight/brain_aging_gse193107`
- predicted fields under `codexAnalysis/pinn/brain_aging_gse193107`

Manuscript role:

- Use as the core task-specific evidence for anisoNET.
- These metrics are more aligned with the revised task than random held-out interpolation.

## Secondary Diagnostic Endpoints

### Compartment Contrast Retention

Question:

- Does the predicted field preserve known anatomical, compartment, or annotation contrasts?

Suggested use:

- Kidney tubule-compartment targets.
- Liver periportal/central annotation-boundary tasks.

Directionality:

- Higher retention is better when the target has known compartment enrichment.

Sources:

- kidney evidence-boundary summaries;
- liver/APAP annotation-boundary summaries;
- future annotation-aware tables if generated.

Manuscript role:

- Use mainly in Figure 5 and Supplementary.

### Source Fidelity Versus Smoothness Pareto

Question:

- Does a method achieve source fidelity without excessive roughness, over-smoothing, or leakage?

Metrics:

- source Pearson or MSE;
- roughness p95 or mean gradient;
- leakage ratio if available.

Directionality:

- Prefer high fidelity, low roughness, and low leakage.
- Do not collapse these into an opaque composite score.

Sources:

- brain batch metrics;
- held-out benchmark tables;
- sensitivity summaries.

Manuscript role:

- Use to explain why a model setting can be source-fit optimized but not automatically a universal default.

### Wrong-Barrier Or Shuffled-Barrier Negative Control

Question:

- Does barrier-control performance depend on meaningful barrier structure?

Benchmark variants:

- true transcriptomic barrier;
- no transcriptomic barrier;
- shuffled barrier;
- biologically mismatched marker barrier.

Directionality:

- True barrier should improve leakage/attenuation endpoints relative to wrong or shuffled barriers.

Status:

- `no transcriptomic barrier` is already available in synthetic benchmarks.
- Shuffled/wrong-barrier controls should be added only if needed for a main figure or reviewer response.

## Benchmark Families

### Generic Held-Out Benchmark

Purpose:

- Fairly compare ordinary interpolation behavior.

Role:

- Claim-boundary benchmark.
- Do not suppress negative results where Gaussian/IDW baselines win.

### Barrier-Stress Held-Out Benchmark

Purpose:

- Evaluate spots near high-barrier or compartment-boundary regions.

Role:

- Better aligned with the method's intended operating condition than random held-out splits.

### Synthetic Barrier Benchmark

Purpose:

- Mechanistically validate leakage control under known truth.

Role:

- Primary controlled evidence for barrier-aware behavior.

### Annotation-Boundary Benchmark

Purpose:

- Test whether predictions respect known tissue annotations or compartments.

Role:

- Cross-tissue validation-design evidence.

### Ablation Benchmark

Purpose:

- Attribute performance to the transcriptomic barrier, histology prior, tissue mask, prior-only baseline, or PINN refinement.

Role:

- Prevents black-box claims.

## Reporting Rules

- State directionality for every metric.
- Report generic held-out benchmarks even when anisoNET is not the best method.
- Do not use a single composite superiority score as the main result.
- Every metric must be tied to a figure panel, table, or manuscript sentence.
- Every metric must be reproducible from saved files under `codexAnalysis`.
- Prefer paired comparisons for ablations.
- Report both mean effects and variability when replicate seeds, sections, or train/test splits are available.

## Initial Implementation Scope

The first unified metric workflow should generate:

- brain GSE193107 source-fidelity and roughness summaries from batch CSVs;
- brain GSE193107 barrier-quantile field summaries from preflight/PINN NPY arrays;
- tissue-support leakage for representative PINN fields;
- synthetic barrier leakage, attenuation, and paired ablation summaries from existing CSVs;
- a compact metric manifest mapping each metric to source files and candidate figure panels.
