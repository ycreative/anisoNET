"""Compute unified barrier-aware field metrics from existing codexAnalysis outputs."""

from __future__ import annotations

import os

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ROOT = PROJECT_ROOT / "codexAnalysis"
OUT_DIR = ROOT / "barrier_field_metrics"
BRAIN_BATCH = ROOT / "batch" / "brain_aging_gse193107"
BRAIN_PREFLIGHT = ROOT / "preflight" / "brain_aging_gse193107"
BRAIN_PINN = ROOT / "pinn" / "brain_aging_gse193107"
SYNTHETIC_ROOT = (
    ROOT
    / "synthetic_barrier"
    / "brain_aging_gse193107"
    / "GSM5773457_Old_mouse_brain_A1-2"
    / "Apoe_CNS_Myelin"
)


TARGETS = {
    "Apoe": "Apoe_CNS_Myelin",
    "Gfap": "Gfap_CNS_Myelin",
}
PINN_RUN = "fourier_refined_16g_gauss07_batch"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return float("nan")
    return float(num / den)


def mean_sem(values: pd.Series) -> tuple[float, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return float("nan"), float("nan")
    mean = float(clean.mean())
    sem = float(clean.std(ddof=1) / np.sqrt(len(clean))) if len(clean) > 1 else 0.0
    return mean, sem


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_brain_source_fidelity() -> pd.DataFrame:
    rows = []
    for gene, task in TARGETS.items():
        path = BRAIN_BATCH / task / "batch_metrics_summary.csv"
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            rows.extend(
                [
                    {
                        "dataset": "GSE193107",
                        "target": gene,
                        "sample": row["sample"],
                        "condition": row["condition"],
                        "field_type": row["field_type"],
                        "metric_family": "source_field_fidelity",
                        "metric": "source_pearson",
                        "value": row["spot_pearson_source"],
                        "direction": "higher_is_better",
                        "source_file": str(path.relative_to(PROJECT_ROOT)),
                        "candidate_figure": "Figure2",
                    },
                    {
                        "dataset": "GSE193107",
                        "target": gene,
                        "sample": row["sample"],
                        "condition": row["condition"],
                        "field_type": row["field_type"],
                        "metric_family": "source_field_fidelity",
                        "metric": "source_mse",
                        "value": row["spot_mse_source"],
                        "direction": "lower_is_better",
                        "source_file": str(path.relative_to(PROJECT_ROOT)),
                        "candidate_figure": "Figure2",
                    },
                    {
                        "dataset": "GSE193107",
                        "target": gene,
                        "sample": row["sample"],
                        "condition": row["condition"],
                        "field_type": row["field_type"],
                        "metric_family": "smoothness_diagnostic",
                        "metric": "roughness_grad_p95",
                        "value": row["roughness_grad_p95"],
                        "direction": "lower_is_smoother",
                        "source_file": str(path.relative_to(PROJECT_ROOT)),
                        "candidate_figure": "Figure2/Figure4",
                    },
                    {
                        "dataset": "GSE193107",
                        "target": gene,
                        "sample": row["sample"],
                        "condition": row["condition"],
                        "field_type": row["field_type"],
                        "metric_family": "barrier_control",
                        "metric": "spot_high_to_low_barrier_prediction_ratio",
                        "value": row["high_to_low_barrier_prediction_ratio"],
                        "direction": "task_dependent_lower_if_attenuation_expected",
                        "source_file": str(path.relative_to(PROJECT_ROOT)),
                        "candidate_figure": "Figure2/Figure3",
                    },
                ]
            )
    return pd.DataFrame(rows)


def compute_brain_grid_barrier_metrics() -> pd.DataFrame:
    rows = []
    for gene, task in TARGETS.items():
        manifest_path = BRAIN_BATCH / task / "batch_manifest.json"
        manifest = load_json(manifest_path)
        for sample in manifest["samples"]:
            condition = "Young" if "_Young_" in sample else "Old"
            preflight = BRAIN_PREFLIGHT / sample / task
            pinn = BRAIN_PINN / sample / task / PINN_RUN
            mask = np.asarray(np.load(preflight / "tissue_mask.npy")) > 0
            barrier = np.asarray(np.load(preflight / "barrier_grid.npy"), dtype=float)
            field_paths = {
                "raw": pinn / "pinn_grid_prediction_raw.npy",
                "clean": pinn / "pinn_grid_prediction_clean.npy",
                "norm": pinn / "pinn_grid_prediction_norm.npy",
                "postprocessed": pinn / "pinn_grid_prediction_postprocessed.npy",
            }
            tissue_barrier = barrier[mask]
            low_threshold = float(np.nanpercentile(tissue_barrier, 20))
            high_threshold = float(np.nanpercentile(tissue_barrier, 80))
            low_mask = mask & (barrier <= low_threshold)
            high_mask = mask & (barrier >= high_threshold)
            leakage_by_field_type: dict[str, float] = {}

            for field_type, field_path in field_paths.items():
                if not field_path.exists():
                    continue
                field = np.asarray(np.load(field_path), dtype=float)
                field_nonnegative = np.where(np.isfinite(field), np.maximum(field, 0), 0)
                total_mass = float(np.sum(field_nonnegative))
                outside_mass = float(np.sum(field_nonnegative[~mask]))
                tissue_mass = float(np.sum(field_nonnegative[mask]))
                tissue_leakage = safe_ratio(outside_mass, total_mass)
                leakage_by_field_type[field_type] = tissue_leakage
                high_mean = float(np.nanmean(field[high_mask])) if np.any(high_mask) else float("nan")
                low_mean = float(np.nanmean(field[low_mask])) if np.any(low_mask) else float("nan")
                tissue_mean = float(np.nanmean(field[mask])) if np.any(mask) else float("nan")
                rows.extend(
                    [
                        {
                            "dataset": "GSE193107",
                            "target": gene,
                            "sample": sample,
                            "condition": condition,
                            "field_type": field_type,
                            "metric_family": "barrier_control",
                            "metric": "grid_high_to_low_barrier_ratio",
                            "value": safe_ratio(high_mean, low_mean),
                            "direction": "task_dependent_lower_if_attenuation_expected",
                            "high_barrier_threshold": high_threshold,
                            "low_barrier_threshold": low_threshold,
                            "source_file": str(field_path.relative_to(PROJECT_ROOT)),
                            "candidate_figure": "Figure2/Figure3",
                        },
                        {
                            "dataset": "GSE193107",
                            "target": gene,
                            "sample": sample,
                            "condition": condition,
                            "field_type": field_type,
                            "metric_family": "barrier_control",
                            "metric": "tissue_support_leakage",
                            "value": tissue_leakage,
                            "direction": "lower_is_better",
                            "high_barrier_threshold": high_threshold,
                            "low_barrier_threshold": low_threshold,
                            "source_file": str(field_path.relative_to(PROJECT_ROOT)),
                            "candidate_figure": "Figure2/Figure4",
                        },
                        {
                            "dataset": "GSE193107",
                            "target": gene,
                            "sample": sample,
                            "condition": condition,
                            "field_type": field_type,
                            "metric_family": "field_distribution",
                            "metric": "tissue_field_mean",
                            "value": tissue_mean,
                            "direction": "descriptive",
                            "high_barrier_threshold": high_threshold,
                            "low_barrier_threshold": low_threshold,
                            "source_file": str(field_path.relative_to(PROJECT_ROOT)),
                            "candidate_figure": "Figure2",
                        },
                        {
                            "dataset": "GSE193107",
                            "target": gene,
                            "sample": sample,
                            "condition": condition,
                            "field_type": field_type,
                            "metric_family": "field_distribution",
                            "metric": "tissue_field_mass",
                            "value": tissue_mass,
                            "direction": "descriptive",
                            "high_barrier_threshold": high_threshold,
                            "low_barrier_threshold": low_threshold,
                            "source_file": str(field_path.relative_to(PROJECT_ROOT)),
                            "candidate_figure": "Figure2",
                        },
                    ]
                )
            if "raw" in leakage_by_field_type and "postprocessed" in leakage_by_field_type:
                raw_leakage = leakage_by_field_type["raw"]
                post_leakage = leakage_by_field_type["postprocessed"]
                reduction = (raw_leakage - post_leakage) / raw_leakage if raw_leakage and np.isfinite(raw_leakage) else np.nan
                rows.append(
                    {
                        "dataset": "GSE193107",
                        "target": gene,
                        "sample": sample,
                        "condition": condition,
                        "field_type": "raw_to_postprocessed",
                        "metric_family": "barrier_control",
                        "metric": "tissue_mask_leakage_reduction_index",
                        "value": reduction,
                        "direction": "higher_is_better",
                        "high_barrier_threshold": high_threshold,
                        "low_barrier_threshold": low_threshold,
                        "source_file": str((pinn / "pinn_grid_prediction_raw.npy").relative_to(PROJECT_ROOT))
                        + ";"
                        + str((pinn / "pinn_grid_prediction_postprocessed.npy").relative_to(PROJECT_ROOT)),
                        "candidate_figure": "Figure2/Figure4",
                    }
                )
    return pd.DataFrame(rows)


def compute_synthetic_metrics() -> pd.DataFrame:
    summary_path = SYNTHETIC_ROOT / "synthetic_barrier_summary.csv"
    paired_path = SYNTHETIC_ROOT / "synthetic_barrier_paired_ablation_stats.csv"
    summary = pd.read_csv(summary_path)
    paired = pd.read_csv(paired_path)
    rows = []

    metric_map = {
        "grid_pearson_truth_mean": ("source_field_fidelity", "grid_truth_pearson", "higher_is_better"),
        "grid_mse_truth_mean": ("source_field_fidelity", "grid_truth_mse", "lower_is_better"),
        "high_to_low_barrier_ratio_mean": ("barrier_control", "barrier_leakage_ratio", "lower_is_better"),
    }
    for _, row in summary.iterrows():
        split = f"train{int(row['train_fraction'] * 100)}_test{int(row['test_fraction'] * 100)}"
        for source_col, (family, metric, direction) in metric_map.items():
            rows.append(
                {
                    "dataset": "synthetic_barrier_GSE193107_oldA1_Apoe",
                    "target": "Apoe",
                    "sample": row["method"],
                    "condition": split,
                    "field_type": row["method"],
                    "metric_family": family,
                    "metric": metric,
                    "value": row[source_col],
                    "direction": direction,
                    "source_file": str(summary_path.relative_to(PROJECT_ROOT)),
                    "candidate_figure": "Figure3",
                }
            )

    for _, row in paired.iterrows():
        split = f"train{int(row['train_fraction'] * 100)}_test{int(row['test_fraction'] * 100)}"
        metric = str(row["metric"])
        if metric == "high_to_low_barrier_ratio":
            leakage_reference = float(row["no_barrier_mean"])
            leakage_barrier = float(row["barrier_mean"])
            attenuation = (leakage_reference - leakage_barrier) / leakage_reference if leakage_reference else np.nan
            rows.append(
                {
                    "dataset": "synthetic_barrier_GSE193107_oldA1_Apoe",
                    "target": "Apoe",
                    "sample": "paired_barrier_ablation",
                    "condition": split,
                    "field_type": "anisoNET_original_vs_no_transcript_barrier",
                    "metric_family": "barrier_control",
                    "metric": "barrier_attenuation_index",
                    "value": attenuation,
                    "direction": "higher_is_better",
                    "source_file": str(paired_path.relative_to(PROJECT_ROOT)),
                    "candidate_figure": "Figure3",
                }
            )
        direction = "higher_is_better" if "pearson" in metric else "lower_is_better"
        rows.append(
            {
                "dataset": "synthetic_barrier_GSE193107_oldA1_Apoe",
                "target": "Apoe",
                "sample": "paired_barrier_ablation",
                "condition": split,
                "field_type": "anisoNET_original_vs_no_transcript_barrier",
                "metric_family": "paired_ablation",
                "metric": f"paired_delta_{metric}",
                "value": row["paired_difference_mean"],
                "direction": direction,
                "source_file": str(paired_path.relative_to(PROJECT_ROOT)),
                "candidate_figure": "Figure3",
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "target", "condition", "field_type", "metric_family", "metric", "direction", "candidate_figure"]
    rows = []
    for keys, group in metrics.groupby(group_cols, dropna=False):
        mean, sem = mean_sem(group["value"])
        row = dict(zip(group_cols, keys))
        row["n"] = int(pd.to_numeric(group["value"], errors="coerce").notna().sum())
        row["value_mean"] = mean
        row["value_sem"] = sem
        row["source_files"] = ";".join(sorted(set(group["source_file"].dropna().astype(str))))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_markdown_summary(summary: pd.DataFrame, out_path: Path) -> None:
    interesting = summary[
        (
            (summary["dataset"] == "synthetic_barrier_GSE193107_oldA1_Apoe")
            & summary["metric"].isin(["barrier_attenuation_index", "paired_delta_high_to_low_barrier_ratio", "paired_delta_grid_mse_truth"])
        )
        | (
            (summary["dataset"] == "GSE193107")
            & summary["field_type"].isin(["raw", "postprocessed", "raw_to_postprocessed"])
            & summary["metric"].isin(
                ["grid_high_to_low_barrier_ratio", "tissue_support_leakage", "tissue_mask_leakage_reduction_index"]
            )
        )
    ].copy()
    lines = [
        "# Barrier-Aware Field Metric Summary",
        "",
        "This file is generated by `Script/workflows/compute_barrier_field_metrics.py`.",
        "",
        "Interpretation rules:",
        "",
        "- Pearson/source-fidelity metrics: higher is better.",
        "- MSE/leakage metrics: lower is better.",
        "- Barrier attenuation index: higher is better.",
        "- Generic held-out interpolation remains a claim-boundary benchmark, not the sole definition of success.",
        "",
        "## Selected Metrics",
        "",
    ]
    if interesting.empty:
        lines.append("No selected metrics found.")
    else:
        cols = ["dataset", "target", "condition", "field_type", "metric", "n", "value_mean", "value_sem", "direction"]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in interesting[cols].iterrows():
            values = []
            for col in cols:
                value = row[col]
                if isinstance(value, float):
                    values.append(f"{value:.4g}")
                else:
                    values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(out_path: Path) -> None:
    rows = [
        {
            "metric": "source_pearson/source_mse",
            "family": "source_field_fidelity",
            "source": "batch_metrics_summary.csv; synthetic_barrier_summary.csv",
            "candidate_figure": "Figure2/Figure3",
            "status": "implemented_initial",
        },
        {
            "metric": "grid_high_to_low_barrier_ratio",
            "family": "barrier_control",
            "source": "preflight barrier_grid.npy + PINN field NPY",
            "candidate_figure": "Figure2/Figure3",
            "status": "implemented_initial",
        },
        {
            "metric": "tissue_support_leakage",
            "family": "barrier_control",
            "source": "tissue_mask.npy + PINN field NPY",
            "candidate_figure": "Figure2/Figure4",
            "status": "implemented_initial",
        },
        {
            "metric": "tissue_mask_leakage_reduction_index",
            "family": "barrier_control",
            "source": "raw and postprocessed PINN field NPY",
            "candidate_figure": "Figure2/Figure4",
            "status": "implemented_initial",
        },
        {
            "metric": "barrier_attenuation_index",
            "family": "barrier_control",
            "source": "synthetic_barrier_paired_ablation_stats.csv",
            "candidate_figure": "Figure3",
            "status": "implemented_initial",
        },
        {
            "metric": "compartment_contrast_retention",
            "family": "annotation_boundary",
            "source": "kidney/liver annotation-boundary summaries",
            "candidate_figure": "Figure5",
            "status": "specified_not_yet_unified",
        },
        {
            "metric": "wrong_or_shuffled_barrier_delta",
            "family": "negative_control",
            "source": "future shuffled/wrong barrier benchmark",
            "candidate_figure": "Figure3/Figure5",
            "status": "specified_future_if_needed",
        },
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    ensure_dir(OUT_DIR)
    brain_source = compute_brain_source_fidelity()
    brain_grid = compute_brain_grid_barrier_metrics()
    synthetic = compute_synthetic_metrics()
    all_metrics = pd.concat([brain_source, brain_grid, synthetic], ignore_index=True)
    summary = summarize_metrics(all_metrics)

    brain_source.to_csv(OUT_DIR / "brain_source_fidelity_long.csv", index=False)
    brain_grid.to_csv(OUT_DIR / "brain_grid_barrier_metrics_long.csv", index=False)
    synthetic.to_csv(OUT_DIR / "synthetic_barrier_metrics_long.csv", index=False)
    all_metrics.to_csv(OUT_DIR / "barrier_field_metrics_long.csv", index=False)
    summary.to_csv(OUT_DIR / "barrier_field_metrics_summary.csv", index=False)
    write_manifest(OUT_DIR / "barrier_field_metric_manifest.csv")
    write_markdown_summary(summary, OUT_DIR / "barrier_field_metrics_summary.md")

    print(f"Wrote barrier-aware field metrics to {OUT_DIR}")
    print(f"Rows: long={len(all_metrics)}, summary={len(summary)}")


if __name__ == "__main__":
    main()

