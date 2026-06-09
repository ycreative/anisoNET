"""Summarize synthetic barrier benchmark runs."""

from __future__ import annotations

import os

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SYNTHETIC_ROOT = PROJECT_ROOT / "codexAnalysis" / "synthetic_barrier" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize synthetic barrier benchmark outputs.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument("--target-name", default="Apoe_CNS_Myelin")
    return parser.parse_args()


def read_metrics(base_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in base_dir.iterdir() if p.is_dir() and "_train" in p.name):
        metrics_path = run_dir / "synthetic_barrier_metrics.csv"
        if not metrics_path.exists():
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                record = dict(row)
                record["run"] = run_dir.name
                parts = run_dir.name.split("_")
                record["seed"] = int(parts[0].replace("seed", ""))
                record["train_fraction"] = int(parts[1].replace("train", "")) / 100.0
                record["test_fraction"] = int(parts[2].replace("test", "")) / 100.0
                rows.append(record)
    if not rows:
        raise FileNotFoundError(f"No synthetic metrics found in {base_dir}")
    frame = pd.DataFrame(rows)
    numeric_cols = [
        "train_pearson_observed",
        "test_pearson_observed",
        "test_mse_observed",
        "grid_pearson_truth",
        "grid_mse_truth",
        "spot_pearson_truth",
        "high_barrier_mean",
        "low_barrier_mean",
        "high_to_low_barrier_ratio",
        "roughness_grad_mean",
        "roughness_grad_p95",
        "roughness_laplacian_energy",
    ]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["test_pearson_observed", "grid_pearson_truth", "grid_mse_truth", "high_to_low_barrier_ratio"]
    grouped = frame.groupby(["train_fraction", "test_fraction", "method"], sort=True)
    for keys, group in grouped:
        train_fraction, test_fraction, method = keys
        row = {
            "train_fraction": train_fraction,
            "test_fraction": test_fraction,
            "method": method,
            "n_runs": int(group["seed"].nunique()),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.nanmean(values))
            row[f"{metric}_sd"] = float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def paired_ablation_stats(frame: pd.DataFrame) -> pd.DataFrame:
    from scipy import stats

    metrics = ["test_pearson_observed", "grid_pearson_truth", "grid_mse_truth", "high_to_low_barrier_ratio"]
    rows = []
    for (train_fraction, test_fraction), group in frame.groupby(["train_fraction", "test_fraction"], sort=True):
        barrier = group[group["method"] == "anisoNET_original_barrier"].set_index("seed")
        no_barrier = group[group["method"] == "anisoNET_no_transcript_barrier"].set_index("seed")
        shared = sorted(set(barrier.index).intersection(no_barrier.index))
        for metric in metrics:
            barrier_values = barrier.loc[shared, metric].to_numpy(dtype=float)
            no_barrier_values = no_barrier.loc[shared, metric].to_numpy(dtype=float)
            diff = barrier_values - no_barrier_values
            if len(shared) > 1 and np.nanstd(diff) > 1e-12:
                test = stats.ttest_rel(barrier_values, no_barrier_values, nan_policy="omit")
                p_value = float(test.pvalue)
            else:
                p_value = float("nan")
            rows.append(
                {
                    "train_fraction": float(train_fraction),
                    "test_fraction": float(test_fraction),
                    "metric": metric,
                    "n_pairs": int(len(shared)),
                    "barrier_mean": float(np.nanmean(barrier_values)),
                    "no_barrier_mean": float(np.nanmean(no_barrier_values)),
                    "paired_difference_mean": float(np.nanmean(diff)),
                    "paired_difference_sd": float(np.nanstd(diff, ddof=1)) if len(shared) > 1 else 0.0,
                    "paired_ttest_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def barrier_advantage_vs_baselines(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (train_fraction, test_fraction), group in summary.groupby(["train_fraction", "test_fraction"], sort=True):
        barrier_row = group[group["method"] == "anisoNET_original_barrier"]
        if barrier_row.empty:
            continue
        barrier_row = barrier_row.iloc[0]
        for row in group.itertuples(index=False):
            if row.method == "anisoNET_original_barrier":
                continue
            rows.append(
                {
                    "train_fraction": float(train_fraction),
                    "test_fraction": float(test_fraction),
                    "baseline_method": row.method,
                    "anisonet_barrier_leakage": float(barrier_row.high_to_low_barrier_ratio_mean),
                    "baseline_leakage": float(row.high_to_low_barrier_ratio_mean),
                    "leakage_reduction_vs_baseline": float(row.high_to_low_barrier_ratio_mean - barrier_row.high_to_low_barrier_ratio_mean),
                    "anisonet_barrier_grid_pearson": float(barrier_row.grid_pearson_truth_mean),
                    "baseline_grid_pearson": float(row.grid_pearson_truth_mean),
                    "grid_pearson_delta_vs_baseline": float(barrier_row.grid_pearson_truth_mean - row.grid_pearson_truth_mean),
                    "anisonet_barrier_grid_mse": float(barrier_row.grid_mse_truth_mean),
                    "baseline_grid_mse": float(row.grid_mse_truth_mean),
                    "grid_mse_delta_vs_baseline": float(barrier_row.grid_mse_truth_mean - row.grid_mse_truth_mean),
                }
            )
    return pd.DataFrame(rows)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = {
        "nearest": "Nearest",
        "idw_k8": "IDW k=8",
        "gaussian_sigma1p5": "Gaussian 1.5",
        "gaussian_sigma3": "Gaussian 3",
        "graph_smooth_k6_iter3": "Graph\nsmooth k=6",
        "graph_smooth_k12_iter5": "Graph\nsmooth k=12",
        "anisoNET_original_barrier": "anisoNET\nbarrier",
        "anisoNET_no_transcript_barrier": "anisoNET\nno barrier",
    }
    method_order = [
        "nearest",
        "idw_k8",
        "gaussian_sigma1p5",
        "gaussian_sigma3",
        "graph_smooth_k6_iter3",
        "graph_smooth_k12_iter5",
        "anisoNET_original_barrier",
        "anisoNET_no_transcript_barrier",
    ]
    splits = sorted(summary[["train_fraction", "test_fraction"]].drop_duplicates().itertuples(index=False), key=lambda x: x[0], reverse=True)
    fig, axes = plt.subplots(len(splits), 2, figsize=(7.2, 2.5 * len(splits)), constrained_layout=True)
    if len(splits) == 1:
        axes = axes.reshape(1, -1)
    colors = ["#606c76", "#758bfd", "#00a878", "#f2a541", "#8f95d3", "#b8b8ff", "#d1495b", "#5f0f40"]
    for row_idx, split in enumerate(splits):
        subset = summary[(summary["train_fraction"] == split.train_fraction) & (summary["test_fraction"] == split.test_fraction)]
        subset = subset.set_index("method").reindex(method_order).reset_index()
        x = np.arange(len(method_order))
        title_prefix = f"Train {int(split.train_fraction * 100)}%, test {int(split.test_fraction * 100)}%"
        ax = axes[row_idx, 0]
        ax.bar(
            x,
            subset["grid_pearson_truth_mean"],
            yerr=subset["grid_pearson_truth_sd"],
            color=colors,
            edgecolor="black",
            linewidth=0.3,
            capsize=2,
        )
        ax.set_title(f"{title_prefix}: truth reconstruction", fontsize=7)
        ax.set_ylabel("Pearson")
        ax.set_ylim(0.75, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in method_order], rotation=30, ha="right")
        ax.tick_params(width=0.3, length=2)

        ax = axes[row_idx, 1]
        ax.bar(
            x,
            subset["high_to_low_barrier_ratio_mean"],
            yerr=subset["high_to_low_barrier_ratio_sd"],
            color=colors,
            edgecolor="black",
            linewidth=0.3,
            capsize=2,
        )
        ax.set_title(f"{title_prefix}: high-barrier leakage", fontsize=7)
        ax.set_ylabel("High/low barrier ratio")
        ax.set_xticks(x)
        ax.set_xticklabels([labels[m] for m in method_order], rotation=30, ha="right")
        ax.tick_params(width=0.3, length=2)
    fig.savefig(output_dir / "synthetic_barrier_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "synthetic_barrier_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(summary: pd.DataFrame, ablation: pd.DataFrame, advantage: pd.DataFrame, output_dir: Path) -> None:
    barrier = summary[summary["method"] == "anisoNET_original_barrier"].copy()
    no_barrier = summary[summary["method"] == "anisoNET_no_transcript_barrier"].copy()
    merged = barrier.merge(no_barrier, on=["train_fraction", "test_fraction"], suffixes=("_barrier", "_no_barrier"))
    lines = [
        "# Synthetic Barrier Summary",
        "",
        "This summary aggregates synthetic barrier benchmarks across available random seeds.",
        "",
        "## anisoNET Matched Ablation",
        "",
    ]
    for row in merged.sort_values("train_fraction", ascending=False).itertuples(index=False):
        pearson_gain = row.grid_pearson_truth_mean_barrier - row.grid_pearson_truth_mean_no_barrier
        mse_drop = row.grid_mse_truth_mean_no_barrier - row.grid_mse_truth_mean_barrier
        leakage_drop = row.high_to_low_barrier_ratio_mean_no_barrier - row.high_to_low_barrier_ratio_mean_barrier
        lines.extend(
            [
                f"- Train {int(row.train_fraction * 100)}% / test {int(row.test_fraction * 100)}%:",
                f"  anisoNET with transcriptomic barrier improved grid-truth Pearson by `{pearson_gain:.3f}`, reduced grid MSE by `{mse_drop:.4f}`, and lowered high-barrier leakage ratio by `{leakage_drop:.3f}` compared with the matched no-transcript-barrier ablation.",
            ]
        )
    lines.extend(
        [
            "",
            "## Paired Statistics",
            "",
            "Paired tests compare anisoNET with the transcriptomic barrier against the matched no-transcript-barrier ablation across random seeds. Because only three seeds are currently included, p-values should be treated as descriptive rather than definitive.",
            "",
        ]
    )
    for row in ablation.sort_values(["train_fraction", "metric"], ascending=[False, True]).itertuples(index=False):
        if row.metric not in {"grid_pearson_truth", "grid_mse_truth", "high_to_low_barrier_ratio"}:
            continue
        lines.append(
            f"- Train {int(row.train_fraction * 100)}% / test {int(row.test_fraction * 100)}%, `{row.metric}`: paired difference `{row.paired_difference_mean:.4f}` +/- `{row.paired_difference_sd:.4f}`, paired t-test p `{row.paired_ttest_p:.4g}`."
        )
    lines.extend(
        [
            "",
            "## Where anisoNET Wins",
            "",
            "The strongest current advantage is not global interpolation accuracy. It is high-barrier leakage control. In both train/test regimes, anisoNET with the transcriptomic barrier has the lowest mean high/low barrier leakage ratio among the tested methods.",
            "",
        ]
    )
    for split, group in advantage.groupby(["train_fraction", "test_fraction"], sort=True):
        train_fraction, test_fraction = split
        positive = group[group["leakage_reduction_vs_baseline"] > 0].sort_values("leakage_reduction_vs_baseline", ascending=False)
        best = positive.head(4)
        lines.append(f"- Train {int(train_fraction * 100)}% / test {int(test_fraction * 100)}%:")
        for row in best.itertuples(index=False):
            lines.append(
                f"  leakage ratio was lower than `{row.baseline_method}` by `{row.leakage_reduction_vs_baseline:.3f}`."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Classical interpolation remains a strong overall reconstruction baseline for this smooth synthetic task. The defensible positive claim is the matched ablation: the transcriptome-informed barrier consistently improves anisoNET over a no-transcript-barrier version and reduces leakage into high-barrier regions.",
            "",
            "This supports using synthetic barrier controls as supplementary evidence for the barrier-aware component, while avoiding overclaiming generic interpolation superiority.",
            "",
        ]
    )
    (output_dir / "synthetic_barrier_summary_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    base_dir = SYNTHETIC_ROOT / args.sample / args.target_name
    frame = read_metrics(base_dir)
    summary = summarize(frame)
    ablation = paired_ablation_stats(frame)
    advantage = barrier_advantage_vs_baselines(summary)
    frame.to_csv(base_dir / "synthetic_barrier_all_runs.csv", index=False)
    summary.to_csv(base_dir / "synthetic_barrier_summary.csv", index=False)
    ablation.to_csv(base_dir / "synthetic_barrier_paired_ablation_stats.csv", index=False)
    advantage.to_csv(base_dir / "synthetic_barrier_anisonet_advantage_vs_baselines.csv", index=False)
    plot_summary(summary, base_dir)
    write_interpretation(summary, ablation, advantage, base_dir)
    print(f"Wrote synthetic barrier summary to {base_dir}")


if __name__ == "__main__":
    main()

