"""Summarize multi-section default vs low-PDE profile validation."""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "loss_weight_sensitivity" / "brain_aging_gse193107" / "multi_section_low_pde"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize default vs low-PDE validation.")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    frame = pd.read_csv(output_dir / "low_pde_profile_validation_metrics.csv")
    delta = compute_delta(frame)
    paired = summarize_paired(delta)
    group = summarize_group(frame, delta)

    delta.to_csv(output_dir / "low_pde_profile_validation_delta_from_default.csv", index=False)
    paired.to_csv(output_dir / "low_pde_profile_validation_paired_summary.csv", index=False)
    group.to_csv(output_dir / "low_pde_profile_validation_group_summary.csv", index=False)
    plot_summary(delta, paired, output_dir)
    write_interpretation(frame, delta, paired, group, output_dir)
    print(f"Wrote low-PDE profile validation summary to {output_dir}", flush=True)


def compute_delta(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "roughness_grad_mean",
        "high_to_low_barrier_prediction_ratio",
        "spot_pearson_barrier",
    ]
    reference = frame[frame["setting"] == "default"].set_index(["sample", "target_gene", "field_type"])
    rows = []
    for row in frame[frame["setting"] == "low_pde"].itertuples(index=False):
        base = reference.loc[(row.sample, row.target_gene, row.field_type)]
        record = {
            "sample": row.sample,
            "condition": row.condition,
            "target_gene": row.target_gene,
            "field_type": row.field_type,
            "histology_prior": row.histology_prior,
            "setting": row.setting,
            "data_weight": float(row.data_weight),
            "pde_weight": float(row.pde_weight),
            "data_to_pde_weight_ratio": float(row.data_to_pde_weight_ratio),
        }
        for metric in metrics:
            value = float(getattr(row, metric))
            base_value = float(base[metric])
            record[metric] = value
            record[f"{metric}_default"] = base_value
            record[f"{metric}_delta_from_default"] = value - base_value
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_paired(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "spot_pearson_source_delta_from_default",
        "spot_mse_source_delta_from_default",
        "roughness_grad_p95_delta_from_default",
        "roughness_grad_mean_delta_from_default",
        "high_to_low_barrier_prediction_ratio_delta_from_default",
    ]
    for (target_gene, field_type), group in delta.groupby(["target_gene", "field_type"], sort=True):
        record = {
            "target_gene": target_gene,
            "field_type": field_type,
            "n_samples": int(len(group)),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_median"] = float(np.median(values))
            record[f"{metric}_sem"] = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            record[f"{metric}_n_positive"] = int(np.sum(values > 0))
            record[f"{metric}_n_negative"] = int(np.sum(values < 0))
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_group(frame: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target_gene, field_type, setting), group in frame.groupby(["target_gene", "field_type", "setting"], sort=True):
        rows.append(
            {
                "summary_type": "absolute",
                "target_gene": target_gene,
                "field_type": field_type,
                "setting": setting,
                "n_samples": int(len(group)),
                "spot_pearson_source_mean": group["spot_pearson_source"].mean(),
                "spot_pearson_source_sem": group["spot_pearson_source"].std(ddof=1) / np.sqrt(len(group)),
                "spot_mse_source_mean": group["spot_mse_source"].mean(),
                "roughness_grad_p95_mean": group["roughness_grad_p95"].mean(),
                "high_to_low_barrier_prediction_ratio_mean": group["high_to_low_barrier_prediction_ratio"].mean(),
            }
        )
    for (target_gene, field_type), group in delta.groupby(["target_gene", "field_type"], sort=True):
        rows.append(
            {
                "summary_type": "delta_low_pde_minus_default",
                "target_gene": target_gene,
                "field_type": field_type,
                "setting": "low_pde_minus_default",
                "n_samples": int(len(group)),
                "spot_pearson_source_mean": group["spot_pearson_source_delta_from_default"].mean(),
                "spot_pearson_source_sem": group["spot_pearson_source_delta_from_default"].std(ddof=1) / np.sqrt(len(group)),
                "spot_mse_source_mean": group["spot_mse_source_delta_from_default"].mean(),
                "roughness_grad_p95_mean": group["roughness_grad_p95_delta_from_default"].mean(),
                "high_to_low_barrier_prediction_ratio_mean": group[
                    "high_to_low_barrier_prediction_ratio_delta_from_default"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(delta: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
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
    metrics = [
        ("spot_pearson_source_delta_from_default", "Delta source Pearson"),
        ("spot_mse_source_delta_from_default", "Delta source MSE"),
        ("roughness_grad_p95_delta_from_default", "Delta roughness p95"),
        ("high_to_low_barrier_prediction_ratio_delta_from_default", "Delta high/low barrier ratio"),
    ]
    targets = ["Apoe", "Gfap"]
    fields = ["masked", "gauss07"]
    colors = {"Young": "#2a9d8f", "Old": "#6a4c93"}
    fig, axes = plt.subplots(len(metrics), len(targets), figsize=(6.9, 6.4), constrained_layout=True)
    x_positions = np.arange(len(fields))
    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, target_gene in enumerate(targets):
            ax = axes[row_idx, col_idx]
            target = delta[delta["target_gene"] == target_gene]
            for field_idx, field_type in enumerate(fields):
                group = target[target["field_type"] == field_type]
                jitter = np.linspace(-0.08, 0.08, max(len(group), 1))
                for offset, item in zip(jitter, group.itertuples(index=False)):
                    ax.scatter(
                        field_idx + offset,
                        getattr(item, metric),
                        s=12,
                        color=colors.get(item.condition, "black"),
                        alpha=0.85,
                        linewidths=0,
                    )
                summary = paired[(paired["target_gene"] == target_gene) & (paired["field_type"] == field_type)]
                if not summary.empty:
                    mean_value = float(summary.iloc[0][f"{metric}_mean"])
                    sem_value = float(summary.iloc[0][f"{metric}_sem"])
                    ax.errorbar(
                        field_idx,
                        mean_value,
                        yerr=sem_value,
                        fmt="s",
                        color="black",
                        markersize=3,
                        linewidth=0.8,
                        capsize=2,
                    )
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.set_xticks(x_positions)
            ax.set_xticklabels(["Masked", "Gauss07"])
            ax.set_ylabel(ylabel)
            ax.set_title(target_gene, fontsize=7, pad=2)
            ax.tick_params(width=0.3, length=2)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=4, label=condition)
        for condition, color in colors.items()
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=6, loc="best")
    fig.savefig(output_dir / "low_pde_profile_validation_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "low_pde_profile_validation_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(
    frame: pd.DataFrame,
    delta: pd.DataFrame,
    paired: pd.DataFrame,
    group: pd.DataFrame,
    output_dir: Path,
) -> None:
    masked = delta[delta["field_type"] == "masked"]
    overall_pearson_delta = masked["spot_pearson_source_delta_from_default"].mean()
    overall_mse_delta = masked["spot_mse_source_delta_from_default"].mean()
    overall_rough_delta = masked["roughness_grad_p95_delta_from_default"].mean()
    improved_count = int((masked["spot_pearson_source_delta_from_default"] > 0).sum())
    total_count = int(len(masked))
    best_lines = []
    for row in paired.itertuples(index=False):
        best_lines.append(
            "- "
            f"{row.target_gene} {row.field_type}: mean Pearson delta "
            f"`{row.spot_pearson_source_delta_from_default_mean:+.4f}` "
            f"({row.spot_pearson_source_delta_from_default_n_positive}/{row.n_samples} samples positive), "
            f"mean MSE delta `{row.spot_mse_source_delta_from_default_mean:+.4f}`, "
            f"mean roughness-p95 delta `{row.roughness_grad_p95_delta_from_default_mean:+.4f}`."
        )
    lines = [
        "# Low-PDE Profile Validation Interpretation",
        "",
        "Multi-section validation comparing `low_pde` (data weight 8, PDE weight 0.04) against the current `fourier_refined_16g` default (data weight 8, PDE weight 0.12) across all 8 GSE193107 brain sections for `Apoe` and `Gfap`.",
        "",
        "## Main Result",
        "",
        f"- Across masked fields, low-PDE improved source Pearson in `{improved_count}/{total_count}` sample-target pairs.",
        f"- Across masked fields, mean source-Pearson delta was `{overall_pearson_delta:+.4f}`.",
        f"- Across masked fields, mean source-MSE delta was `{overall_mse_delta:+.4f}`.",
        f"- Across masked fields, mean roughness-p95 delta was `{overall_rough_delta:+.4f}`.",
        "",
        "## Paired Target-Level Summary",
        "",
        *best_lines,
        "",
        "## Interpretation",
        "",
        "The low-PDE setting is a stronger source-fitting candidate than the current conservative default if the primary goal is fitted-source agreement. The trade-off is increased roughness in several masked fields, and the improvement is target- and field-type-dependent after gauss07 postprocessing. This supports reporting low-PDE as a candidate optimized profile or sensitivity result, but not silently replacing the default without deciding whether the manuscript prioritizes source fit or physics-regularized smoothness.",
        "",
        "## Output Files",
        "",
        "- `low_pde_profile_validation_metrics.csv`: raw metrics.",
        "- `low_pde_profile_validation_delta_from_default.csv`: paired low-PDE minus default deltas.",
        "- `low_pde_profile_validation_paired_summary.csv`: target/field paired summaries.",
        "- `low_pde_profile_validation_group_summary.csv`: absolute and delta group summaries.",
        "- `low_pde_profile_validation_summary.pdf` and `.png`: publication-style paired-delta figure.",
        "",
    ]
    (output_dir / "low_pde_profile_validation_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

