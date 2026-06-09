"""Summarize representative PINN comparisons for H&E histology priors."""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "histology_prior" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize histology-prior PINN comparisons.")
    parser.add_argument("--metrics-csv", default=str(OUTPUT_ROOT / "histology_prior_pinn_metrics_summary.csv"))
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.metrics_csv)
    metrics = [
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "high_to_low_barrier_prediction_ratio",
    ]
    group = (
        frame.groupby(["target_gene", "field_type", "histology_prior"], as_index=False)
        .agg(
            n_runs=("sample", "count"),
            spot_pearson_source_mean=("spot_pearson_source", "mean"),
            spot_pearson_source_sd=("spot_pearson_source", "std"),
            spot_mse_source_mean=("spot_mse_source", "mean"),
            spot_mse_source_sd=("spot_mse_source", "std"),
            roughness_grad_p95_mean=("roughness_grad_p95", "mean"),
            roughness_grad_p95_sd=("roughness_grad_p95", "std"),
            high_to_low_barrier_prediction_ratio_mean=("high_to_low_barrier_prediction_ratio", "mean"),
            high_to_low_barrier_prediction_ratio_sd=("high_to_low_barrier_prediction_ratio", "std"),
        )
    )
    group.to_csv(output_dir / "histology_prior_pinn_group_summary.csv", index=False)

    paired_rows = []
    for (target_gene, field_type), subset in frame.groupby(["target_gene", "field_type"], sort=True):
        pivot = subset.pivot_table(
            index=["sample", "condition"],
            columns="histology_prior",
            values=metrics,
        )
        for metric in metrics:
            diff = pivot[(metric, "hematoxylin")] - pivot[(metric, "brightness")]
            paired_rows.append(
                {
                    "target_gene": target_gene,
                    "field_type": field_type,
                    "metric": metric,
                    "n_pairs": int(diff.notna().sum()),
                    "hematoxylin_minus_brightness_mean": float(diff.mean()),
                    "hematoxylin_minus_brightness_sd": float(diff.std(ddof=1)),
                    "n_improved": int((diff > 0).sum()) if metric in {"spot_pearson_source"} else int((diff < 0).sum()),
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(output_dir / "histology_prior_pinn_paired_deltas.csv", index=False)
    plot_summary(group, paired, output_dir)
    write_interpretation(group, paired, output_dir)
    print(f"Wrote histology prior PINN summaries to {output_dir}")


def plot_summary(group: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
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
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), constrained_layout=True)
    metrics = [
        ("spot_pearson_source_mean", "spot_pearson_source_sd", "Source Pearson", False),
        ("spot_mse_source_mean", "spot_mse_source_sd", "Source MSE", True),
        ("roughness_grad_p95_mean", "roughness_grad_p95_sd", "Roughness p95", True),
        (
            "high_to_low_barrier_prediction_ratio_mean",
            "high_to_low_barrier_prediction_ratio_sd",
            "High/low barrier ratio",
            None,
        ),
    ]
    colors = {"brightness": "#6c757d", "hematoxylin": "#7b2cbf"}
    x_labels = []
    for target in ["Apoe", "Gfap"]:
        for field in ["masked", "gauss07"]:
            x_labels.append((target, field))
    x = np.arange(len(x_labels))
    width = 0.36
    for ax, (metric, sd_metric, ylabel, lower_is_better) in zip(axes.reshape(-1), metrics):
        for offset, prior in [(-width / 2, "brightness"), (width / 2, "hematoxylin")]:
            values = []
            errors = []
            for target, field in x_labels:
                row = group[
                    (group["target_gene"] == target)
                    & (group["field_type"] == field)
                    & (group["histology_prior"] == prior)
                ]
                values.append(float(row.iloc[0][metric]))
                errors.append(float(row.iloc[0][sd_metric]))
            ax.bar(
                x + offset,
                values,
                yerr=errors,
                width=width,
                label=prior.capitalize(),
                color=colors[prior],
                edgecolor="black",
                linewidth=0.3,
                capsize=2,
            )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{target}\n{field}" for target, field in x_labels])
        ax.tick_params(width=0.3, length=2)
        if lower_is_better is True:
            ax.set_title("lower is better", fontsize=6, pad=2)
        elif lower_is_better is False:
            ax.set_title("higher is better", fontsize=6, pad=2)
    axes[0, 0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.savefig(output_dir / "histology_prior_pinn_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "histology_prior_pinn_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(group: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Histology Prior PINN Summary",
        "",
        "Representative PINN comparison of brightness-derived versus hematoxylin-derived H&E structural priors.",
        "",
        "## Mean Metrics",
        "",
    ]
    for row in group.itertuples(index=False):
        lines.append(
            f"- {row.target_gene}, {row.field_type}, {row.histology_prior}: n={row.n_runs}, Pearson `{row.spot_pearson_source_mean:.3f}`, MSE `{row.spot_mse_source_mean:.4f}`, roughness p95 `{row.roughness_grad_p95_mean:.3f}`."
        )
    lines.extend(["", "## Paired Deltas", ""])
    for row in paired.itertuples(index=False):
        lines.append(
            f"- {row.target_gene}, {row.field_type}, `{row.metric}`: hematoxylin - brightness `{row.hematoxylin_minus_brightness_mean:.4f}` +/- `{row.hematoxylin_minus_brightness_sd:.4f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Hematoxylin is clearly better as a physiological H&E structural-resistance proxy at the preflight level, but representative PINN-level performance is mixed. It tends to slightly reduce roughness, shows modest Apoe masked-field gains, and slightly worsens Gfap source-fit metrics in this subset. It should remain an ablation/upgrade candidate rather than becoming the default until larger PINN-level validation is completed.",
            "",
        ]
    )
    (output_dir / "histology_prior_pinn_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

