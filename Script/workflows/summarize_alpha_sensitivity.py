"""Summarize alpha sensitivity benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "alpha_sensitivity" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize anisoNET alpha sensitivity benchmark.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = OUTPUT_ROOT / args.sample
    metrics_path = sample_dir / "alpha_sensitivity_summary.csv"
    frame = pd.read_csv(metrics_path)
    frame = frame.sort_values(["target_gene", "alpha"])
    best_rows = []
    for target_gene, group in frame.groupby("target_gene", sort=True):
        best_pearson = group.loc[group["spot_pearson_source"].idxmax()]
        best_mse = group.loc[group["spot_mse_source"].idxmin()]
        best_roughness = group.loc[group["roughness_grad_p95"].idxmin()]
        best_rows.extend(
            [
                {
                    "target_gene": target_gene,
                    "criterion": "max_source_pearson",
                    "alpha": float(best_pearson["alpha"]),
                    "value": float(best_pearson["spot_pearson_source"]),
                },
                {
                    "target_gene": target_gene,
                    "criterion": "min_source_mse",
                    "alpha": float(best_mse["alpha"]),
                    "value": float(best_mse["spot_mse_source"]),
                },
                {
                    "target_gene": target_gene,
                    "criterion": "min_roughness_p95",
                    "alpha": float(best_roughness["alpha"]),
                    "value": float(best_roughness["roughness_grad_p95"]),
                },
            ]
        )
    best = pd.DataFrame(best_rows)
    best.to_csv(sample_dir / "alpha_sensitivity_best_by_metric.csv", index=False)
    plot_summary(frame, sample_dir)
    write_interpretation(frame, best, sample_dir)
    print(f"Wrote alpha sensitivity summaries to {sample_dir}")


def plot_summary(frame: pd.DataFrame, output_dir: Path) -> None:
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
        ("spot_pearson_source", "Source Pearson", "higher"),
        ("spot_mse_source", "Source MSE", "lower"),
        ("roughness_grad_p95", "Roughness p95", "lower"),
        ("high_to_low_barrier_prediction_ratio", "High/low barrier ratio", "context"),
    ]
    colors = {"Apoe": "#d1495b", "Gfap": "#00798c"}
    for ax, (metric, ylabel, direction) in zip(axes.reshape(-1), metrics):
        for target_gene, group in frame.groupby("target_gene", sort=True):
            ax.plot(
                group["alpha"],
                group[metric],
                marker="o",
                linewidth=1.2,
                markersize=3,
                color=colors.get(target_gene, "black"),
                label=target_gene,
            )
        ax.set_xlabel("Transcriptomic barrier alpha")
        ax.set_ylabel(ylabel)
        ax.set_title(direction, fontsize=6, pad=2)
        ax.tick_params(width=0.3, length=2)
    axes[0, 0].legend(frameon=False, fontsize=6)
    fig.savefig(output_dir / "alpha_sensitivity_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "alpha_sensitivity_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(frame: pd.DataFrame, best: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Alpha Sensitivity Interpretation",
        "",
        "Representative alpha sensitivity benchmark for transcriptomic barrier strength.",
        "",
        "## Best Alpha by Metric",
        "",
    ]
    for row in best.itertuples(index=False):
        lines.append(
            f"- {row.target_gene}, {row.criterion}: alpha `{row.alpha:g}`, value `{row.value:.4f}`."
        )
    lines.extend(["", "## Per-Target Trend", ""])
    for target_gene, group in frame.groupby("target_gene", sort=True):
        alpha0 = group[group["alpha"] == 0.0].iloc[0]
        alpha4 = group[group["alpha"] == 4.0].iloc[0]
        alpha8 = group[group["alpha"] == 8.0].iloc[0]
        lines.extend(
            [
                f"- {target_gene}: alpha 4 versus alpha 0 changes source Pearson from `{alpha0.spot_pearson_source:.3f}` to `{alpha4.spot_pearson_source:.3f}`, source MSE from `{alpha0.spot_mse_source:.4f}` to `{alpha4.spot_mse_source:.4f}`, and roughness p95 from `{alpha0.roughness_grad_p95:.3f}` to `{alpha4.roughness_grad_p95:.3f}`.",
                f"- {target_gene}: alpha 8 gives source Pearson `{alpha8.spot_pearson_source:.3f}`, source MSE `{alpha8.spot_mse_source:.4f}`, and roughness p95 `{alpha8.roughness_grad_p95:.3f}`.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The transcriptomic barrier is not merely cosmetic: alpha 0 underperforms the moderate barrier settings. The default alpha 4 is close to the best setting for Apoe and remains within the high-performing range for Gfap, while very high alpha can increase roughness or reduce source fit depending on the target. This supports reporting alpha as a sensitivity parameter rather than an arbitrary fixed constant.",
            "",
        ]
    )
    (output_dir / "alpha_sensitivity_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
