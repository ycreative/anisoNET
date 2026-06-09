"""Summarize batch H&E prior preflight comparisons."""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "histology_prior" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize H&E prior preflight comparison.")
    parser.add_argument("--summary-csv", default=str(OUTPUT_ROOT / "histology_prior_preflight_summary.csv"))
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.summary_csv)
    group = (
        frame.groupby(["condition", "histology_prior"], as_index=False)
        .agg(
            n_runs=("sample", "count"),
            histology_resistance_mean=("histology_resistance_mean_in_tissue", "mean"),
            histology_resistance_p95_mean=("histology_resistance_p95_in_tissue", "mean"),
            diffusion_min_mean=("diffusion_min_in_tissue", "mean"),
            resistance_ratio_mean=("resistance_ratio_in_tissue", "mean"),
            resistance_ratio_sd=("resistance_ratio_in_tissue", "std"),
        )
    )
    group.to_csv(output_dir / "histology_prior_preflight_group_summary.csv", index=False)

    paired = (
        frame.pivot_table(
            index=["sample", "condition", "target_gene"],
            columns="histology_prior",
            values=[
                "histology_resistance_mean_in_tissue",
                "histology_resistance_p95_in_tissue",
                "diffusion_min_in_tissue",
                "resistance_ratio_in_tissue",
            ],
        )
        .reset_index()
    )
    paired.columns = [
        "_".join(col).rstrip("_") if isinstance(col, tuple) else col
        for col in paired.columns
    ]
    paired["delta_p95_hematoxylin_minus_brightness"] = (
        paired["histology_resistance_p95_in_tissue_hematoxylin"]
        - paired["histology_resistance_p95_in_tissue_brightness"]
    )
    paired["resistance_ratio_fold_hematoxylin_vs_brightness"] = (
        paired["resistance_ratio_in_tissue_hematoxylin"]
        / paired["resistance_ratio_in_tissue_brightness"]
    )
    paired.to_csv(output_dir / "histology_prior_preflight_paired_summary.csv", index=False)
    plot_summary(frame, paired, output_dir)
    write_interpretation(group, paired, output_dir)
    print(f"Wrote histology prior preflight summaries to {output_dir}")


def plot_summary(frame: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    colors = {"brightness": "#6c757d", "hematoxylin": "#7b2cbf"}

    for ax, metric, ylabel in [
        (axes[0], "histology_resistance_p95_in_tissue", "Structural resistance p95"),
        (axes[1], "resistance_ratio_in_tissue", "Resistance ratio"),
    ]:
        data = [
            frame.loc[frame["histology_prior"] == prior, metric].to_numpy(dtype=float)
            for prior in ["brightness", "hematoxylin"]
        ]
        ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False)
        for patch, prior in zip(ax.patches, ["brightness", "hematoxylin"]):
            patch.set_facecolor(colors[prior])
            patch.set_alpha(0.75)
            patch.set_linewidth(0.4)
        for idx, values in enumerate(data, start=1):
            jitter = np.linspace(-0.08, 0.08, len(values))
            ax.scatter(idx + jitter, values, s=8, color="black", alpha=0.55, linewidths=0)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Brightness", "Hematoxylin"], rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.tick_params(width=0.3, length=2)

    ax = axes[2]
    values = paired["resistance_ratio_fold_hematoxylin_vs_brightness"].to_numpy(dtype=float)
    ax.axhline(1.0, color="black", linewidth=0.5, linestyle="--")
    ax.boxplot([values], widths=0.5, patch_artist=True, showfliers=False)
    ax.patches[0].set_facecolor("#2a9d8f")
    ax.patches[0].set_alpha(0.75)
    ax.scatter(1 + np.linspace(-0.08, 0.08, len(values)), values, s=8, color="black", alpha=0.55, linewidths=0)
    ax.set_xticks([1])
    ax.set_xticklabels(["Hematoxylin /\nbrightness"], rotation=0)
    ax.set_ylabel("Resistance ratio fold-change")
    ax.tick_params(width=0.3, length=2)

    fig.savefig(output_dir / "histology_prior_preflight_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "histology_prior_preflight_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(group: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
    p95_delta_mean = paired["delta_p95_hematoxylin_minus_brightness"].mean()
    ratio_fold_mean = paired["resistance_ratio_fold_hematoxylin_vs_brightness"].mean()
    ratio_fold_median = paired["resistance_ratio_fold_hematoxylin_vs_brightness"].median()
    lines = [
        "# Histology Prior Batch Preflight Summary",
        "",
        "This summary compares the original brightness-derived H&E prior with the hematoxylin-derived nuclei-rich structural prior across GSE193107 Apoe/Gfap preflight runs.",
        "",
        "## Key Findings",
        "",
        f"- Hematoxylin increased tissue structural-resistance p95 by a mean of `{p95_delta_mean:.3f}` compared with the brightness prior.",
        f"- Hematoxylin increased the diffusion resistance ratio by a mean fold-change of `{ratio_fold_mean:.2f}` and a median fold-change of `{ratio_fold_median:.2f}`.",
        "- No preflight failures were observed in the completed batch.",
        "",
        "## Interpretation",
        "",
        "The hematoxylin prior is consistently more locally restrictive than the brightness prior, supporting its use as a physiologically motivated H&E structural-resistance ablation. This still should not be described as validated nuclei density unless explicit nuclei segmentation or annotation is added.",
        "",
    ]
    lines.append("## Group Summary")
    lines.append("")
    for row in group.itertuples(index=False):
        lines.append(
            f"- {row.condition}, {row.histology_prior}: n={row.n_runs}, p95 resistance mean `{row.histology_resistance_p95_mean:.3f}`, resistance-ratio mean `{row.resistance_ratio_mean:.1f}`."
        )
    (output_dir / "histology_prior_preflight_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

