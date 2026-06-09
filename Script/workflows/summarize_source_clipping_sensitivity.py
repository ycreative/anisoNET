"""Summarize source clipping sensitivity benchmark outputs."""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "source_clipping_sensitivity" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize source clipping sensitivity outputs.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument("--reference-percentile", type=float, default=99.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / args.sample
    frame = pd.read_csv(output_dir / "source_clipping_sensitivity_summary.csv")
    frame = frame.sort_values(["target_gene", "field_type", "train_source_percentile"])
    delta = compute_delta(frame, args.reference_percentile)
    delta.to_csv(output_dir / "source_clipping_sensitivity_delta_from_p99.csv", index=False)
    plot_summary(frame, output_dir)
    write_interpretation(frame, delta, output_dir, args.reference_percentile)
    print(f"Wrote source clipping sensitivity summaries to {output_dir}")


def compute_delta(frame: pd.DataFrame, reference_percentile: float) -> pd.DataFrame:
    metrics = [
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "high_to_low_barrier_prediction_ratio",
    ]
    reference = frame[frame["train_source_percentile"] == reference_percentile].set_index(["target_gene", "field_type"])
    rows = []
    for row in frame.itertuples(index=False):
        base = reference.loc[(row.target_gene, row.field_type)]
        record = {
            "sample": row.sample,
            "target_gene": row.target_gene,
            "field_type": row.field_type,
            "train_source_percentile": float(row.train_source_percentile),
            "reference_percentile": float(reference_percentile),
        }
        for metric in metrics:
            value = float(getattr(row, metric))
            base_value = float(base[metric])
            record[metric] = value
            record[f"{metric}_p99"] = base_value
            record[f"{metric}_delta_from_p99"] = value - base_value
        rows.append(record)
    return pd.DataFrame(rows)


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
    metrics = [
        ("spot_pearson_source", "Source Pearson"),
        ("spot_mse_source", "Source MSE"),
        ("roughness_grad_p95", "Roughness p95"),
        ("high_to_low_barrier_prediction_ratio", "High/low barrier ratio"),
    ]
    colors = {"masked": "#2a9d8f", "gauss07": "#e76f51"}
    targets = list(frame["target_gene"].drop_duplicates())
    fig, axes = plt.subplots(len(targets), len(metrics), figsize=(7.2, 2.1 * len(targets)), constrained_layout=True)
    if len(targets) == 1:
        axes = axes.reshape(1, -1)
    for row_idx, target_gene in enumerate(targets):
        target = frame[frame["target_gene"] == target_gene]
        for col_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for field_type, group in target.groupby("field_type", sort=True):
                ax.plot(
                    group["train_source_percentile"],
                    group[metric],
                    marker="o",
                    linewidth=1.1,
                    markersize=3,
                    color=colors.get(field_type, "black"),
                    label=field_type,
                )
            ax.axvline(99.0, color="black", linewidth=0.5, linestyle="--")
            ax.set_xlabel("Source clipping percentile")
            ax.set_ylabel(ylabel)
            ax.set_title(target_gene, fontsize=7, pad=2)
            ax.tick_params(width=0.3, length=2)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=6)
    fig.savefig(output_dir / "source_clipping_sensitivity_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "source_clipping_sensitivity_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(
    frame: pd.DataFrame,
    delta: pd.DataFrame,
    output_dir: Path,
    reference_percentile: float,
) -> None:
    masked = delta[(delta["field_type"] == "masked") & (delta["train_source_percentile"] != reference_percentile)]
    max_abs_pearson_delta = masked["spot_pearson_source_delta_from_p99"].abs().max()
    max_abs_mse_delta = masked["spot_mse_source_delta_from_p99"].abs().max()
    max_abs_rough_delta = masked["roughness_grad_p95_delta_from_p99"].abs().max()
    best_rows = []
    for (target_gene, field_type), group in frame.groupby(["target_gene", "field_type"], sort=True):
        best = group.loc[group["spot_pearson_source"].idxmax()]
        best_rows.append(
            f"- {target_gene} {field_type}: best Pearson at percentile `{best.train_source_percentile:g}` with Pearson `{best.spot_pearson_source:.3f}`."
        )
    lines = [
        "# Source Clipping Sensitivity Interpretation",
        "",
        f"Representative benchmark varying source-expression clipping percentile, with all fields evaluated against source percentile `{reference_percentile:g}`.",
        "",
        "## Main Result",
        "",
        f"- Across masked fields, the largest absolute Pearson delta from p99 was `{max_abs_pearson_delta:.4f}`.",
        f"- Across masked fields, the largest absolute MSE delta from p99 was `{max_abs_mse_delta:.4f}`.",
        f"- Across masked fields, the largest absolute roughness p95 delta from p99 was `{max_abs_rough_delta:.4f}`.",
        "",
        "## Best Pearson by Target",
        "",
        *best_rows,
        "",
        "## Interpretation",
        "",
        "The representative source-clipping benchmark tests whether anisoNET fields depend strongly on a single source-expression clipping choice. If p99 remains near the best or produces only small deltas, the default can be defended as a robust, conservative clipping setting. If lower clipping improves smoothness but weakens source agreement, report this as a fit-versus-regularity trade-off.",
        "",
    ]
    (output_dir / "source_clipping_sensitivity_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

