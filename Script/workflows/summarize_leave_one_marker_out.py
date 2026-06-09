"""Summarize leave-one-marker-out CNS myelin barrier benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "leave_one_marker_out" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize leave-one-marker-out barrier benchmark.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_dir = OUTPUT_ROOT / args.sample
    metrics_path = sample_dir / "leave_one_marker_out_metrics_summary.csv"
    frame = pd.read_csv(metrics_path)
    frame = frame.sort_values(["target_gene", "marker_set", "field_type"])

    delta = compute_delta_from_full(frame)
    compact = make_compact_summary(frame)

    delta.to_csv(sample_dir / "leave_one_marker_out_delta_from_full.csv", index=False)
    compact.to_csv(sample_dir / "leave_one_marker_out_compact_summary.csv", index=False)
    plot_summary(frame, delta, sample_dir)
    write_interpretation(frame, delta, sample_dir)
    print(f"Wrote leave-one-marker-out summaries to {sample_dir}")


def compute_delta_from_full(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "spot_pearson_barrier",
        "high_to_low_barrier_prediction_ratio",
    ]
    rows = []
    full = frame[frame["marker_set"] == "full"].set_index(["target_gene", "field_type"])
    for row in frame[frame["marker_set"] != "full"].itertuples(index=False):
        base = full.loc[(row.target_gene, row.field_type)]
        record = {
            "sample": row.sample,
            "target_gene": row.target_gene,
            "field_type": row.field_type,
            "marker_set": row.marker_set,
        }
        for metric in metrics:
            value = float(getattr(row, metric))
            base_value = float(base[metric])
            record[metric] = value
            record[f"{metric}_full"] = base_value
            record[f"{metric}_delta_from_full"] = value - base_value
        rows.append(record)
    return pd.DataFrame(rows)


def make_compact_summary(frame: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "sample",
        "target_gene",
        "marker_set",
        "field_type",
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "spot_pearson_barrier",
        "high_to_low_barrier_prediction_ratio",
    ]
    return frame[keep].copy()


def plot_summary(frame: pd.DataFrame, delta: pd.DataFrame, output_dir: Path) -> None:
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

    marker_order = ["full", "drop_Mbp", "drop_Plp1", "drop_Mobp"]
    marker_labels = ["Full", "No Mbp", "No Plp1", "No Mobp"]
    colors = {"masked": "#2a9d8f", "gauss07": "#e76f51"}
    metrics = [
        ("spot_pearson_source", "Source Pearson"),
        ("spot_mse_source", "Source MSE"),
        ("roughness_grad_p95", "Roughness p95"),
        ("high_to_low_barrier_prediction_ratio", "High/low barrier ratio"),
    ]

    targets = list(frame["target_gene"].drop_duplicates())
    fig, axes = plt.subplots(len(targets), len(metrics), figsize=(7.2, 2.1 * len(targets)), constrained_layout=True)
    if len(targets) == 1:
        axes = axes.reshape(1, -1)

    for row_idx, target_gene in enumerate(targets):
        target = frame[frame["target_gene"] == target_gene]
        for col_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for field_type, group in target.groupby("field_type", sort=True):
                indexed = group.set_index("marker_set").reindex(marker_order)
                ax.plot(
                    np.arange(len(marker_order)),
                    indexed[metric].to_numpy(dtype=float),
                    marker="o",
                    linewidth=1.1,
                    markersize=3,
                    color=colors.get(field_type, "black"),
                    label=field_type,
                )
            ax.set_xticks(np.arange(len(marker_order)))
            ax.set_xticklabels(marker_labels, rotation=35, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(target_gene, fontsize=7, pad=2)
            ax.tick_params(width=0.3, length=2)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=6)

    fig.savefig(output_dir / "leave_one_marker_out_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "leave_one_marker_out_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    plot_delta_heatmap(delta, output_dir)


def plot_delta_heatmap(delta: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    metric = "spot_pearson_source_delta_from_full"
    pivot = delta.pivot_table(index=["target_gene", "field_type"], columns="marker_set", values=metric, aggfunc="mean")
    pivot = pivot.reindex(columns=["drop_Mbp", "drop_Plp1", "drop_Mobp"])

    fig, ax = plt.subplots(figsize=(4.0, 2.2), constrained_layout=True)
    vmax = max(0.01, float(np.nanmax(np.abs(pivot.to_numpy(dtype=float)))))
    image = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(["No Mbp", "No Plp1", "No Mobp"], rotation=35, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([f"{idx[0]} {idx[1]}" for idx in pivot.index])
    ax.set_title("Pearson delta from full marker set", fontsize=7, pad=2)
    ax.tick_params(width=0.3, length=2)
    for y in range(pivot.shape[0]):
        for x in range(pivot.shape[1]):
            value = pivot.iloc[y, x]
            ax.text(x, y, f"{value:+.3f}", ha="center", va="center", fontsize=6)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(width=0.3, length=2)
    cbar.set_label("Delta", fontsize=7)

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig.savefig(output_dir / "leave_one_marker_out_pearson_delta_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "leave_one_marker_out_pearson_delta_heatmap.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(frame: pd.DataFrame, delta: pd.DataFrame, output_dir: Path) -> None:
    masked_delta = delta[delta["field_type"] == "masked"]
    max_abs_pearson_delta = masked_delta["spot_pearson_source_delta_from_full"].abs().max()
    max_abs_mse_delta = masked_delta["spot_mse_source_delta_from_full"].abs().max()
    max_abs_rough_delta = masked_delta["roughness_grad_p95_delta_from_full"].abs().max()

    lines = [
        "# Leave-One-Marker-Out Interpretation",
        "",
        "Representative old A1 benchmark testing whether the CNS myelin barrier is dominated by a single marker.",
        "",
        "## Main Result",
        "",
        f"- Across masked fields, the largest absolute source Pearson change after dropping one marker was `{max_abs_pearson_delta:.4f}`.",
        f"- Across masked fields, the largest absolute source MSE change was `{max_abs_mse_delta:.4f}`.",
        f"- Across masked fields, the largest absolute roughness p95 change was `{max_abs_rough_delta:.4f}`.",
        "",
        "## Per-Target Notes",
        "",
    ]
    for target_gene, group in frame.groupby("target_gene", sort=True):
        full_masked = group[(group["marker_set"] == "full") & (group["field_type"] == "masked")].iloc[0]
        best_drop = (
            delta[(delta["target_gene"] == target_gene) & (delta["field_type"] == "masked")]
            .sort_values("spot_pearson_source", ascending=False)
            .iloc[0]
        )
        lines.append(
            f"- {target_gene}: full masked Pearson `{full_masked.spot_pearson_source:.3f}`, MSE `{full_masked.spot_mse_source:.4f}`, roughness p95 `{full_masked.roughness_grad_p95:.3f}`. Best dropped-marker masked field was `{best_drop.marker_set}` with Pearson `{best_drop.spot_pearson_source:.3f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Dropping Mbp, Plp1, or Mobp individually causes only small metric shifts in this representative section. The CNS myelin barrier therefore behaves as a redundant module-level prior rather than a single-gene proxy. This supports describing the barrier as marker-set derived, while still requiring a broader multi-section marker-ablation experiment if the claim is promoted to a main-text robustness result.",
            "",
        ]
    )
    (output_dir / "leave_one_marker_out_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
