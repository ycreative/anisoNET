"""Summarize data/PDE loss weight sensitivity benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "loss_weight_sensitivity" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize loss weight sensitivity benchmark.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / args.sample
    frame = pd.read_csv(output_dir / "loss_weight_sensitivity_summary.csv")
    frame["setting"] = pd.Categorical(
        frame["setting"],
        categories=["low_pde", "default", "high_pde", "low_data", "high_data"],
        ordered=True,
    )
    frame = frame.sort_values(["target_gene", "field_type", "setting"])
    delta = compute_delta(frame)
    delta.to_csv(output_dir / "loss_weight_sensitivity_delta_from_default.csv", index=False)
    plot_summary(frame, output_dir)
    write_interpretation(frame, delta, output_dir)
    print(f"Wrote loss weight sensitivity summaries to {output_dir}")


def compute_delta(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "high_to_low_barrier_prediction_ratio",
    ]
    reference = frame[frame["setting"] == "default"].set_index(["target_gene", "field_type"])
    rows = []
    for row in frame.itertuples(index=False):
        base = reference.loc[(row.target_gene, row.field_type)]
        record = {
            "sample": row.sample,
            "target_gene": row.target_gene,
            "field_type": row.field_type,
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
    labels = {
        "low_pde": "Low PDE",
        "default": "Default",
        "high_pde": "High PDE",
        "low_data": "Low data",
        "high_data": "High data",
    }
    colors = {"masked": "#2a9d8f", "gauss07": "#e76f51"}
    settings = ["low_pde", "default", "high_pde", "low_data", "high_data"]
    targets = list(frame["target_gene"].drop_duplicates())
    fig, axes = plt.subplots(len(targets), len(metrics), figsize=(7.2, 2.15 * len(targets)), constrained_layout=True)
    if len(targets) == 1:
        axes = axes.reshape(1, -1)
    x = np.arange(len(settings))
    for row_idx, target_gene in enumerate(targets):
        target = frame[frame["target_gene"] == target_gene]
        for col_idx, (metric, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for field_type, group in target.groupby("field_type", sort=True):
                indexed = group.set_index("setting").reindex(settings)
                ax.plot(
                    x,
                    indexed[metric].to_numpy(dtype=float),
                    marker="o",
                    linewidth=1.1,
                    markersize=3,
                    color=colors.get(field_type, "black"),
                    label=field_type,
                )
            ax.axvline(1, color="black", linewidth=0.5, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels([labels[s] for s in settings], rotation=35, ha="right")
            ax.set_ylabel(ylabel)
            ax.set_title(target_gene, fontsize=7, pad=2)
            ax.tick_params(width=0.3, length=2)
            if row_idx == 0 and col_idx == 0:
                ax.legend(frameon=False, fontsize=6)
    fig.savefig(output_dir / "loss_weight_sensitivity_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "loss_weight_sensitivity_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(frame: pd.DataFrame, delta: pd.DataFrame, output_dir: Path) -> None:
    masked = delta[(delta["field_type"] == "masked") & (delta["setting"] != "default")]
    max_abs_pearson_delta = masked["spot_pearson_source_delta_from_default"].abs().max()
    max_abs_mse_delta = masked["spot_mse_source_delta_from_default"].abs().max()
    max_abs_rough_delta = masked["roughness_grad_p95_delta_from_default"].abs().max()
    low_pde = delta[(delta["field_type"] == "masked") & (delta["setting"] == "low_pde")]
    low_pde_pearson_delta = low_pde["spot_pearson_source_delta_from_default"].mean()
    low_pde_rough_delta = low_pde["roughness_grad_p95_delta_from_default"].mean()
    best_rows = []
    for (target_gene, field_type), group in frame.groupby(["target_gene", "field_type"], sort=True):
        best_pearson = group.loc[group["spot_pearson_source"].idxmax()]
        best_mse = group.loc[group["spot_mse_source"].idxmin()]
        best_rows.append(
            f"- {target_gene} {field_type}: best Pearson `{best_pearson.setting}` (`{best_pearson.spot_pearson_source:.3f}`), best MSE `{best_mse.setting}` (`{best_mse.spot_mse_source:.4f}`)."
        )
    lines = [
        "# Loss Weight Sensitivity Interpretation",
        "",
        "Representative benchmark varying data/PDE loss weights around the default `fourier_refined_16g` setting.",
        "",
        "## Main Result",
        "",
        f"- Across masked fields, the largest absolute Pearson delta from default was `{max_abs_pearson_delta:.4f}`.",
        f"- Across masked fields, the largest absolute MSE delta from default was `{max_abs_mse_delta:.4f}`.",
        f"- Across masked fields, the largest absolute roughness p95 delta from default was `{max_abs_rough_delta:.4f}`.",
        f"- Low-PDE masked fields changed Pearson by a mean of `{low_pde_pearson_delta:+.4f}` and roughness p95 by a mean of `{low_pde_rough_delta:+.4f}` relative to default.",
        "",
        "## Best Settings",
        "",
        *best_rows,
        "",
        "## Interpretation",
        "",
        "Loss-weight sensitivity shows that the chosen balance between source-data fitting and PDE regularization materially affects representative fields. In this run, lower PDE weight improved source agreement for both targets, while Gfap became rougher and high-PDE or low-data settings underfit. The current default should therefore be described as a conservative physics-regularized setting, not as a proven optimum. A lower-PDE or higher-data profile is a concrete candidate for future optimization if the manuscript needs stronger source-fit performance.",
        "",
    ]
    (output_dir / "loss_weight_sensitivity_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
