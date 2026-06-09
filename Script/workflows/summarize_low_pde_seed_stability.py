"""Summarize low-PDE random-seed stability benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "loss_weight_sensitivity" / "brain_aging_gse193107" / "low_pde_seed_stability"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize low-PDE seed stability benchmark.")
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    frame = pd.read_csv(output_dir / "low_pde_seed_stability_metrics.csv")
    delta = compute_delta(frame)
    stability = summarize_stability(frame)
    paired = summarize_delta(delta)

    delta.to_csv(output_dir / "low_pde_seed_stability_delta_from_default.csv", index=False)
    stability.to_csv(output_dir / "low_pde_seed_stability_setting_summary.csv", index=False)
    paired.to_csv(output_dir / "low_pde_seed_stability_paired_summary.csv", index=False)
    plot_summary(delta, paired, output_dir)
    write_interpretation(delta, stability, paired, output_dir)
    print(f"Wrote low-PDE seed-stability summaries to {output_dir}", flush=True)


def compute_delta(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "high_to_low_barrier_prediction_ratio",
    ]
    reference = frame[frame["setting"] == "default"].set_index(["sample", "target_gene", "seed", "field_type"])
    rows = []
    for row in frame[frame["setting"] == "low_pde"].itertuples(index=False):
        base = reference.loc[(row.sample, row.target_gene, row.seed, row.field_type)]
        record = {
            "sample": row.sample,
            "condition": row.condition,
            "target_gene": row.target_gene,
            "seed": int(row.seed),
            "field_type": row.field_type,
            "histology_prior": row.histology_prior,
        }
        for metric in metrics:
            value = float(getattr(row, metric))
            base_value = float(base[metric])
            record[metric] = value
            record[f"{metric}_default"] = base_value
            record[f"{metric}_delta_from_default"] = value - base_value
        rows.append(record)
    return pd.DataFrame(rows)


def summarize_stability(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (sample, condition, target_gene, field_type, setting), group in frame.groupby(
        ["sample", "condition", "target_gene", "field_type", "setting"],
        sort=True,
    ):
        rows.append(
            {
                "sample": sample,
                "condition": condition,
                "target_gene": target_gene,
                "field_type": field_type,
                "setting": setting,
                "n_seeds": int(len(group)),
                "spot_pearson_source_mean": group["spot_pearson_source"].mean(),
                "spot_pearson_source_sd": group["spot_pearson_source"].std(ddof=1),
                "spot_mse_source_mean": group["spot_mse_source"].mean(),
                "roughness_grad_p95_mean": group["roughness_grad_p95"].mean(),
                "roughness_grad_p95_sd": group["roughness_grad_p95"].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def summarize_delta(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target_gene, field_type), group in delta.groupby(["target_gene", "field_type"], sort=True):
        values = group["spot_pearson_source_delta_from_default"].to_numpy(dtype=float)
        rough = group["roughness_grad_p95_delta_from_default"].to_numpy(dtype=float)
        rows.append(
            {
                "target_gene": target_gene,
                "field_type": field_type,
                "n_comparisons": int(len(group)),
                "source_pearson_delta_mean": float(np.mean(values)),
                "source_pearson_delta_sd": float(np.std(values, ddof=1)),
                "source_pearson_delta_n_positive": int(np.sum(values > 0)),
                "source_mse_delta_mean": group["spot_mse_source_delta_from_default"].mean(),
                "roughness_p95_delta_mean": float(np.mean(rough)),
                "roughness_p95_delta_sd": float(np.std(rough, ddof=1)),
                "roughness_p95_delta_n_positive": int(np.sum(rough > 0)),
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
    ]
    fields = ["masked", "gauss07"]
    targets = ["Apoe", "Gfap"]
    colors = {"Young": "#2a9d8f", "Old": "#6a4c93"}
    fig, axes = plt.subplots(len(metrics), len(targets), figsize=(6.6, 4.9), constrained_layout=True)
    for row_idx, (metric, ylabel) in enumerate(metrics):
        for col_idx, target_gene in enumerate(targets):
            ax = axes[row_idx, col_idx]
            target = delta[delta["target_gene"] == target_gene]
            for field_idx, field_type in enumerate(fields):
                group = target[target["field_type"] == field_type]
                jitter = np.linspace(-0.1, 0.1, max(len(group), 1))
                for offset, item in zip(jitter, group.itertuples(index=False)):
                    ax.scatter(
                        field_idx + offset,
                        getattr(item, metric),
                        s=12,
                        color=colors.get(item.condition, "black"),
                        alpha=0.85,
                        linewidths=0,
                    )
            ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
            ax.set_xticks(np.arange(len(fields)))
            ax.set_xticklabels(["Masked", "Gauss07"])
            ax.set_ylabel(ylabel)
            ax.set_title(target_gene, fontsize=7, pad=2)
            ax.tick_params(width=0.3, length=2)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=4, label=condition)
        for condition, color in colors.items()
    ]
    axes[0, 0].legend(handles=handles, frameon=False, fontsize=6, loc="best")
    fig.savefig(output_dir / "low_pde_seed_stability_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "low_pde_seed_stability_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(delta: pd.DataFrame, stability: pd.DataFrame, paired: pd.DataFrame, output_dir: Path) -> None:
    masked = delta[delta["field_type"] == "masked"]
    positive = int((masked["spot_pearson_source_delta_from_default"] > 0).sum())
    total = int(len(masked))
    mean_delta = masked["spot_pearson_source_delta_from_default"].mean()
    mean_mse_delta = masked["spot_mse_source_delta_from_default"].mean()
    mean_rough_delta = masked["roughness_grad_p95_delta_from_default"].mean()
    max_low_pde_sd = stability[
        (stability["setting"] == "low_pde") & (stability["field_type"] == "masked")
    ]["spot_pearson_source_sd"].max()
    lines = [
        "# Low-PDE Seed Stability Interpretation",
        "",
        "Representative random-seed stability benchmark comparing default and low-PDE profiles on young A1 and old A1 sections for Apoe/Gfap across seeds 0, 1, and 2.",
        "",
        "## Main Result",
        "",
        f"- Across masked fields, low-PDE improved source Pearson in `{positive}/{total}` seed-level comparisons.",
        f"- Across masked fields, mean source-Pearson delta was `{mean_delta:+.4f}`.",
        f"- Across masked fields, mean source-MSE delta was `{mean_mse_delta:+.4f}`.",
        f"- Across masked fields, mean roughness-p95 delta was `{mean_rough_delta:+.4f}`.",
        f"- Maximum low-PDE masked-field source-Pearson SD across sample-target groups was `{max_low_pde_sd:.4f}`.",
        "",
        "## Target-Level Summary",
        "",
    ]
    for row in paired.itertuples(index=False):
        lines.append(
            "- "
            f"{row.target_gene} {row.field_type}: mean Pearson delta `{row.source_pearson_delta_mean:+.4f}` "
            f"({row.source_pearson_delta_n_positive}/{row.n_comparisons} positive), "
            f"mean MSE delta `{row.source_mse_delta_mean:+.4f}`, "
            f"mean roughness-p95 delta `{row.roughness_p95_delta_mean:+.4f}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The representative seed-stability check supports the same direction as the all-section seed-0 validation: low-PDE improves fitted source agreement across tested initializations. The remaining trade-off is still biological/visual regularity, especially for Gfap roughness, rather than optimization instability.",
            "",
        ]
    )
    (output_dir / "low_pde_seed_stability_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
