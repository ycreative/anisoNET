"""Summarize anisoNET batch validation metrics and plot QC panels."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = [
    "spot_pearson_source",
    "spot_mse_source",
    "roughness_grad_p95",
    "roughness_grad_mean",
    "spot_pearson_barrier",
    "background_to_tissue_ratio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize anisoNET batch metrics.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="anisoNET batch validation")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else float("nan")


def mean_sem(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), sem


def grouped_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["condition"], row["field_type"])].append(row)

    out = []
    for (condition, field_type), group_rows in sorted(groups.items()):
        result: dict[str, object] = {
            "condition": condition,
            "field_type": field_type,
            "n": len(group_rows),
        }
        for metric in METRICS:
            mean, sem = mean_sem([f(row, metric) for row in group_rows])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sem"] = sem
        out.append(result)
    return out


def delta_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_sample: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        by_sample[row["sample"]][row["field_type"]] = row

    deltas = []
    for sample, fields in sorted(by_sample.items()):
        if "masked" not in fields or "gauss07" not in fields:
            continue
        row: dict[str, object] = {
            "sample": sample,
            "condition": fields["masked"]["condition"],
        }
        for metric in METRICS:
            row[f"delta_{metric}"] = f(fields["gauss07"], metric) - f(fields["masked"], metric)
        deltas.append(row)

    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in deltas:
        groups[str(row["condition"])].append(row)

    out = []
    for condition, group_rows in sorted(groups.items()):
        result: dict[str, object] = {
            "condition": condition,
            "n": len(group_rows),
        }
        for metric in METRICS:
            values = [float(row[f"delta_{metric}"]) for row in group_rows]
            mean, sem = mean_sem(values)
            result[f"delta_{metric}_mean"] = mean
            result[f"delta_{metric}_sem"] = sem
        out.append(result)
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_metrics(rows: list[dict[str, str]], output_dir: Path, title: str) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2), constrained_layout=True)
    panels = [
        ("Source Pearson", "spot_pearson_source", False),
        ("Source MSE", "spot_mse_source", False),
        ("Roughness p95", "roughness_grad_p95", False),
    ]
    colors = {"masked": "#3b6ea8", "gauss07": "#d57a33"}
    x_base = {"Young": 0, "Old": 1}
    offsets = {"masked": -0.14, "gauss07": 0.14}
    rng = np.random.default_rng(0)

    for ax, (label, metric, zero_floor) in zip(axes, panels):
        for condition in ["Young", "Old"]:
            for field_type in ["masked", "gauss07"]:
                values = [
                    f(row, metric)
                    for row in rows
                    if row["condition"] == condition and row["field_type"] == field_type
                ]
                mean, sem = mean_sem(values)
                x = x_base[condition] + offsets[field_type]
                ax.bar(x, mean, yerr=sem, width=0.24, color=colors[field_type], edgecolor="black", linewidth=0.3)
                jitter = rng.normal(0, 0.025, size=len(values))
                ax.scatter(np.full(len(values), x) + jitter, values, s=9, color="black", linewidths=0, zorder=3)
        ax.set_title(label, fontsize=7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Young", "Old"])
        if zero_floor:
            ax.set_ylim(bottom=0)
        ax.tick_params(length=2, width=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    handles = [
        mpl.patches.Patch(facecolor=colors["masked"], edgecolor="black", linewidth=0.3, label="Masked"),
        mpl.patches.Patch(facecolor=colors["gauss07"], edgecolor="black", linewidth=0.3, label="Gauss 0.7"),
    ]
    fig.legend(handles=handles, loc="upper center", ncols=2, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle(title, fontsize=8, y=1.18)
    fig.savefig(output_dir / "batch_metric_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "batch_metric_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(Path(args.input_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(grouped_summary(rows), output_dir / "group_summary.csv")
    write_csv(delta_summary(rows), output_dir / "postprocess_delta_summary.csv")
    plot_metrics(rows, output_dir, args.title)
    print(f"Wrote batch summaries to {output_dir}")


if __name__ == "__main__":
    main()
