"""Summarize held-out anisoNET benchmark results."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ["test_pearson", "test_mse", "roughness_grad_p95", "roughness_grad_mean"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize held-out benchmark CSV files.")
    parser.add_argument("--input-csv", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="anisoNET held-out benchmark")
    return parser.parse_args()


def read_rows(paths: list[str]) -> list[dict[str, str]]:
    rows = []
    for value in paths:
        with Path(value).open("r", encoding="utf-8", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def f(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value else float("nan")


def mean_sem(values: list[float]) -> tuple[float, float]:
    arr = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    sem = float(np.std(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return float(np.mean(arr)), sem


def method_label(row: dict[str, str]) -> str:
    if row["method"] != "anisoNET":
        return row["method"]
    field_type = row["field_type"]
    if field_type.endswith("_gauss07"):
        return "anisoNET_gauss07"
    if field_type.endswith("_masked"):
        return "anisoNET_masked"
    return f"anisoNET_{field_type}"


def grouped_summary(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["target_gene"], row["condition"], row["method"], method_label(row))].append(row)

    out = []
    for (target, condition, method, label), group_rows in sorted(groups.items()):
        result: dict[str, object] = {
            "target_gene": target,
            "condition": condition,
            "method": method,
            "method_label": label,
            "n": len(group_rows),
        }
        for metric in METRICS:
            mean, sem = mean_sem([f(row, metric) for row in group_rows])
            result[f"{metric}_mean"] = mean
            result[f"{metric}_sem"] = sem
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


def plot_summary(rows: list[dict[str, str]], output_dir: Path, title: str) -> None:
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

    methods = ["nearest", "idw_k8", "gaussian_sigma1p5", "gaussian_sigma3", "anisoNET_masked", "anisoNET_gauss07"]
    colors = {
        "nearest": "#8c8c8c",
        "idw_k8": "#5b8fd1",
        "gaussian_sigma1p5": "#5aa469",
        "gaussian_sigma3": "#9ac66d",
        "anisoNET_masked": "#d57a33",
        "anisoNET_gauss07": "#b14b4b",
    }
    targets = sorted({row["target_gene"] for row in rows})
    fig, axes = plt.subplots(1, len(targets), figsize=(4.0 * len(targets), 2.7), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)
    rng = np.random.default_rng(0)

    for ax, target in zip(axes_arr, targets):
        target_rows = [row for row in rows if row["target_gene"] == target]
        for index, method in enumerate(methods):
            values = [f(row, "test_pearson") for row in target_rows if method_label(row) == method]
            mean, sem = mean_sem(values)
            ax.bar(index, mean, yerr=sem, width=0.72, color=colors[method], edgecolor="black", linewidth=0.3)
            jitter = rng.normal(0, 0.045, size=len(values))
            ax.scatter(np.full(len(values), index) + jitter, values, s=8, color="black", linewidths=0, zorder=3)
        ax.set_title(target, fontsize=8)
        ax.set_ylabel("Held-out Pearson")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.tick_params(length=2, width=0.4)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    fig.suptitle(title, fontsize=8, y=1.05)
    fig.savefig(output_dir / "heldout_test_pearson_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "heldout_test_pearson_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(grouped_summary(rows), output_dir / "heldout_group_summary.csv")
    plot_summary(rows, output_dir, args.title)
    print(f"Wrote held-out summaries to {output_dir}")


if __name__ == "__main__":
    main()
