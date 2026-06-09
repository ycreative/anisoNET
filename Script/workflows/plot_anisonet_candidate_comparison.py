"""Plot publication-style comparison of selected anisoNET candidate fields."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot selected anisoNET candidate fields.")
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--candidate", action="append", required=True, help="Label=path_to_grid.npy")
    parser.add_argument("--metrics-csv", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="anisoNET candidate comparison")
    return parser.parse_args()


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--candidate must use Label=path_to_grid.npy")
    label, path = value.split("=", 1)
    return label, Path(path)


def load_metrics(paths: list[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for value in paths:
        path = Path(value)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key = row.get("run") or row.get("method") or row.get("candidate")
                if key:
                    out[key] = row
    return out


def metric_label(label: str, metrics: dict[str, dict[str, str]]) -> str:
    row = metrics.get(label, {})
    if not row:
        return label
    pearson = row.get("spot_pearson_source", "")
    mse = row.get("spot_mse_source", "")
    rough = row.get("roughness_grad_p95", "")
    if not pearson or not mse:
        return label
    return f"{label}\nr={float(pearson):.3f}, MSE={float(mse):.4f}, R95={float(rough):.3f}"


def main() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    args = parse_args()
    preflight_dir = Path(args.preflight_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = load_metrics(args.metrics_csv)

    source_grid = np.load(preflight_dir / "source_grid.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    candidates = [(label, np.load(path)) for label, path in map(parse_candidate, args.candidate)]

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    panels = [
        ("Apoe source grid", source_grid, "magma"),
        ("CNS myelin barrier prior", barrier_grid, "Reds"),
        *[(metric_label(label, metrics), grid, "magma") for label, grid in candidates],
    ]
    n_panels = len(panels)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2, 2.35 * n_rows), constrained_layout=True)
    axes_arr = np.asarray(axes).reshape(-1)

    field_values = np.concatenate([grid[tissue_mask].reshape(-1) for _, grid in candidates])
    field_vmax = float(np.percentile(field_values, 99.5)) if field_values.size else 1.0

    for ax, (title, values, cmap) in zip(axes_arr, panels):
        if cmap == "magma" and title != "Apoe source grid":
            image = ax.imshow(values, origin="lower", cmap=cmap, vmin=0.0, vmax=field_vmax)
        else:
            image = ax.imshow(values, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=7, pad=3)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.015)
        cbar.outline.set_linewidth(0.3)
        cbar.ax.tick_params(labelsize=5, length=1.5, width=0.3)

    for ax in axes_arr[n_panels:]:
        ax.axis("off")

    fig.suptitle(args.title, fontsize=8, y=1.02)
    fig.savefig(output_dir / "candidate_comparison.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "candidate_comparison.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote candidate comparison to {output_dir}")


if __name__ == "__main__":
    main()
