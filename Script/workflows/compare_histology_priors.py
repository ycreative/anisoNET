"""Compare H&E brightness and hematoxylin structural priors."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
DEFAULT_ROOT = PROJECT_ROOT / "codexAnalysis" / "histology_prior" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare histology-derived anisoNET priors.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument("--target-name", default="Apoe_CNS_Myelin")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    return parser.parse_args()


def summarize_prior(name: str, prior_dir: Path, tissue_mask: np.ndarray) -> dict[str, object]:
    histology = np.load(prior_dir / "histology_resistance_grid.npy")
    diffusion = np.load(prior_dir / "diffusion_grid.npy")
    barrier = np.load(prior_dir / "barrier_grid.npy")
    tissue_histology = histology[tissue_mask]
    tissue_diffusion = diffusion[tissue_mask]
    return {
        "prior": name,
        "histology_resistance_mean": float(np.mean(tissue_histology)),
        "histology_resistance_p50": float(np.percentile(tissue_histology, 50)),
        "histology_resistance_p95": float(np.percentile(tissue_histology, 95)),
        "diffusion_mean": float(np.mean(tissue_diffusion)),
        "diffusion_min": float(np.min(tissue_diffusion)),
        "diffusion_p05": float(np.percentile(tissue_diffusion, 5)),
        "diffusion_p95": float(np.percentile(tissue_diffusion, 95)),
        "diffusion_max": float(np.max(tissue_diffusion)),
        "barrier_histology_pearson": pearsonr_safe(barrier[tissue_mask], histology[tissue_mask]),
    }


def pearsonr_safe(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(base_dir: Path, tissue_mask: np.ndarray) -> None:
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
    priors = ["brightness", "hematoxylin"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True)
    panels = [
        ("histology_resistance_grid.npy", "H&E structural resistance", "cividis"),
        ("diffusion_grid.npy", "Combined diffusion", "viridis"),
        ("resistance_grid.npy", "Resistance (1 / D)", "magma"),
    ]
    for row_idx, prior in enumerate(priors):
        prior_dir = base_dir / prior
        for col_idx, (filename, label, cmap) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            arr = np.load(prior_dir / filename)
            masked = np.where(tissue_mask, arr, np.nan)
            vmax = float(np.nanpercentile(masked, 99.0))
            vmin = float(np.nanpercentile(masked, 1.0))
            image = ax.imshow(masked, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{prior}: {label}", fontsize=7, pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            cbar = fig.colorbar(image, ax=ax, fraction=0.040, pad=0.010)
            cbar.outline.set_linewidth(0.3)
            cbar.ax.tick_params(labelsize=5, length=1.5, width=0.3)
    fig.savefig(base_dir / "histology_prior_comparison.pdf", bbox_inches="tight")
    fig.savefig(base_dir / "histology_prior_comparison.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir = Path(args.root) / args.sample / args.target_name
    tissue_mask = np.load(base_dir / "brightness" / "tissue_mask.npy")
    rows = [
        summarize_prior("brightness", base_dir / "brightness", tissue_mask),
        summarize_prior("hematoxylin", base_dir / "hematoxylin", tissue_mask),
    ]
    write_csv(rows, base_dir / "histology_prior_comparison_metrics.csv")
    plot_comparison(base_dir, tissue_mask)
    print(f"Wrote histology prior comparison to {base_dir}")


if __name__ == "__main__":
    main()
