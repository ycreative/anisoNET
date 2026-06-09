"""Plot barrier perturbation fields for anisoNET."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
PREFLIGHT_ROOT = PROJECT_ROOT / "codexAnalysis" / "preflight" / "brain_aging_gse193107"
PERTURB_ROOT = PROJECT_ROOT / "codexAnalysis" / "barrier_perturbation" / "brain_aging_gse193107"


VARIANTS = (
    ("original_barrier", "Original barrier"),
    ("no_transcript_barrier", "No transcript barrier"),
    ("amplified_barrier", "Amplified barrier"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot anisoNET barrier perturbation fields.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument("--target-gene", action="append", default=["Apoe", "Gfap"])
    parser.add_argument("--output-dir", default=str(PERTURB_ROOT / "GSM5773457_Old_mouse_brain_A1-2"))
    return parser.parse_args()


def read_metrics(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {(row["target_gene"], row["variant"]): row for row in rows}


def main() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(PERTURB_ROOT / args.sample / "barrier_perturbation_summary.csv")

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    nrows = len(args.target_gene)
    fig, axes = plt.subplots(nrows, len(VARIANTS) + 1, figsize=(7.2, 2.25 * nrows), constrained_layout=True)
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for row_idx, target_gene in enumerate(args.target_gene):
        target_name = f"{target_gene}_CNS_Myelin"
        preflight_dir = PREFLIGHT_ROOT / args.sample / target_name
        barrier = np.load(preflight_dir / "barrier_grid.npy")
        tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
        barrier_panel = np.where(tissue_mask, barrier, np.nan)
        barrier_vmax = float(np.nanpercentile(barrier_panel, 99.0))
        fields = []
        for variant, _ in VARIANTS:
            field = np.load(PERTURB_ROOT / target_name / args.sample / f"{variant}_field.npy")
            fields.append(field)
        vmax = float(np.percentile(np.concatenate([f[tissue_mask] for f in fields]), 99.5))

        ax = axes[row_idx, 0]
        barrier_image = ax.imshow(barrier_panel, origin="lower", cmap="viridis", vmin=0, vmax=barrier_vmax)
        ax.set_title(f"{target_gene}\nCNS-myelin barrier", fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        for col_idx, ((variant, label), field) in enumerate(zip(VARIANTS, fields), start=1):
            row = metrics[(target_gene, variant)]
            ax = axes[row_idx, col_idx]
            image = ax.imshow(field, origin="lower", cmap="magma", vmin=0, vmax=vmax)
            ax.set_title(
                f"{label}\nr={float(row['spot_pearson_source']):.3f}, R95={float(row['roughness_grad_p95']):.3f}",
                fontsize=7,
                pad=2,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

        cbar = fig.colorbar(image, ax=axes[row_idx, 1:].ravel().tolist(), fraction=0.024, pad=0.010)
        cbar.outline.set_linewidth(0.3)
        cbar.ax.tick_params(labelsize=5, length=1.5, width=0.3)
        bcbar = fig.colorbar(barrier_image, ax=axes[row_idx, 0], fraction=0.050, pad=0.010)
        bcbar.outline.set_linewidth(0.3)
        bcbar.ax.tick_params(labelsize=5, length=1.5, width=0.3)

    fig.savefig(output_dir / "barrier_perturbation_fields.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "barrier_perturbation_fields.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote barrier perturbation field figure to {output_dir}")


if __name__ == "__main__":
    main()
