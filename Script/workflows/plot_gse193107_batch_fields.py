"""Plot anisoNET batch field montage for GSE193107 brain sections."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
PREFLIGHT_ROOT = PROJECT_ROOT / "codexAnalysis" / "preflight" / "brain_aging_gse193107"
PINN_ROOT = PROJECT_ROOT / "codexAnalysis" / "pinn" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot GSE193107 anisoNET batch fields.")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-name", default="Apoe_CNS_Myelin")
    parser.add_argument("--run-name", default="fourier_refined_16g_gauss07_batch")
    parser.add_argument("--field-name", default="pinn_grid_prediction_postprocessed.npy")
    parser.add_argument("--title", default="anisoNET GSE193107 Apoe/CNS-myelin fields")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row["field_type"] == "gauss07"]


def sample_label(sample: str) -> str:
    stem = sample.replace("GSM577", "G")
    condition = "Young" if "_Young_" in sample else "Old"
    replicate = sample.split("_brain_")[-1]
    return f"{condition} {replicate}\n{stem.split('_')[0]}"


def main() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    args = parse_args()
    rows = read_rows(Path(args.metrics_csv))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fields = []
    for row in rows:
        sample = row["sample"]
        field_path = PINN_ROOT / sample / args.target_name / args.run_name / args.field_name
        fields.append((row, np.load(field_path)))

    tissue_values = []
    for row, field in fields:
        tissue_mask = np.load(PREFLIGHT_ROOT / row["sample"] / args.target_name / "tissue_mask.npy")
        tissue_values.append(field[tissue_mask].reshape(-1))
    vmax = float(np.percentile(np.concatenate(tissue_values), 99.5))

    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.7), constrained_layout=True)
    for ax, (row, field) in zip(axes.reshape(-1), fields):
        image = ax.imshow(field, origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(
            f"{sample_label(row['sample'])}\nr={float(row['spot_pearson_source']):.3f}, R95={float(row['roughness_grad_p95']):.3f}",
            fontsize=6,
            pad=2,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012)
    cbar.outline.set_linewidth(0.3)
    cbar.ax.tick_params(labelsize=5, length=1.5, width=0.3)
    fig.suptitle(args.title, fontsize=8, y=1.03)
    fig.savefig(output_dir / "batch_field_montage.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "batch_field_montage.png", dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote batch field montage to {output_dir}")


if __name__ == "__main__":
    main()
