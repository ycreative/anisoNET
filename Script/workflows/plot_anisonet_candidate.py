"""Plot a candidate anisoNET field against preflight inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot an anisoNET candidate field.")
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--candidate-grid", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="anisoNET candidate")
    return parser.parse_args()


def main() -> None:
    import matplotlib.pyplot as plt

    args = parse_args()
    preflight_dir = Path(args.preflight_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_grid = np.load(preflight_dir / "source_grid.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    diffusion_grid = np.load(preflight_dir / "diffusion_grid.npy")
    candidate = np.load(args.candidate_grid)

    panels = [
        ("Source", source_grid, "magma"),
        ("Barrier prior", barrier_grid, "Reds"),
        ("Diffusion coefficient", diffusion_grid, "viridis"),
        ("Candidate field", candidate, "magma"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.0), constrained_layout=True)
    for ax, (title, values, cmap) in zip(axes, panels):
        image = ax.imshow(values, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6, length=2)

    fig.suptitle(args.title, fontsize=9)
    fig.savefig(output_dir / "candidate_fields.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "candidate_fields.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
