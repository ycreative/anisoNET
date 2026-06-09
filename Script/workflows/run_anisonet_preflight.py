"""Build anisoNET preprocessing grids for one Visium sample.

This workflow is intentionally lightweight: it validates local data paths and
exports the fields that will later be consumed by the PINN and benchmark
workflows.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.diffusion import (
    BARRIER_MODULES,
    build_barrier_grid,
    build_diffusion_grid,
    build_histology_resistance_grid,
    build_optical_diffusion_grid,
    diffusion_from_histology_resistance,
    build_source_grid,
    resistance_from_diffusion,
)
from anisonet.preprocessing import (
    build_spot_mask,
    build_tissue_mask_from_hires,
    clip_and_normalize,
    keep_largest_mask_component,
)
from anisonet.visium_io import (
    load_visium_lite,
    normalize_pixel_coordinates,
    normalized_gene_sum,
    normalized_gene_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anisoNET field-construction preflight for one sample.")
    parser.add_argument("--sample-dir", required=True, help="Path to a 10x Visium sample directory.")
    parser.add_argument("--target-gene", required=True, help="Source/target gene used to build S(x,y).")
    parser.add_argument("--output-dir", required=True, help="Directory for generated preflight outputs.")
    parser.add_argument("--barrier-genes", nargs="*", default=None, help="Barrier marker genes. If omitted, infer module.")
    parser.add_argument("--grid-size", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=4.0)
    parser.add_argument("--source-percentile", type=float, default=99.0)
    parser.add_argument("--barrier-percentile", type=float, default=99.0)
    parser.add_argument("--barrier-sigma", type=float, default=1.5)
    parser.add_argument("--source-sigma", type=float, default=1.5)
    parser.add_argument("--d-min", type=float, default=0.0002)
    parser.add_argument("--d-free", type=float, default=0.02)
    parser.add_argument(
        "--histology-prior",
        choices=["brightness", "hematoxylin"],
        default="brightness",
        help="H&E structural prior used for optical diffusion.",
    )
    parser.add_argument("--background-diffusion", type=float, default=0.005)
    parser.add_argument("--keep-largest-tissue-component", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-figure", action="store_true", help="Skip publication-style QC figure export.")
    return parser.parse_args()


def infer_lite_barrier_module(sample: object) -> tuple[str, tuple[str, ...]]:
    scores = {}
    active_by_module = {}
    for name, genes in BARRIER_MODULES.items():
        active = tuple(g for g in genes if g in sample.genes)
        active_by_module[name] = active
        if active:
            values, _ = normalized_gene_sum(sample, active)
            scores[name] = float(np.sum(values))
        else:
            scores[name] = 0.0
    best_name = max(scores, key=scores.get)
    if scores[best_name] <= 0:
        raise ValueError("No supported barrier module was detected in this sample")
    return best_name, active_by_module[best_name]


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_visium_lite(args.sample_dir)
    hires_image = sample.hires_image
    coords_norm = normalize_pixel_coordinates(sample, flip_y=True)

    if args.barrier_genes:
        barrier_module = "custom"
        barrier_genes = tuple(args.barrier_genes)
    else:
        barrier_module, barrier_genes = infer_lite_barrier_module(sample)

    barrier_values, active_barrier_genes = normalized_gene_sum(sample, barrier_genes)
    barrier_score = clip_and_normalize(barrier_values, percentile=args.barrier_percentile)
    source_score = clip_and_normalize(
        normalized_gene_vector(sample, args.target_gene),
        percentile=args.source_percentile,
    )

    tissue_from_image = build_tissue_mask_from_hires(
        hires_image,
        grid_size=args.grid_size,
        flip_y=True,
    )
    tissue_from_spots = build_spot_mask(
        coords_norm,
        grid_size=args.grid_size,
        dilation_iterations=1,
    )
    tissue_mask = tissue_from_image | tissue_from_spots
    if args.keep_largest_tissue_component:
        tissue_mask = keep_largest_mask_component(tissue_mask)

    histology_resistance = build_histology_resistance_grid(
        hires_image,
        grid_size=args.grid_size,
        mode=args.histology_prior,
        flip_y=True,
    )
    optical_diffusion = diffusion_from_histology_resistance(
        histology_resistance,
        d_min=args.d_min,
        d_free=args.d_free,
    )
    barrier_grid = build_barrier_grid(
        coords_norm,
        barrier_score,
        grid_size=args.grid_size,
        sigma=args.barrier_sigma,
    )
    diffusion_grid = build_diffusion_grid(
        optical_diffusion,
        barrier_grid,
        tissue_mask,
        alpha=args.alpha,
        background_diffusion=args.background_diffusion,
    )
    source_grid = build_source_grid(
        coords_norm,
        source_score,
        grid_size=args.grid_size,
        sigma=args.source_sigma,
    )
    resistance_grid = resistance_from_diffusion(diffusion_grid, mode="inverse")

    np.save(output_dir / "coords_norm.npy", coords_norm)
    np.save(output_dir / "tissue_mask.npy", tissue_mask)
    np.save(output_dir / "barrier_grid.npy", barrier_grid)
    np.save(output_dir / "source_grid.npy", source_grid)
    np.save(output_dir / "histology_resistance_grid.npy", histology_resistance)
    np.save(output_dir / "diffusion_grid.npy", diffusion_grid)
    np.save(output_dir / "resistance_grid.npy", resistance_grid)

    tissue_values = diffusion_grid[tissue_mask]
    metrics = {
        "sample_dir": str(Path(args.sample_dir).resolve()),
        "target_gene": args.target_gene,
        "barrier_module": barrier_module,
        "barrier_genes_requested": list(barrier_genes),
        "barrier_genes_active": list(active_barrier_genes),
        "n_spots": int(len(sample.barcodes)),
        "n_genes": int(len(sample.genes)),
        "grid_size": int(args.grid_size),
        "alpha": float(args.alpha),
        "histology_prior": args.histology_prior,
        "histology_resistance_mean_in_tissue": float(np.mean(histology_resistance[tissue_mask])),
        "histology_resistance_p95_in_tissue": float(np.percentile(histology_resistance[tissue_mask], 95)),
        "diffusion_min_in_tissue": float(np.min(tissue_values)),
        "diffusion_max_in_tissue": float(np.max(tissue_values)),
        "resistance_ratio_in_tissue": float(np.max(1.0 / tissue_values) / np.min(1.0 / tissue_values)),
        "barrier_grid_max": float(np.max(barrier_grid)),
        "source_grid_max": float(np.max(source_grid)),
    }
    with (output_dir / "preflight_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    if not args.skip_figure:
        save_preflight_figure(
            output_dir,
            barrier_grid=barrier_grid,
            source_grid=source_grid,
            histology_resistance=histology_resistance,
            diffusion_grid=diffusion_grid,
            resistance_grid=resistance_grid,
            target_gene=args.target_gene,
            barrier_label="+".join(active_barrier_genes),
        )

    print(json.dumps(metrics, indent=2))


def save_preflight_figure(
    output_dir: Path,
    *,
    barrier_grid: np.ndarray,
    source_grid: np.ndarray,
    histology_resistance: np.ndarray,
    diffusion_grid: np.ndarray,
    resistance_grid: np.ndarray,
    target_gene: str,
    barrier_label: str,
) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("Barrier prior", barrier_grid, "Reds"),
        (f"Source: {target_gene}", source_grid, "magma"),
        ("H&E structural resistance", histology_resistance, "cividis"),
        ("Diffusion coefficient", diffusion_grid, "viridis"),
        ("Resistance (1 / D)", resistance_grid, "cividis"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.0), constrained_layout=True)
    for ax, (title, values, cmap) in zip(axes, panels):
        image = ax.imshow(values, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6, length=2)

    fig.suptitle(f"anisoNET preflight fields | barrier: {barrier_label}", fontsize=9)
    fig.savefig(output_dir / "preflight_fields.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "preflight_fields.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
