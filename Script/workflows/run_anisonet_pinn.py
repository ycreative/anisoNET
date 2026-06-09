"""Run anisoNET PINN inference from a preflight directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.pinn import get_profile, robust_normalize, solve_scalar_reaction_diffusion
from anisonet.postprocessing import mask_field, normalize_masked_field, smooth_inside_mask
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run scalar anisoNET PINN inference.")
    parser.add_argument("--sample-dir", required=True, help="Standardized Visium sample directory.")
    parser.add_argument("--preflight-dir", required=True, help="Directory produced by run_anisonet_preflight.py.")
    parser.add_argument("--target-gene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--profile",
        default="local_16g",
        choices=[
            "debug_fit",
            "debug_fourier",
            "fourier_balanced",
            "fourier_refined_16g",
            "fourier_refined_low_pde_16g",
            "smoke",
            "local_16g",
            "l20_24g",
        ],
    )
    parser.add_argument("--k", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda or cpu. Defaults to auto.")
    parser.add_argument("--source-percentile", type=float, default=99.0)
    parser.add_argument("--output-percentile", type=float, default=99.5)
    parser.add_argument("--noise-threshold", type=float, default=0.05)
    parser.add_argument(
        "--postprocess-sigma",
        type=float,
        default=0.0,
        help="Optional tissue-aware Gaussian smoothing sigma. Use 0 to disable.",
    )
    return parser.parse_args()


def save_pinn_figure(
    output_dir: Path,
    *,
    source_grid: np.ndarray,
    diffusion_grid: np.ndarray,
    prediction_grid: np.ndarray,
    normalized_prediction_grid: np.ndarray,
    target_gene: str,
    profile_name: str,
) -> None:
    import matplotlib.pyplot as plt

    panels = [
        (f"Source: {target_gene}", source_grid, "magma"),
        ("Diffusion coefficient", diffusion_grid, "viridis"),
        ("PINN prediction", prediction_grid, "magma"),
        ("Normalized prediction", normalized_prediction_grid, "magma"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.0), constrained_layout=True)
    for ax, (title, values, cmap) in zip(axes, panels):
        image = ax.imshow(values, origin="lower", cmap=cmap)
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6, length=2)

    fig.suptitle(f"anisoNET scalar PINN | profile: {profile_name}", fontsize=9)
    fig.savefig(output_dir / "pinn_fields.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pinn_fields.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    preflight_dir = Path(args.preflight_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    source_grid = np.load(preflight_dir / "source_grid.npy")
    diffusion_grid = np.load(preflight_dir / "diffusion_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")

    sample = load_visium_lite(args.sample_dir)
    source_values = clip_and_normalize(
        normalized_gene_vector(sample, args.target_gene),
        percentile=args.source_percentile,
    )

    profile = get_profile(args.profile)
    result = solve_scalar_reaction_diffusion(
        coords_norm,
        source_values,
        diffusion_grid,
        source_grid,
        profile=profile,
        tissue_mask=tissue_mask,
        k=args.k,
        device=args.device,
        seed=args.seed,
        prediction_grid_size=diffusion_grid.shape[0],
    )

    spot_norm = robust_normalize(result.spot_prediction, percentile=args.output_percentile)
    grid_norm = robust_normalize(result.grid_prediction, percentile=args.output_percentile)
    spot_clean = np.where(spot_norm < args.noise_threshold, 0.0, spot_norm)
    grid_clean = np.where(grid_norm < args.noise_threshold, 0.0, grid_norm)
    grid_clean_tissue_masked = mask_field(grid_clean, tissue_mask)
    grid_postprocessed = None
    if args.postprocess_sigma > 0:
        grid_smoothed = smooth_inside_mask(grid_clean_tissue_masked, tissue_mask, sigma=args.postprocess_sigma)
        grid_postprocessed = normalize_masked_field(
            grid_smoothed,
            tissue_mask,
            percentile=args.output_percentile,
        )

    np.save(output_dir / "pinn_spot_prediction_raw.npy", result.spot_prediction)
    np.save(output_dir / "pinn_spot_prediction_norm.npy", spot_norm)
    np.save(output_dir / "pinn_spot_prediction_clean.npy", spot_clean)
    np.save(output_dir / "pinn_grid_prediction_raw.npy", result.grid_prediction)
    np.save(output_dir / "pinn_grid_prediction_norm.npy", grid_norm)
    np.save(output_dir / "pinn_grid_prediction_clean.npy", grid_clean)
    np.save(output_dir / "pinn_grid_prediction_clean_tissue_masked.npy", grid_clean_tissue_masked)
    if grid_postprocessed is not None:
        np.save(output_dir / "pinn_grid_prediction_postprocessed.npy", grid_postprocessed)
    with (output_dir / "pinn_history.json").open("w", encoding="utf-8") as handle:
        json.dump(result.history, handle, indent=2)

    metrics = {
        "sample_dir": str(Path(args.sample_dir).resolve()),
        "preflight_dir": str(preflight_dir.resolve()),
        "target_gene": args.target_gene,
        "profile": args.profile,
        "k": float(args.k),
        "seed": int(args.seed),
        "n_spots": int(coords_norm.shape[0]),
        "grid_size": int(diffusion_grid.shape[0]),
        "spot_prediction_min": float(np.min(result.spot_prediction)),
        "spot_prediction_max": float(np.max(result.spot_prediction)),
        "grid_prediction_min": float(np.min(result.grid_prediction)),
        "grid_prediction_max": float(np.max(result.grid_prediction)),
        "spot_clean_nonzero_fraction": float(np.mean(spot_clean > 0)),
        "postprocess_sigma": float(args.postprocess_sigma),
        "has_postprocessed_grid": bool(grid_postprocessed is not None),
    }
    with (output_dir / "pinn_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    save_pinn_figure(
        output_dir,
        source_grid=source_grid,
        diffusion_grid=diffusion_grid,
        prediction_grid=result.grid_prediction,
        normalized_prediction_grid=grid_postprocessed if grid_postprocessed is not None else grid_clean_tissue_masked,
        target_gene=args.target_gene,
        profile_name=args.profile,
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
