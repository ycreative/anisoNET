"""Alpha sensitivity benchmark for anisoNET transcriptomic barrier strength."""

from __future__ import annotations

import os

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.metrics import evaluate_field
from anisonet.pinn import get_profile, robust_normalize, solve_scalar_reaction_diffusion
from anisonet.postprocessing import mask_field, normalize_masked_field, smooth_inside_mask
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


PROCESSED_ROOT = PROJECT_ROOT / "codexAnalysis" / "processed_visium" / "brain_aging_gse193107"
PREFLIGHT_ROOT = PROJECT_ROOT / "codexAnalysis" / "preflight" / "brain_aging_gse193107"
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "alpha_sensitivity" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run anisoNET alpha sensitivity benchmark.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument("--target-gene", action="append", default=["Apoe", "Gfap"])
    parser.add_argument("--alpha", type=float, nargs="*", default=[0.0, 2.0, 4.0, 6.0, 8.0])
    parser.add_argument("--reference-alpha", type=float, default=4.0)
    parser.add_argument("--profile", default="fourier_refined_16g")
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--num-domain", type=int, default=1600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--postprocess-sigma", type=float, default=0.7)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def reconstruct_optical_diffusion(
    diffusion_grid: np.ndarray,
    barrier_grid: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    reference_alpha: float,
) -> np.ndarray:
    optical = np.zeros_like(diffusion_grid, dtype=np.float32)
    optical[tissue_mask] = diffusion_grid[tissue_mask] / np.exp(-reference_alpha * barrier_grid[tissue_mask])
    background = float(np.median(diffusion_grid[~tissue_mask])) if np.any(~tissue_mask) else 0.005
    return np.where(tissue_mask, optical, background).astype(np.float32)


def build_alpha_diffusion(
    optical_diffusion: np.ndarray,
    barrier_grid: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    background = float(np.median(optical_diffusion[~tissue_mask])) if np.any(~tissue_mask) else 0.005
    fused = optical_diffusion * np.exp(-float(alpha) * barrier_grid)
    return np.where(tissue_mask, fused, background).astype(np.float32)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_target(args: argparse.Namespace, target_gene: str) -> list[dict[str, object]]:
    target_name = f"{target_gene}_CNS_Myelin"
    sample_dir = PROCESSED_ROOT / args.sample
    preflight_dir = PREFLIGHT_ROOT / args.sample / target_name
    output_dir = OUTPUT_ROOT / args.sample / target_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_visium_lite(sample_dir)
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    source_grid = np.load(preflight_dir / "source_grid.npy")
    diffusion_grid = np.load(preflight_dir / "diffusion_grid.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    source_values = clip_and_normalize(normalized_gene_vector(sample, target_gene), percentile=99.0)
    optical = reconstruct_optical_diffusion(
        diffusion_grid,
        barrier_grid,
        tissue_mask,
        reference_alpha=args.reference_alpha,
    )
    np.save(output_dir / "reconstructed_optical_diffusion.npy", optical)

    base_profile = get_profile(args.profile)
    profile = replace(
        base_profile,
        name=f"{args.profile}_alpha_sensitivity",
        iterations=args.iterations,
        num_domain=args.num_domain,
        display_every=max(args.iterations // 2, 1),
    )
    rows = []
    for alpha in args.alpha:
        label = f"alpha{str(alpha).replace('.', 'p')}"
        alpha_dir = output_dir / label
        alpha_dir.mkdir(parents=True, exist_ok=True)
        field_path = alpha_dir / "field_gauss07.npy"
        diffusion_variant = build_alpha_diffusion(
            optical,
            barrier_grid,
            tissue_mask,
            alpha=alpha,
        )
        np.save(alpha_dir / "diffusion_grid.npy", diffusion_variant)
        if args.skip_existing and field_path.exists():
            field = np.load(field_path)
        else:
            print(f"Running alpha sensitivity: {target_gene} alpha={alpha}", flush=True)
            result = solve_scalar_reaction_diffusion(
                coords_norm,
                source_values,
                diffusion_variant,
                source_grid,
                profile=profile,
                tissue_mask=tissue_mask,
                device=args.device,
                seed=0,
                prediction_grid_size=diffusion_variant.shape[0],
            )
            masked = mask_field(np.clip(robust_normalize(result.grid_prediction, percentile=99.5), 0.0, 1.0), tissue_mask)
            np.save(alpha_dir / "field_masked.npy", masked)
            if args.postprocess_sigma > 0:
                field = normalize_masked_field(
                    smooth_inside_mask(masked, tissue_mask, sigma=args.postprocess_sigma),
                    tissue_mask,
                    percentile=99.5,
                )
            else:
                field = masked
            np.save(field_path, field)
            with (alpha_dir / "pinn_history.json").open("w", encoding="utf-8") as handle:
                json.dump(result.history, handle, indent=2)

        metrics = evaluate_field(
            coords_norm=coords_norm,
            source_spot_values=source_values,
            prediction_grid=field,
            barrier_grid=barrier_grid,
            tissue_mask=tissue_mask,
        ).to_dict()
        rows.append(
            {
                "sample": args.sample,
                "target_gene": target_gene,
                "alpha": float(alpha),
                "diffusion_mean_tissue": float(np.mean(diffusion_variant[tissue_mask])),
                "diffusion_min_tissue": float(np.min(diffusion_variant[tissue_mask])),
                "diffusion_max_tissue": float(np.max(diffusion_variant[tissue_mask])),
                **metrics,
            }
        )
    write_csv(rows, output_dir / "alpha_sensitivity_metrics.csv")
    return rows


def main() -> None:
    args = parse_args()
    all_rows = []
    for target_gene in args.target_gene:
        all_rows.extend(run_target(args, target_gene))
    summary_dir = OUTPUT_ROOT / args.sample
    write_csv(all_rows, summary_dir / "alpha_sensitivity_summary.csv")
    print(f"Wrote alpha sensitivity summary to {summary_dir}", flush=True)


if __name__ == "__main__":
    main()

