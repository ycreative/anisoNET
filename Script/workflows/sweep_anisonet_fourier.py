"""Run a targeted Fourier/PDE/smoothness sweep for anisoNET."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from anisonet.metrics import evaluate_field
from anisonet.pinn import PINNProfile, robust_normalize, solve_scalar_reaction_diffusion
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


QUICK_SWEEP = [
    {"name": "sigma3_smooth001_pde006", "sigma": 3.0, "smooth": 0.001, "pde": 0.06, "data": 7.0},
    {"name": "sigma4_smooth001_pde006", "sigma": 4.0, "smooth": 0.001, "pde": 0.06, "data": 7.0},
    {"name": "sigma4_smooth002_pde008", "sigma": 4.0, "smooth": 0.002, "pde": 0.08, "data": 6.0},
    {"name": "sigma5_smooth002_pde008", "sigma": 5.0, "smooth": 0.002, "pde": 0.08, "data": 6.0},
    {"name": "sigma5_smooth005_pde010", "sigma": 5.0, "smooth": 0.005, "pde": 0.10, "data": 6.0},
    {"name": "sigma6_smooth002_pde012", "sigma": 6.0, "smooth": 0.002, "pde": 0.12, "data": 8.0},
]

REFINED_SWEEP = [
    {"name": "sigma45_smooth0015_pde008", "sigma": 4.5, "smooth": 0.0015, "pde": 0.08, "data": 8.0},
    {"name": "sigma5_smooth0015_pde010", "sigma": 5.0, "smooth": 0.0015, "pde": 0.10, "data": 8.0},
    {"name": "sigma55_smooth002_pde010", "sigma": 5.5, "smooth": 0.0020, "pde": 0.10, "data": 8.0},
    {"name": "sigma55_smooth003_pde012", "sigma": 5.5, "smooth": 0.0030, "pde": 0.12, "data": 8.0},
    {"name": "sigma6_smooth003_pde012", "sigma": 6.0, "smooth": 0.0030, "pde": 0.12, "data": 8.0},
    {"name": "sigma6_smooth004_pde012", "sigma": 6.0, "smooth": 0.0040, "pde": 0.12, "data": 8.0},
    {"name": "sigma6_smooth002_pde014", "sigma": 6.0, "smooth": 0.0020, "pde": 0.14, "data": 8.0},
    {"name": "sigma65_smooth002_pde012", "sigma": 6.5, "smooth": 0.0020, "pde": 0.12, "data": 8.0},
]

SWEEP_SETS = {
    "quick": QUICK_SWEEP,
    "refined": REFINED_SWEEP,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted anisoNET Fourier sweep.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--target-gene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--num-domain", type=int, default=2000)
    parser.add_argument("--hidden-width", type=int, default=128)
    parser.add_argument("--hidden-depth", type=int, default=3)
    parser.add_argument("--fourier-features", type=int, default=48)
    parser.add_argument("--k", type=float, default=2.5)
    parser.add_argument("--noise-threshold", type=float, default=0.05)
    parser.add_argument("--sweep-set", choices=sorted(SWEEP_SETS), default="quick")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight_dir = Path(args.preflight_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    source_grid = np.load(preflight_dir / "source_grid.npy")
    diffusion_grid = np.load(preflight_dir / "diffusion_grid.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")

    sample = load_visium_lite(args.sample_dir)
    source_values = clip_and_normalize(normalized_gene_vector(sample, args.target_gene), percentile=99.0)

    rows = []
    for item in SWEEP_SETS[args.sweep_set]:
        run_dir = output_dir / item["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        profile = PINNProfile(
            name=item["name"],
            hidden_width=args.hidden_width,
            hidden_depth=args.hidden_depth,
            num_domain=args.num_domain,
            num_boundary=100,
            iterations=args.iterations,
            learning_rate=8e-4,
            data_weight=item["data"],
            boundary_weight=0.02,
            pde_weight=item["pde"],
            background_weight=1.5,
            smoothness_weight=item["smooth"],
            display_every=max(args.iterations // 2, 1),
            network="fourier",
            fourier_features=args.fourier_features,
            fourier_sigma=item["sigma"],
        )
        with (run_dir / "profile.json").open("w", encoding="utf-8") as handle:
            json.dump(profile.__dict__, handle, indent=2)

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
        grid_norm = robust_normalize(result.grid_prediction, percentile=99.5)
        grid_clean = np.where(grid_norm < args.noise_threshold, 0.0, grid_norm).astype(np.float32)
        grid_masked = np.where(tissue_mask, grid_clean, 0.0).astype(np.float32)

        np.save(run_dir / "pinn_grid_prediction_raw.npy", result.grid_prediction)
        np.save(run_dir / "pinn_grid_prediction_clean_tissue_masked.npy", grid_masked)
        with (run_dir / "pinn_history.json").open("w", encoding="utf-8") as handle:
            json.dump(result.history, handle, indent=2)

        metrics = evaluate_field(
            coords_norm=coords_norm,
            source_spot_values=source_values,
            prediction_grid=grid_masked,
            barrier_grid=barrier_grid,
            tissue_mask=tissue_mask,
        ).to_dict()
        row = {
            "run": item["name"],
            "sigma": item["sigma"],
            "smoothness_weight": item["smooth"],
            "pde_weight": item["pde"],
            "data_weight": item["data"],
            **metrics,
        }
        rows.append(row)
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)
        print(json.dumps(row, indent=2), flush=True)

    csv_path = output_dir / "sweep_metrics.csv"
    columns = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote sweep metrics to {csv_path}")


if __name__ == "__main__":
    main()
