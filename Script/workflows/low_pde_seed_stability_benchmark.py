"""Check random-seed stability for the default and low-PDE profiles."""

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
PREFLIGHT_ROOT = PROJECT_ROOT / "codexAnalysis" / "histology_prior" / "brain_aging_gse193107"
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "loss_weight_sensitivity" / "brain_aging_gse193107" / "low_pde_seed_stability"

DEFAULT_SAMPLES = [
    "GSM5773453_Young_mouse_brain_A1-1",
    "GSM5773457_Old_mouse_brain_A1-2",
]

WEIGHT_SETTINGS = [
    {"setting": "default", "data_weight": 8.0, "pde_weight": 0.12},
    {"setting": "low_pde", "data_weight": 8.0, "pde_weight": 0.04},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run low-PDE random-seed stability benchmark.")
    parser.add_argument("--samples", nargs="*", default=DEFAULT_SAMPLES)
    parser.add_argument("--target-genes", nargs="*", default=["Apoe", "Gfap"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--profile", default="fourier_refined_16g")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--histology-prior", default="brightness", choices=["brightness", "hematoxylin"])
    parser.add_argument("--postprocess-sigma", type=float, default=0.7)
    parser.add_argument("--source-percentile", type=float, default=99.0)
    parser.add_argument("--output-percentile", type=float, default=99.5)
    parser.add_argument("--noise-threshold", type=float, default=0.05)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def condition(sample: str) -> str:
    if "_Young_" in sample:
        return "Young"
    if "_Old_" in sample:
        return "Old"
    return "Unknown"


def main() -> None:
    args = parse_args()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for sample in args.samples:
        for target_gene in args.target_genes:
            rows.extend(run_sample_target(args, sample, target_gene))

    summary_path = OUTPUT_ROOT / "low_pde_seed_stability_metrics.csv"
    write_csv(rows, summary_path)
    manifest = {
        "samples": args.samples,
        "target_genes": args.target_genes,
        "seeds": args.seeds,
        "profile": args.profile,
        "settings": WEIGHT_SETTINGS,
        "histology_prior": args.histology_prior,
        "postprocess_sigma": args.postprocess_sigma,
        "summary_csv": str(summary_path),
    }
    (OUTPUT_ROOT / "low_pde_seed_stability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def run_sample_target(args: argparse.Namespace, sample: str, target_gene: str) -> list[dict[str, object]]:
    sample_dir = PROCESSED_ROOT / sample
    preflight_dir = PREFLIGHT_ROOT / sample / f"{target_gene}_CNS_Myelin" / args.histology_prior
    output_dir = OUTPUT_ROOT / sample / f"{target_gene}_CNS_Myelin" / args.histology_prior
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_data = load_visium_lite(sample_dir)
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    source_grid = np.load(preflight_dir / "source_grid.npy")
    diffusion_grid = np.load(preflight_dir / "diffusion_grid.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    source_values = clip_and_normalize(
        normalized_gene_vector(sample_data, target_gene),
        percentile=args.source_percentile,
    )

    base_profile = get_profile(args.profile)
    rows = []
    for seed in args.seeds:
        for weights in WEIGHT_SETTINGS:
            setting = str(weights["setting"])
            setting_dir = output_dir / f"seed{seed}" / setting
            setting_dir.mkdir(parents=True, exist_ok=True)
            masked_path = setting_dir / "field_masked.npy"
            gauss_path = setting_dir / "field_gauss07.npy"
            history_path = setting_dir / "pinn_history.json"

            if args.skip_existing and masked_path.exists() and gauss_path.exists():
                masked = np.load(masked_path)
                gauss = np.load(gauss_path)
            else:
                profile = replace(
                    base_profile,
                    name=f"{args.profile}_{setting}_{sample}_{target_gene}_seed{seed}",
                    data_weight=float(weights["data_weight"]),
                    pde_weight=float(weights["pde_weight"]),
                )
                print(
                    f"Running seed stability {sample} {target_gene} seed={seed} {setting}: "
                    f"data={profile.data_weight:g}, pde={profile.pde_weight:g}",
                    flush=True,
                )
                result = solve_scalar_reaction_diffusion(
                    coords_norm,
                    source_values,
                    diffusion_grid,
                    source_grid,
                    profile=profile,
                    tissue_mask=tissue_mask,
                    device=args.device,
                    seed=seed,
                    prediction_grid_size=diffusion_grid.shape[0],
                )
                grid_norm = robust_normalize(result.grid_prediction, percentile=args.output_percentile)
                grid_clean = np.where(grid_norm < args.noise_threshold, 0.0, grid_norm)
                masked = mask_field(grid_clean, tissue_mask)
                if args.postprocess_sigma > 0:
                    gauss = normalize_masked_field(
                        smooth_inside_mask(masked, tissue_mask, sigma=args.postprocess_sigma),
                        tissue_mask,
                        percentile=args.output_percentile,
                    )
                else:
                    gauss = masked
                np.save(masked_path, masked.astype(np.float32))
                np.save(gauss_path, gauss.astype(np.float32))
                history_path.write_text(json.dumps(result.history, indent=2), encoding="utf-8")

            for field_type, field in [("masked", masked), ("gauss07", gauss)]:
                metrics = evaluate_field(
                    coords_norm=coords_norm,
                    source_spot_values=source_values,
                    prediction_grid=field,
                    barrier_grid=barrier_grid,
                    tissue_mask=tissue_mask,
                ).to_dict()
                rows.append(
                    {
                        "sample": sample,
                        "condition": condition(sample),
                        "target_gene": target_gene,
                        "seed": seed,
                        "setting": setting,
                        "field_type": field_type,
                        "histology_prior": args.histology_prior,
                        "data_weight": float(weights["data_weight"]),
                        "pde_weight": float(weights["pde_weight"]),
                        "data_to_pde_weight_ratio": float(weights["data_weight"]) / float(weights["pde_weight"]),
                        **metrics,
                    }
                )

    write_csv(rows, output_dir / "low_pde_seed_stability_metrics.csv")
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = [
        "sample",
        "condition",
        "target_gene",
        "seed",
        "setting",
        "field_type",
        "histology_prior",
        "data_weight",
        "pde_weight",
        "data_to_pde_weight_ratio",
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "roughness_grad_mean",
        "spot_pearson_barrier",
        "high_to_low_barrier_prediction_ratio",
        "background_to_tissue_ratio",
    ]
    extras = sorted({key for row in rows for key in row if key not in columns})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + extras)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

