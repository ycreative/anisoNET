"""Held-out spot benchmark for anisoNET and traditional interpolation baselines."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from anisonet.diffusion import build_source_grid
from anisonet.metrics import field_roughness_metrics, mse, pearsonr_safe, sample_grid_at_spots
from anisonet.pinn import get_profile, robust_normalize, solve_scalar_reaction_diffusion
from anisonet.postprocessing import mask_field, normalize_masked_field, smooth_inside_mask
from anisonet.preprocessing import clip_and_normalize, spot_grid_indices
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
PROCESSED_ROOT = PROJECT_ROOT / "codexAnalysis" / "processed_visium" / "brain_aging_gse193107"
PREFLIGHT_ROOT = PROJECT_ROOT / "codexAnalysis" / "preflight" / "brain_aging_gse193107"
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "heldout" / "brain_aging_gse193107"


SAMPLES = [
    "GSM5773453_Young_mouse_brain_A1-1",
    "GSM5773454_Young_mouse_brain_B1-1",
    "GSM5773455_Young_mouse_brain_C1-1",
    "GSM5773456_Young_mouse_brain_D1-1",
    "GSM5773457_Old_mouse_brain_A1-2",
    "GSM5773458_Old_mouse_brain_B1-2",
    "GSM5773459_Old_mouse_brain_C1-2",
    "GSM5773460_Old_mouse_brain_D1-2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run held-out anisoNET benchmark on GSE193107.")
    parser.add_argument("--target-gene", required=True)
    parser.add_argument("--samples", nargs="*", default=SAMPLES)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", default="fourier_refined_16g")
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--num-domain", type=int, default=1600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--postprocess-sigma", type=float, default=0.7)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--pde-weight", type=float)
    parser.add_argument("--data-weight", type=float)
    parser.add_argument("--smoothness-weight", type=float)
    parser.add_argument("--fourier-sigma", type=float)
    parser.add_argument("--fourier-features", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def condition(sample: str) -> str:
    return "Young" if "_Young_" in sample else "Old" if "_Old_" in sample else "Unknown"


def split_indices(n: int, *, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    test_idx = np.sort(indices[:n_test])
    train_idx = np.sort(indices[n_test:])
    return train_idx, test_idx


def rasterize_spots(coords_norm: np.ndarray, values: np.ndarray, *, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    grid_x, grid_y = spot_grid_indices(coords_norm, grid_size=grid_size)
    value_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    support_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    value_grid[grid_y, grid_x] = np.asarray(values, dtype=np.float32)
    support_grid[grid_y, grid_x] = 1.0
    return value_grid, support_grid


def gaussian_interpolation_grid(
    coords_norm: np.ndarray,
    values: np.ndarray,
    *,
    grid_size: int,
    sigma: float,
    tissue_mask: np.ndarray,
) -> np.ndarray:
    value_grid, support_grid = rasterize_spots(coords_norm, values, grid_size=grid_size)
    numerator = gaussian_filter(value_grid, sigma=sigma)
    denominator = gaussian_filter(support_grid, sigma=sigma)
    grid = numerator / (denominator + 1e-8)
    return mask_field(np.clip(grid, 0.0, 1.0), tissue_mask)


def kd_interpolation_grid(
    coords_norm: np.ndarray,
    values: np.ndarray,
    *,
    grid_size: int,
    tissue_mask: np.ndarray,
    method: str,
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    axis = np.linspace(0, 1, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    grid_coords = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    tree = cKDTree(coords_norm)
    if method == "nearest":
        _, nearest_idx = tree.query(grid_coords, k=1)
        pred = values[nearest_idx]
    elif method == "idw":
        distances, neighbor_idx = tree.query(grid_coords, k=k)
        distances = np.maximum(distances, 1e-6)
        weights = 1.0 / (distances**power)
        pred = np.sum(weights * values[neighbor_idx], axis=1) / np.sum(weights, axis=1)
    else:
        raise ValueError("method must be nearest or idw")
    grid = pred.reshape(grid_size, grid_size).astype(np.float32)
    return mask_field(np.clip(grid, 0.0, 1.0), tissue_mask)


def evaluate_prediction(
    *,
    sample: str,
    condition_name: str,
    target_gene: str,
    method: str,
    field_type: str,
    grid: np.ndarray,
    coords_norm: np.ndarray,
    source_values: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    tissue_mask: np.ndarray,
) -> dict[str, object]:
    pred_train = sample_grid_at_spots(grid, coords_norm[train_idx])
    pred_test = sample_grid_at_spots(grid, coords_norm[test_idx])
    roughness = field_roughness_metrics(grid, tissue_mask)
    tissue_mean = float(np.mean(grid[tissue_mask]))
    background_mean = float(np.mean(grid[~tissue_mask]))
    return {
        "sample": sample,
        "condition": condition_name,
        "target_gene": target_gene,
        "method": method,
        "field_type": field_type,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "train_pearson": pearsonr_safe(pred_train, source_values[train_idx]),
        "train_mse": mse(pred_train, source_values[train_idx]),
        "test_pearson": pearsonr_safe(pred_test, source_values[test_idx]),
        "test_mse": mse(pred_test, source_values[test_idx]),
        "background_to_tissue_ratio": background_mean / (tissue_mean + 1e-8),
        **roughness,
    }


def save_grid(path: Path, grid: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, grid.astype(np.float32))


def run_sample(args: argparse.Namespace, sample_name: str) -> list[dict[str, object]]:
    target_name = f"{args.target_gene}_CNS_Myelin"
    run_label = args.run_label or f"{args.profile}_it{args.iterations}_domain{args.num_domain}"
    sample_dir = PROCESSED_ROOT / sample_name
    preflight_dir = PREFLIGHT_ROOT / sample_name / target_name
    output_dir = OUTPUT_ROOT / target_name / sample_name / f"seed{args.seed}_test{int(args.test_fraction * 100)}" / run_label
    metrics_csv = output_dir / "heldout_metrics.csv"
    if args.skip_existing and metrics_csv.exists():
        with metrics_csv.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    output_dir.mkdir(parents=True, exist_ok=True)
    sample = load_visium_lite(sample_dir)
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    diffusion_grid = np.load(preflight_dir / "diffusion_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    source_values = clip_and_normalize(normalized_gene_vector(sample, args.target_gene), percentile=99.0)
    grid_size = diffusion_grid.shape[0]
    train_idx, test_idx = split_indices(len(source_values), test_fraction=args.test_fraction, seed=args.seed)
    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    train_coords = coords_norm[train_idx]
    train_values = source_values[train_idx]
    train_source_grid = build_source_grid(train_coords, train_values, grid_size=grid_size, sigma=1.5)
    rows = []

    baseline_grids = {
        "nearest": kd_interpolation_grid(
            train_coords,
            train_values,
            grid_size=grid_size,
            tissue_mask=tissue_mask,
            method="nearest",
        ),
        "idw_k8": kd_interpolation_grid(
            train_coords,
            train_values,
            grid_size=grid_size,
            tissue_mask=tissue_mask,
            method="idw",
            k=8,
        ),
        "gaussian_sigma1p5": gaussian_interpolation_grid(
            train_coords,
            train_values,
            grid_size=grid_size,
            sigma=1.5,
            tissue_mask=tissue_mask,
        ),
        "gaussian_sigma3": gaussian_interpolation_grid(
            train_coords,
            train_values,
            grid_size=grid_size,
            sigma=3.0,
            tissue_mask=tissue_mask,
        ),
    }
    for method, grid in baseline_grids.items():
        save_grid(output_dir / "grids" / f"{method}.npy", grid)
        rows.append(
            evaluate_prediction(
                sample=sample_name,
                condition_name=condition(sample_name),
                target_gene=args.target_gene,
                method=method,
                field_type="traditional_baseline",
                grid=grid,
                coords_norm=coords_norm,
                source_values=source_values,
                train_idx=train_idx,
                test_idx=test_idx,
                tissue_mask=tissue_mask,
            )
        )

    base_profile = get_profile(args.profile)
    profile = replace(
        base_profile,
        name=f"{run_label}_heldout",
        iterations=args.iterations,
        num_domain=args.num_domain,
        display_every=max(args.iterations // 2, 1),
    )
    profile_updates = {}
    if args.pde_weight is not None:
        profile_updates["pde_weight"] = args.pde_weight
    if args.data_weight is not None:
        profile_updates["data_weight"] = args.data_weight
    if args.smoothness_weight is not None:
        profile_updates["smoothness_weight"] = args.smoothness_weight
    if args.fourier_sigma is not None:
        profile_updates["fourier_sigma"] = args.fourier_sigma
    if args.fourier_features is not None:
        profile_updates["fourier_features"] = args.fourier_features
    if profile_updates:
        profile = replace(profile, **profile_updates)
    result = solve_scalar_reaction_diffusion(
        train_coords,
        train_values,
        diffusion_grid,
        train_source_grid,
        profile=profile,
        tissue_mask=tissue_mask,
        device=args.device,
        seed=args.seed,
        prediction_grid_size=grid_size,
    )
    raw_norm = robust_normalize(result.grid_prediction, percentile=99.5)
    masked_grid = mask_field(np.clip(raw_norm, 0.0, 1.0), tissue_mask)
    save_grid(output_dir / "grids" / "anisonet_masked.npy", masked_grid)
    rows.append(
        evaluate_prediction(
            sample=sample_name,
            condition_name=condition(sample_name),
            target_gene=args.target_gene,
            method="anisoNET",
            field_type=f"{run_label}_masked",
            grid=masked_grid,
            coords_norm=coords_norm,
            source_values=source_values,
            train_idx=train_idx,
            test_idx=test_idx,
            tissue_mask=tissue_mask,
        )
    )
    if args.postprocess_sigma > 0:
        smoothed = smooth_inside_mask(masked_grid, tissue_mask, sigma=args.postprocess_sigma)
        post_grid = normalize_masked_field(smoothed, tissue_mask, percentile=99.5)
        save_grid(output_dir / "grids" / "anisonet_gauss07.npy", post_grid)
        rows.append(
            evaluate_prediction(
                sample=sample_name,
                condition_name=condition(sample_name),
                target_gene=args.target_gene,
                method="anisoNET",
                field_type=f"{run_label}_gauss07",
                grid=post_grid,
                coords_norm=coords_norm,
                source_values=source_values,
                train_idx=train_idx,
                test_idx=test_idx,
                tissue_mask=tissue_mask,
            )
        )

    with (output_dir / "pinn_history.json").open("w", encoding="utf-8") as handle:
        json.dump(result.history, handle, indent=2)
    write_csv(rows, metrics_csv)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    all_rows = []
    for sample_name in args.samples:
        print(f"Running held-out benchmark: {args.target_gene} {sample_name}", flush=True)
        all_rows.extend(run_sample(args, sample_name))

    target_name = f"{args.target_gene}_CNS_Myelin"
    summary_dir = OUTPUT_ROOT / target_name
    write_csv(all_rows, summary_dir / "heldout_metrics_summary.csv")
    manifest = {
        "target_gene": args.target_gene,
        "samples": args.samples,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "profile": args.profile,
        "run_label": args.run_label,
        "iterations": args.iterations,
        "num_domain": args.num_domain,
        "summary_csv": str(summary_dir / "heldout_metrics_summary.csv"),
    }
    with (summary_dir / "heldout_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
