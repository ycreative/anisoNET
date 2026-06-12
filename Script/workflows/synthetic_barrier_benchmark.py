"""Synthetic barrier benchmark using real Visium tissue masks and barriers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.diffusion import build_source_grid
from anisonet.metrics import field_roughness_metrics, mse, pearsonr_safe, sample_grid_at_spots
from anisonet.pinn import get_profile, robust_normalize, solve_scalar_reaction_diffusion
from anisonet.postprocessing import mask_field, normalize_masked_field, smooth_inside_mask
from anisonet.preprocessing import spot_grid_indices


PREFLIGHT_ROOT = PROJECT_ROOT / "codexAnalysis" / "preflight" / "brain_aging_gse193107"
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "synthetic_barrier" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic anisoNET barrier benchmark.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument("--target-name", default="Apoe_CNS_Myelin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--num-domain", type=int, default=1400)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", default="fourier_refined_16g")
    parser.add_argument("--postprocess-sigma", type=float, default=0.7)
    parser.add_argument("--plot-only", action="store_true")
    return parser.parse_args()


def split_indices(n: int, *, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    return np.sort(indices[n_test:]), np.sort(indices[:n_test])


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
    return mask_field(np.clip(numerator / (denominator + 1e-8), 0.0, 1.0), tissue_mask)


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
    return mask_field(np.clip(pred.reshape(grid_size, grid_size).astype(np.float32), 0.0, 1.0), tissue_mask)


def graph_smoothed_spot_values(coords_norm: np.ndarray, values: np.ndarray, *, k: int, iterations: int) -> np.ndarray:
    """Smooth spot values over a spatial kNN graph, mimicking neighborhood-based local blur."""

    tree = cKDTree(coords_norm)
    _, neighbor_idx = tree.query(coords_norm, k=k + 1)
    neighbor_idx = neighbor_idx[:, 1:]
    smoothed = np.asarray(values, dtype=np.float32).copy()
    for _ in range(iterations):
        neighbor_mean = np.mean(smoothed[neighbor_idx], axis=1)
        smoothed = 0.5 * smoothed + 0.5 * neighbor_mean
    return np.clip(smoothed, 0.0, 1.0).astype(np.float32)


def graph_smoothing_grid(
    coords_norm: np.ndarray,
    values: np.ndarray,
    *,
    grid_size: int,
    tissue_mask: np.ndarray,
    k: int,
    iterations: int,
    sigma: float,
) -> np.ndarray:
    smoothed_values = graph_smoothed_spot_values(coords_norm, values, k=k, iterations=iterations)
    return gaussian_interpolation_grid(
        coords_norm,
        smoothed_values,
        grid_size=grid_size,
        sigma=sigma,
        tissue_mask=tissue_mask,
    )


def make_synthetic_truth(
    coords_norm: np.ndarray,
    barrier_grid: np.ndarray,
    tissue_mask: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid_size = barrier_grid.shape[0]
    barrier_spots = sample_grid_at_spots(barrier_grid, coords_norm)
    low_barrier_idx = np.where(barrier_spots <= np.quantile(barrier_spots, 0.35))[0]
    chosen = rng.choice(low_barrier_idx, size=min(5, low_barrier_idx.size), replace=False)
    source_values = rng.uniform(0.7, 1.0, size=chosen.size).astype(np.float32)
    seed_grid = build_source_grid(coords_norm[chosen], source_values, grid_size=grid_size, sigma=1.2)
    diffused = gaussian_filter(seed_grid, sigma=10.0)
    suppressed = diffused * np.exp(-3.5 * barrier_grid)
    truth = mask_field(suppressed, tissue_mask)
    truth = normalize_masked_field(truth, tissue_mask, percentile=99.5)
    return truth.astype(np.float32)


def perturb_diffusion(diffusion_grid: np.ndarray, barrier_grid: np.ndarray, tissue_mask: np.ndarray, *, alpha: float = 4.0) -> dict[str, np.ndarray]:
    optical = np.zeros_like(diffusion_grid, dtype=np.float32)
    optical[tissue_mask] = diffusion_grid[tissue_mask] / np.exp(-alpha * barrier_grid[tissue_mask])
    background = float(np.median(diffusion_grid[~tissue_mask])) if np.any(~tissue_mask) else 0.005
    optical = np.where(tissue_mask, optical, background).astype(np.float32)
    return {
        "anisoNET_original_barrier": diffusion_grid.astype(np.float32),
        "anisoNET_no_transcript_barrier": optical,
    }


def evaluate_grid(
    *,
    method: str,
    grid: np.ndarray,
    truth_grid: np.ndarray,
    coords_norm: np.ndarray,
    observed_values: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    tissue_mask: np.ndarray,
    barrier_grid: np.ndarray,
) -> dict[str, object]:
    pred_train = sample_grid_at_spots(grid, coords_norm[train_idx])
    pred_test = sample_grid_at_spots(grid, coords_norm[test_idx])
    truth_spots = sample_grid_at_spots(truth_grid, coords_norm)
    barrier_spots = sample_grid_at_spots(barrier_grid, coords_norm)
    high = barrier_spots >= np.quantile(barrier_spots, 0.9)
    low = barrier_spots <= np.quantile(barrier_spots, 0.1)
    roughness = field_roughness_metrics(grid, tissue_mask)
    return {
        "method": method,
        "train_pearson_observed": pearsonr_safe(pred_train, observed_values[train_idx]),
        "test_pearson_observed": pearsonr_safe(pred_test, observed_values[test_idx]),
        "test_mse_observed": mse(pred_test, observed_values[test_idx]),
        "grid_pearson_truth": pearsonr_safe(grid[tissue_mask], truth_grid[tissue_mask]),
        "grid_mse_truth": mse(grid[tissue_mask], truth_grid[tissue_mask]),
        "spot_pearson_truth": pearsonr_safe(sample_grid_at_spots(grid, coords_norm), truth_spots),
        "high_barrier_mean": float(np.mean(sample_grid_at_spots(grid, coords_norm)[high])),
        "low_barrier_mean": float(np.mean(sample_grid_at_spots(grid, coords_norm)[low])),
        "high_to_low_barrier_ratio": float(
            np.mean(sample_grid_at_spots(grid, coords_norm)[high])
            / (np.mean(sample_grid_at_spots(grid, coords_norm)[low]) + 1e-8)
        ),
        **roughness,
    }


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def plot_outputs(output_dir: Path, grids: dict[str, np.ndarray], metrics: list[dict[str, object]], tissue_mask: np.ndarray) -> None:
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
    order = [
        "truth",
        "gaussian_sigma1p5",
        "gaussian_sigma3",
        "graph_smooth_k6_iter3",
        "graph_smooth_k12_iter5",
        "anisoNET_original_barrier",
        "anisoNET_no_transcript_barrier",
    ]
    labels = {
        "truth": "Synthetic truth",
        "gaussian_sigma1p5": "Gaussian sigma 1.5",
        "gaussian_sigma3": "Gaussian sigma 3",
        "idw_k8": "IDW k=8",
        "graph_smooth_k6_iter3": "Graph smooth k=6",
        "graph_smooth_k12_iter5": "Graph smooth k=12",
        "anisoNET_original_barrier": "anisoNET original barrier",
        "anisoNET_no_transcript_barrier": "anisoNET no transcript barrier",
    }
    vmax = float(np.percentile(np.concatenate([grids[name][tissue_mask] for name in order]), 99.5))
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.8), constrained_layout=True)
    metric_map = {row["method"]: row for row in metrics}
    flat_axes = axes.reshape(-1)
    for ax, name in zip(flat_axes, order):
        image = ax.imshow(grids[name], origin="lower", cmap="magma", vmin=0, vmax=vmax)
        if name == "truth":
            title = labels[name]
        else:
            row = metric_map[name]
            title = f"{labels[name]}\nr={float(row['grid_pearson_truth']):.3f}, MSE={float(row['grid_mse_truth']):.3f}"
        ax.set_title(title, fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in flat_axes[len(order) :]:
        ax.axis("off")
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012)
    cbar.outline.set_linewidth(0.3)
    cbar.ax.tick_params(labelsize=5, length=1.5, width=0.3)
    fig.savefig(output_dir / "synthetic_barrier_fields.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "synthetic_barrier_fields.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    preflight_dir = PREFLIGHT_ROOT / args.sample / args.target_name
    train_fraction = 1.0 - args.test_fraction
    split_label = f"seed{args.seed}_train{int(round(train_fraction * 100))}_test{int(round(args.test_fraction * 100))}"
    output_dir = OUTPUT_ROOT / args.sample / args.target_name / split_label
    output_dir.mkdir(parents=True, exist_ok=True)

    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    diffusion_grid = np.load(preflight_dir / "diffusion_grid.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    grid_size = diffusion_grid.shape[0]

    if args.plot_only:
        grid_names = [
            "truth",
            "nearest",
            "idw_k8",
            "gaussian_sigma1p5",
            "gaussian_sigma3",
            "graph_smooth_k6_iter3",
            "graph_smooth_k12_iter5",
            "anisoNET_original_barrier",
            "anisoNET_no_transcript_barrier",
        ]
        grids = {name: np.load(output_dir / f"{name}.npy") for name in grid_names}
        metrics = read_csv(output_dir / "synthetic_barrier_metrics.csv")
        plot_outputs(output_dir, grids, metrics, tissue_mask)
        print(f"Replotted synthetic barrier benchmark in {output_dir}", flush=True)
        return

    truth = make_synthetic_truth(coords_norm, barrier_grid, tissue_mask, seed=args.seed)
    rng = np.random.default_rng(args.seed + 100)
    observed = np.clip(sample_grid_at_spots(truth, coords_norm) + rng.normal(0.0, 0.03, size=coords_norm.shape[0]), 0.0, 1.0).astype(np.float32)
    train_idx, test_idx = split_indices(observed.size, test_fraction=args.test_fraction, seed=args.seed)
    train_coords = coords_norm[train_idx]
    train_values = observed[train_idx]
    train_source_grid = build_source_grid(train_coords, train_values, grid_size=grid_size, sigma=1.5)

    grids = {
        "truth": truth,
        "nearest": kd_interpolation_grid(train_coords, train_values, grid_size=grid_size, tissue_mask=tissue_mask, method="nearest"),
        "idw_k8": kd_interpolation_grid(train_coords, train_values, grid_size=grid_size, tissue_mask=tissue_mask, method="idw", k=8),
        "gaussian_sigma1p5": gaussian_interpolation_grid(train_coords, train_values, grid_size=grid_size, sigma=1.5, tissue_mask=tissue_mask),
        "gaussian_sigma3": gaussian_interpolation_grid(train_coords, train_values, grid_size=grid_size, sigma=3.0, tissue_mask=tissue_mask),
        "graph_smooth_k6_iter3": graph_smoothing_grid(
            train_coords,
            train_values,
            grid_size=grid_size,
            tissue_mask=tissue_mask,
            k=6,
            iterations=3,
            sigma=1.5,
        ),
        "graph_smooth_k12_iter5": graph_smoothing_grid(
            train_coords,
            train_values,
            grid_size=grid_size,
            tissue_mask=tissue_mask,
            k=12,
            iterations=5,
            sigma=1.5,
        ),
    }

    base_profile = get_profile(args.profile)
    profile = replace(
        base_profile,
        name=f"{args.profile}_synthetic_barrier",
        iterations=args.iterations,
        num_domain=args.num_domain,
        display_every=max(args.iterations // 2, 1),
    )
    histories = {}
    for method, diffusion_variant in perturb_diffusion(diffusion_grid, barrier_grid, tissue_mask).items():
        print(f"Running synthetic benchmark: {method}", flush=True)
        result = solve_scalar_reaction_diffusion(
            train_coords,
            train_values,
            diffusion_variant,
            train_source_grid,
            profile=profile,
            tissue_mask=tissue_mask,
            device=args.device,
            seed=args.seed,
            prediction_grid_size=grid_size,
        )
        masked = mask_field(np.clip(robust_normalize(result.grid_prediction, percentile=99.5), 0.0, 1.0), tissue_mask)
        if args.postprocess_sigma > 0:
            masked = normalize_masked_field(smooth_inside_mask(masked, tissue_mask, sigma=args.postprocess_sigma), tissue_mask)
        grids[method] = masked.astype(np.float32)
        histories[method] = result.history

    metrics = []
    for method, grid in grids.items():
        np.save(output_dir / f"{method}.npy", grid.astype(np.float32))
        if method == "truth":
            continue
        metrics.append(
            evaluate_grid(
                method=method,
                grid=grid,
                truth_grid=truth,
                coords_norm=coords_norm,
                observed_values=observed,
                train_idx=train_idx,
                test_idx=test_idx,
                tissue_mask=tissue_mask,
                barrier_grid=barrier_grid,
            )
        )
    write_csv(metrics, output_dir / "synthetic_barrier_metrics.csv")
    with (output_dir / "synthetic_barrier_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "sample": args.sample,
                "target_name": args.target_name,
                "seed": args.seed,
                "test_fraction": args.test_fraction,
                "iterations": args.iterations,
                "num_domain": args.num_domain,
                "profile": args.profile,
                "histories": histories,
            },
            handle,
            indent=2,
        )
    plot_outputs(output_dir, grids, metrics, tissue_mask)
    print(f"Wrote synthetic barrier benchmark to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
