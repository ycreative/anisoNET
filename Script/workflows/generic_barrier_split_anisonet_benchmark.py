"""Generic barrier-relevant held-out benchmark including anisoNET PINN."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.diffusion import build_source_grid
from anisonet.metrics import field_roughness_metrics, mse, pearsonr_safe, sample_grid_at_spots
from anisonet.pinn import get_profile, solve_scalar_reaction_diffusion
from anisonet.postprocessing import mask_field, normalize_masked_field, smooth_inside_mask
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a barrier-relevant held-out benchmark with anisoNET.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--target-gene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-mode",
        choices=["random", "high_barrier", "barrier_edge", "high_barrier_or_edge"],
        default="high_barrier_or_edge",
    )
    parser.add_argument("--profile", default="fourier_refined_16g")
    parser.add_argument("--iterations", type=int, default=700)
    parser.add_argument("--num-domain", type=int, default=1600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--postprocess-sigma", type=float, default=0.7)
    parser.add_argument(
        "--anisonet-normalization",
        choices=["robust", "none"],
        default="robust",
        help="Normalize anisoNET output inside tissue or keep native output scale.",
    )
    parser.add_argument("--data-weight", type=float)
    parser.add_argument("--pde-weight", type=float)
    parser.add_argument("--smoothness-weight", type=float)
    parser.add_argument("--fourier-sigma", type=float)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--idw-k", type=int, default=8)
    parser.add_argument("--gaussian-k", type=int, default=32)
    parser.add_argument("--gaussian-scale", type=float, default=3.0)
    parser.add_argument("--resistance-beta", type=float, default=4.0)
    parser.add_argument(
        "--prior-field",
        choices=["none", "line_resistance_idw"],
        default="none",
        help="Optional continuous prior grid used as a soft PINN constraint.",
    )
    parser.add_argument(
        "--prior-mode",
        choices=["loss", "residual_anchor"],
        default="loss",
        help="Use prior as a soft loss or anchor the model as prior plus learned residual.",
    )
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--prior-line-k", type=int, default=32)
    parser.add_argument("--prior-line-samples", type=int, default=7)
    parser.add_argument("--prior-chunk-size", type=int, default=2048)
    return parser.parse_args()


def split_indices_random(n: int, *, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    return np.sort(indices[n_test:]), np.sort(indices[:n_test])


def split_indices_by_score(score: np.ndarray, *, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(score, dtype=np.float32).reshape(-1)
    n_test = max(1, int(round(arr.size * test_fraction)))
    order = np.argsort(arr)[::-1]
    return np.sort(order[n_test:]), np.sort(order[:n_test])


def robust01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = np.nanpercentile(arr, [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def edge_score_from_grid(grid: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(grid, dtype=np.float32))
    return robust01(np.sqrt(gx * gx + gy * gy))


def euclidean_distance_matrix(test_coords: np.ndarray, train_coords: np.ndarray) -> np.ndarray:
    diff = test_coords[:, None, :] - train_coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)


def build_resistance_graph(coords_norm: np.ndarray, resistance_spots: np.ndarray, *, graph_k: int, beta: float) -> csr_matrix:
    n = coords_norm.shape[0]
    tree = cKDTree(coords_norm)
    distances, neighbors = tree.query(coords_norm, k=min(graph_k + 1, n))
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    resistance = robust01(resistance_spots)
    for i in range(n):
        for distance, j in zip(distances[i, 1:], neighbors[i, 1:]):
            if not np.isfinite(distance):
                continue
            cost = 1.0 + beta * 0.5 * (resistance[i] + resistance[j])
            weight = float(max(distance, 1e-8) * cost)
            rows.extend([i, int(j)])
            cols.extend([int(j), i])
            data.extend([weight, weight])
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def predict_idw(distance_matrix: np.ndarray, train_values: np.ndarray, *, k: int, power: float = 2.0) -> np.ndarray:
    k_eff = min(k, train_values.size)
    order = np.argpartition(distance_matrix, kth=k_eff - 1, axis=1)[:, :k_eff]
    selected_dist = np.take_along_axis(distance_matrix, order, axis=1)
    selected_values = train_values[order]
    weights = 1.0 / np.maximum(selected_dist, 1e-8) ** power
    return np.sum(weights * selected_values, axis=1) / np.sum(weights, axis=1)


def predict_gaussian(distance_matrix: np.ndarray, train_values: np.ndarray, *, k: int, scale: float) -> tuple[np.ndarray, float]:
    k_eff = min(k, train_values.size)
    order = np.argpartition(distance_matrix, kth=k_eff - 1, axis=1)[:, :k_eff]
    selected_dist = np.take_along_axis(distance_matrix, order, axis=1)
    selected_values = train_values[order]
    sigma = float(max(np.median(np.min(distance_matrix, axis=1)) * scale, 1e-8))
    weights = np.exp(-0.5 * (selected_dist / sigma) ** 2)
    return np.sum(weights * selected_values, axis=1) / np.sum(weights, axis=1), sigma


def grid_lookup_np(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    scale = grid.shape[0] - 1
    x_idx = np.clip((points[:, 0] * scale).astype(int), 0, scale)
    y_idx = np.clip((points[:, 1] * scale).astype(int), 0, scale)
    return grid[y_idx, x_idx]


def line_resistance_distance(
    query_points: np.ndarray,
    candidate_points: np.ndarray,
    resistance_grid: np.ndarray,
    *,
    beta: float,
    line_samples: int,
) -> np.ndarray:
    diff = candidate_points - query_points[:, None, :]
    euclidean = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)
    ts = np.linspace(0.0, 1.0, max(line_samples, 2), dtype=np.float32)
    resistance_sum = np.zeros_like(euclidean, dtype=np.float32)
    for t in ts:
        points = query_points[:, None, :] + t * diff
        flat = points.reshape(-1, 2)
        resistance_sum += grid_lookup_np(resistance_grid, flat).reshape(euclidean.shape)
    line_resistance = resistance_sum / float(ts.size)
    return euclidean * (1.0 + beta * line_resistance)


def build_line_resistance_idw_grid(
    *,
    train_coords: np.ndarray,
    train_values: np.ndarray,
    resistance_grid: np.ndarray,
    tissue_mask: np.ndarray,
    k: int,
    beta: float,
    line_samples: int,
    chunk_size: int,
) -> np.ndarray:
    grid_size = resistance_grid.shape[0]
    axis = np.linspace(0.0, 1.0, grid_size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    grid_coords = np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32)
    tissue_flat = np.asarray(tissue_mask, dtype=bool).reshape(-1)
    prior = np.zeros(grid_coords.shape[0], dtype=np.float32)
    tree = cKDTree(train_coords)
    k_eff = min(k, train_values.size)
    tissue_indices = np.where(tissue_flat)[0]
    for start in range(0, tissue_indices.size, chunk_size):
        selected = tissue_indices[start : start + chunk_size]
        query = grid_coords[selected]
        _, neighbor_idx = tree.query(query, k=k_eff)
        if k_eff == 1:
            neighbor_idx = neighbor_idx[:, None]
        candidate_coords = train_coords[neighbor_idx]
        candidate_values = train_values[neighbor_idx]
        distances = line_resistance_distance(
            query,
            candidate_coords,
            resistance_grid,
            beta=beta,
            line_samples=line_samples,
        )
        weights = 1.0 / np.maximum(distances, 1e-8) ** 2
        prior[selected] = np.sum(weights * candidate_values, axis=1) / np.sum(weights, axis=1)
    return prior.reshape(grid_size, grid_size).astype(np.float32)


def metric_row(
    *,
    method: str,
    pred_train: np.ndarray,
    pred_test: np.ndarray,
    train_truth: np.ndarray,
    test_truth: np.ndarray,
    extra: dict[str, object],
) -> dict[str, object]:
    return {
        "method": method,
        "train_pearson": pearsonr_safe(pred_train, train_truth),
        "train_mse": mse(pred_train, train_truth),
        "test_pearson": pearsonr_safe(pred_test, test_truth),
        "test_mse": mse(pred_test, test_truth),
        "prediction_mean_test": float(np.mean(pred_test)),
        "prediction_sd_test": float(np.std(pred_test)),
        **extra,
    }


def linear_calibrate_to_train(
    pred_train: np.ndarray,
    pred_test: np.ndarray,
    train_truth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    x = np.asarray(pred_train, dtype=np.float64).reshape(-1)
    y = np.asarray(train_truth, dtype=np.float64).reshape(-1)
    var = float(np.var(x))
    if var <= 1e-12:
        slope = 0.0
        intercept = float(np.mean(y))
    else:
        slope = float(np.cov(x, y, bias=True)[0, 1] / var)
        intercept = float(np.mean(y) - slope * np.mean(x))
    train_cal = np.clip(slope * np.asarray(pred_train, dtype=np.float64) + intercept, 0.0, 1.0)
    test_cal = np.clip(slope * np.asarray(pred_test, dtype=np.float64) + intercept, 0.0, 1.0)
    return train_cal.astype(np.float32), test_cal.astype(np.float32), slope, intercept


def append_calibrated_metric_row(
    rows: list[dict[str, object]],
    *,
    method: str,
    pred_train: np.ndarray,
    pred_test: np.ndarray,
    train_truth: np.ndarray,
    test_truth: np.ndarray,
    extra: dict[str, object],
) -> None:
    train_cal, test_cal, slope, intercept = linear_calibrate_to_train(pred_train, pred_test, train_truth)
    rows.append(
        metric_row(
            method=f"{method}_traincal",
            pred_train=train_cal,
            pred_test=test_cal,
            train_truth=train_truth,
            test_truth=test_truth,
            extra={
                **extra,
                "calibration": "linear_train_spots",
                "calibration_slope": slope,
                "calibration_intercept": intercept,
            },
        )
    )


def choose_split(
    *,
    split_mode: str,
    source_values: np.ndarray,
    barrier_spots: np.ndarray,
    edge_spots: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_mode == "random":
        train_idx, test_idx = split_indices_random(len(source_values), test_fraction=test_fraction, seed=seed)
        split_score = np.full_like(source_values, np.nan, dtype=np.float32)
    elif split_mode == "high_barrier":
        split_score = barrier_spots
        train_idx, test_idx = split_indices_by_score(split_score, test_fraction=test_fraction)
    elif split_mode == "barrier_edge":
        split_score = edge_spots
        train_idx, test_idx = split_indices_by_score(split_score, test_fraction=test_fraction)
    else:
        split_score = np.maximum(barrier_spots, edge_spots)
        train_idx, test_idx = split_indices_by_score(split_score, test_fraction=test_fraction)
    return train_idx, test_idx, split_score


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_visium_lite(args.sample_dir)
    preflight_dir = Path(args.preflight_dir)
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    diffusion_grid = np.load(preflight_dir / "diffusion_grid.npy")
    resistance_grid = np.load(preflight_dir / "resistance_grid.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    source_values = clip_and_normalize(normalized_gene_vector(sample, args.target_gene), percentile=99.0)
    barrier_spots = sample_grid_at_spots(barrier_grid, coords_norm)
    edge_spots = sample_grid_at_spots(edge_score_from_grid(barrier_grid), coords_norm)
    resistance_spots = sample_grid_at_spots(resistance_grid, coords_norm)

    train_idx, test_idx, split_score = choose_split(
        split_mode=args.split_mode,
        source_values=source_values,
        barrier_spots=barrier_spots,
        edge_spots=edge_spots,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    np.save(output_dir / "train_idx.npy", train_idx)
    np.save(output_dir / "test_idx.npy", test_idx)

    train_coords = coords_norm[train_idx]
    test_coords = coords_norm[test_idx]
    train_values = source_values[train_idx]
    train_truth = source_values[train_idx]
    test_truth = source_values[test_idx]

    euclidean_test = euclidean_distance_matrix(test_coords, train_coords)
    euclidean_train = euclidean_distance_matrix(train_coords, train_coords)
    graph = build_resistance_graph(coords_norm, resistance_spots, graph_k=args.graph_k, beta=args.resistance_beta)
    resistance_all_test = dijkstra(graph, directed=False, indices=test_idx)[:, train_idx].astype(np.float32)
    resistance_all_train = dijkstra(graph, directed=False, indices=train_idx)[:, train_idx].astype(np.float32)
    resistance_all_test[~np.isfinite(resistance_all_test)] = euclidean_test[~np.isfinite(resistance_all_test)] * (1.0 + args.resistance_beta)
    resistance_all_train[~np.isfinite(resistance_all_train)] = euclidean_train[~np.isfinite(resistance_all_train)] * (1.0 + args.resistance_beta)

    rows: list[dict[str, object]] = []
    for method, test_dist, train_dist, distance_label in [
        ("euclidean_idw", euclidean_test, euclidean_train, "euclidean"),
        ("resistance_idw", resistance_all_test, resistance_all_train, "resistance_graph"),
    ]:
        pred_test = predict_idw(test_dist, train_values, k=args.idw_k)
        pred_train = predict_idw(train_dist, train_values, k=args.idw_k)
        rows.append(
            metric_row(
                method=method,
                pred_train=pred_train,
                pred_test=pred_test,
                train_truth=train_truth,
                test_truth=test_truth,
                extra={"field_type": "spot_interpolation", "distance": distance_label, "k": args.idw_k, "sigma": ""},
            )
        )

    for method, test_dist, train_dist, distance_label in [
        ("euclidean_gaussian", euclidean_test, euclidean_train, "euclidean"),
        ("resistance_gaussian", resistance_all_test, resistance_all_train, "resistance_graph"),
    ]:
        pred_test, sigma = predict_gaussian(test_dist, train_values, k=args.gaussian_k, scale=args.gaussian_scale)
        pred_train, _ = predict_gaussian(train_dist, train_values, k=args.gaussian_k, scale=args.gaussian_scale)
        rows.append(
            metric_row(
                method=method,
                pred_train=pred_train,
                pred_test=pred_test,
                train_truth=train_truth,
                test_truth=test_truth,
                extra={"field_type": "spot_interpolation", "distance": distance_label, "k": args.gaussian_k, "sigma": sigma},
            )
        )

    train_source_grid = build_source_grid(train_coords, train_values, grid_size=diffusion_grid.shape[0], sigma=1.5)
    prior_grid = None
    if args.prior_field == "line_resistance_idw":
        prior_grid = build_line_resistance_idw_grid(
            train_coords=train_coords,
            train_values=train_values,
            resistance_grid=resistance_grid,
            tissue_mask=tissue_mask,
            k=args.prior_line_k,
            beta=args.resistance_beta,
            line_samples=args.prior_line_samples,
            chunk_size=args.prior_chunk_size,
        )
        prior_grid = mask_field(prior_grid, tissue_mask)
        np.save(output_dir / "prior_line_resistance_idw_grid.npy", prior_grid.astype(np.float32))
        prior_train = sample_grid_at_spots(prior_grid, train_coords)
        prior_test = sample_grid_at_spots(prior_grid, test_coords)
        prior_extra = {
            "field_type": "continuous_prior",
            "distance": "line_resistance_approx",
            "k": args.prior_line_k,
            "sigma": "",
            **field_roughness_metrics(prior_grid, tissue_mask),
        }
        rows.append(
            metric_row(
                method="prior_line_resistance_idw_grid",
                pred_train=prior_train,
                pred_test=prior_test,
                train_truth=train_truth,
                test_truth=test_truth,
                extra=prior_extra,
            )
        )
        append_calibrated_metric_row(
            rows,
            method="prior_line_resistance_idw_grid",
            pred_train=prior_train,
            pred_test=prior_test,
            train_truth=train_truth,
            test_truth=test_truth,
            extra=prior_extra,
        )
    base_profile = get_profile(args.profile)
    updates = {
        "name": f"{args.profile}_{args.split_mode}_heldout",
        "iterations": args.iterations,
        "num_domain": args.num_domain,
        "display_every": max(args.iterations // 2, 1),
    }
    if args.data_weight is not None:
        updates["data_weight"] = args.data_weight
    if args.pde_weight is not None:
        updates["pde_weight"] = args.pde_weight
    if args.smoothness_weight is not None:
        updates["smoothness_weight"] = args.smoothness_weight
    if args.fourier_sigma is not None:
        updates["fourier_sigma"] = args.fourier_sigma
    profile = replace(base_profile, **updates)
    result = solve_scalar_reaction_diffusion(
        train_coords,
        train_values,
        diffusion_grid,
        train_source_grid,
        profile=profile,
        tissue_mask=tissue_mask,
        prior_grid=prior_grid,
        prior_weight=args.prior_weight if prior_grid is not None else 0.0,
        prior_mode=args.prior_mode if prior_grid is not None else "loss",
        device=args.device,
        seed=args.seed,
        prediction_grid_size=diffusion_grid.shape[0],
    )
    anisonet_masked = mask_field(result.grid_prediction, tissue_mask)
    if args.anisonet_normalization == "robust":
        anisonet_masked = normalize_masked_field(anisonet_masked, tissue_mask)
    else:
        anisonet_masked = np.clip(anisonet_masked, 0.0, 1.0)
    anisonet_gauss = smooth_inside_mask(anisonet_masked, tissue_mask, sigma=args.postprocess_sigma)
    if args.anisonet_normalization == "robust":
        anisonet_gauss = normalize_masked_field(anisonet_gauss, tissue_mask)
    else:
        anisonet_gauss = np.clip(anisonet_gauss, 0.0, 1.0)
    np.save(output_dir / "anisonet_masked_grid.npy", anisonet_masked.astype(np.float32))
    np.save(output_dir / "anisonet_gauss07_grid.npy", anisonet_gauss.astype(np.float32))
    with (output_dir / "anisonet_history.json").open("w", encoding="utf-8") as handle:
        json.dump(result.history, handle, indent=2)

    method_prefix = "anisonet_prior" if prior_grid is not None and args.prior_weight > 0 else "anisonet"
    for method, grid in [(f"{method_prefix}_masked", anisonet_masked), (f"{method_prefix}_gauss07", anisonet_gauss)]:
        pred_train = sample_grid_at_spots(grid, train_coords)
        pred_test = sample_grid_at_spots(grid, test_coords)
        roughness = field_roughness_metrics(grid, tissue_mask)
        anisonet_extra = {
            "field_type": "anisonet",
            "distance": "",
            "k": "",
            "sigma": args.postprocess_sigma if "gauss" in method else "",
            "prior_field": args.prior_field,
            "prior_mode": args.prior_mode if prior_grid is not None else "loss",
            "prior_weight": args.prior_weight if prior_grid is not None else 0.0,
            "anisonet_normalization": args.anisonet_normalization,
            **roughness,
        }
        rows.append(
            metric_row(
                method=method,
                pred_train=pred_train,
                pred_test=pred_test,
                train_truth=train_truth,
                test_truth=test_truth,
                extra=anisonet_extra,
            )
        )
        append_calibrated_metric_row(
            rows,
            method=method,
            pred_train=pred_train,
            pred_test=pred_test,
            train_truth=train_truth,
            test_truth=test_truth,
            extra=anisonet_extra,
        )

    csv_path = output_dir / "generic_barrier_split_metrics.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_dir": str(Path(args.sample_dir).resolve()),
        "preflight_dir": str(preflight_dir.resolve()),
        "target_gene": args.target_gene,
        "split_mode": args.split_mode,
        "test_fraction": args.test_fraction,
        "seed": args.seed,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "mean_test_barrier": float(np.mean(barrier_spots[test_idx])),
        "mean_train_barrier": float(np.mean(barrier_spots[train_idx])),
        "mean_test_edge": float(np.mean(edge_spots[test_idx])),
        "mean_train_edge": float(np.mean(edge_spots[train_idx])),
        "mean_test_split_score": float(np.nanmean(split_score[test_idx])) if not np.all(np.isnan(split_score)) else None,
        "mean_train_split_score": float(np.nanmean(split_score[train_idx])) if not np.all(np.isnan(split_score)) else None,
        "profile": args.profile,
        "effective_profile": profile.__dict__,
        "iterations": args.iterations,
        "num_domain": args.num_domain,
        "prior_field": args.prior_field,
        "prior_mode": args.prior_mode if prior_grid is not None else "loss",
        "prior_weight": args.prior_weight if prior_grid is not None else 0.0,
        "anisonet_normalization": args.anisonet_normalization,
        "metrics_csv": str(csv_path),
        "rows": rows,
    }
    with (output_dir / "generic_barrier_split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
