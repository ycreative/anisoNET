"""Compare Euclidean and resistance-aware spot interpolation baselines.

This benchmark asks a narrow question: can a barrier/resistance prior make
traditional IDW or Gaussian interpolation stronger for held-out spot prediction?
It evaluates spot-level predictions directly instead of generating a full field.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.metrics import mse, pearsonr_safe, sample_grid_at_spots
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resistance-aware spot interpolation benchmark.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--target-gene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--split-mode",
        choices=["random", "high_barrier", "barrier_edge", "high_barrier_or_edge"],
        default="random",
    )
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--idw-k", type=int, default=8)
    parser.add_argument("--gaussian-k", type=int, default=32)
    parser.add_argument("--gaussian-scale", type=float, default=3.0)
    parser.add_argument("--resistance-beta", type=float, default=4.0)
    return parser.parse_args()


def split_indices(n: int, *, test_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test = max(1, int(round(n * test_fraction)))
    return np.sort(indices[n_test:]), np.sort(indices[:n_test])


def split_indices_by_score(score: np.ndarray, *, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(score, dtype=np.float32).reshape(-1)
    n_test = max(1, int(round(arr.size * test_fraction)))
    order = np.argsort(arr)[::-1]
    test_idx = np.sort(order[:n_test])
    train_idx = np.sort(order[n_test:])
    return train_idx, test_idx


def euclidean_distance_matrix(test_coords: np.ndarray, train_coords: np.ndarray) -> np.ndarray:
    diff = test_coords[:, None, :] - train_coords[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)


def build_resistance_graph(
    coords_norm: np.ndarray,
    resistance_spots: np.ndarray,
    *,
    graph_k: int,
    beta: float,
) -> csr_matrix:
    n = coords_norm.shape[0]
    tree = cKDTree(coords_norm)
    distances, neighbors = tree.query(coords_norm, k=min(graph_k + 1, n))
    rows = []
    cols = []
    data = []
    resistance = clip_and_normalize(resistance_spots, percentile=99.0)
    for i in range(n):
        for distance, j in zip(distances[i, 1:], neighbors[i, 1:]):
            if not np.isfinite(distance):
                continue
            barrier_cost = 1.0 + beta * 0.5 * (resistance[i] + resistance[j])
            weight = float(max(distance, 1e-8) * barrier_cost)
            rows.extend([i, int(j)])
            cols.extend([int(j), i])
            data.extend([weight, weight])
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def edge_score_from_grid(grid: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(grid, dtype=np.float32))
    edge = np.sqrt(gx * gx + gy * gy)
    lo, hi = np.percentile(edge, [1, 99])
    return np.clip((edge - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def predict_idw(distance_matrix: np.ndarray, train_values: np.ndarray, *, k: int, power: float = 2.0) -> np.ndarray:
    k_eff = min(k, train_values.size)
    order = np.argpartition(distance_matrix, kth=k_eff - 1, axis=1)[:, :k_eff]
    selected_dist = np.take_along_axis(distance_matrix, order, axis=1)
    selected_values = train_values[order]
    weights = 1.0 / np.maximum(selected_dist, 1e-8) ** power
    return np.sum(weights * selected_values, axis=1) / np.sum(weights, axis=1)


def predict_gaussian(
    distance_matrix: np.ndarray,
    train_values: np.ndarray,
    *,
    k: int,
    scale: float,
) -> tuple[np.ndarray, float]:
    k_eff = min(k, train_values.size)
    order = np.argpartition(distance_matrix, kth=k_eff - 1, axis=1)[:, :k_eff]
    selected_dist = np.take_along_axis(distance_matrix, order, axis=1)
    selected_values = train_values[order]
    nearest = np.min(distance_matrix, axis=1)
    sigma = float(max(np.median(nearest) * scale, 1e-8))
    weights = np.exp(-0.5 * (selected_dist / sigma) ** 2)
    return np.sum(weights * selected_values, axis=1) / np.sum(weights, axis=1), sigma


def metric_row(method: str, pred: np.ndarray, truth: np.ndarray, *, extra: dict[str, object]) -> dict[str, object]:
    return {
        "method": method,
        "test_pearson": pearsonr_safe(pred, truth),
        "test_mse": mse(pred, truth),
        "prediction_mean": float(np.mean(pred)),
        "prediction_sd": float(np.std(pred)),
        **extra,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_visium_lite(args.sample_dir)
    coords_norm = np.load(Path(args.preflight_dir) / "coords_norm.npy")
    resistance_grid = np.load(Path(args.preflight_dir) / "resistance_grid.npy")
    barrier_grid = np.load(Path(args.preflight_dir) / "barrier_grid.npy")
    source_values = clip_and_normalize(normalized_gene_vector(sample, args.target_gene), percentile=99.0)
    resistance_spots = sample_grid_at_spots(resistance_grid, coords_norm)
    barrier_spots = sample_grid_at_spots(barrier_grid, coords_norm)
    edge_spots = sample_grid_at_spots(edge_score_from_grid(barrier_grid), coords_norm)

    if args.split_mode == "random":
        train_idx, test_idx = split_indices(len(source_values), test_fraction=args.test_fraction, seed=args.seed)
        split_score = np.full_like(source_values, np.nan, dtype=np.float32)
    elif args.split_mode == "high_barrier":
        split_score = barrier_spots
        train_idx, test_idx = split_indices_by_score(split_score, test_fraction=args.test_fraction)
    elif args.split_mode == "barrier_edge":
        split_score = edge_spots
        train_idx, test_idx = split_indices_by_score(split_score, test_fraction=args.test_fraction)
    else:
        split_score = np.maximum(barrier_spots, edge_spots)
        train_idx, test_idx = split_indices_by_score(split_score, test_fraction=args.test_fraction)
    train_coords = coords_norm[train_idx]
    test_coords = coords_norm[test_idx]
    train_values = source_values[train_idx]
    truth = source_values[test_idx]

    euclidean = euclidean_distance_matrix(test_coords, train_coords)
    graph = build_resistance_graph(
        coords_norm,
        resistance_spots,
        graph_k=args.graph_k,
        beta=args.resistance_beta,
    )
    resistance_all = dijkstra(graph, directed=False, indices=test_idx)
    resistance_distance = resistance_all[:, train_idx].astype(np.float32)
    unreachable = ~np.isfinite(resistance_distance)
    if np.any(unreachable):
        fallback = euclidean_distance_matrix(test_coords, train_coords)
        resistance_distance[unreachable] = fallback[unreachable] * (1.0 + args.resistance_beta)

    rows = []
    rows.append(
        metric_row(
            "euclidean_idw",
            predict_idw(euclidean, train_values, k=args.idw_k),
            truth,
            extra={"distance": "euclidean", "k": args.idw_k, "sigma": ""},
        )
    )
    rows.append(
        metric_row(
            "resistance_idw",
            predict_idw(resistance_distance, train_values, k=args.idw_k),
            truth,
            extra={"distance": "resistance_graph", "k": args.idw_k, "sigma": ""},
        )
    )
    pred, sigma = predict_gaussian(euclidean, train_values, k=args.gaussian_k, scale=args.gaussian_scale)
    rows.append(
        metric_row(
            "euclidean_gaussian",
            pred,
            truth,
            extra={"distance": "euclidean", "k": args.gaussian_k, "sigma": sigma},
        )
    )
    pred, sigma = predict_gaussian(resistance_distance, train_values, k=args.gaussian_k, scale=args.gaussian_scale)
    rows.append(
        metric_row(
            "resistance_gaussian",
            pred,
            truth,
            extra={"distance": "resistance_graph", "k": args.gaussian_k, "sigma": sigma},
        )
    )

    csv_path = output_dir / "barrier_aware_spot_interpolation_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "sample_dir": str(Path(args.sample_dir).resolve()),
        "preflight_dir": str(Path(args.preflight_dir).resolve()),
        "target_gene": args.target_gene,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "test_fraction": float(args.test_fraction),
        "seed": int(args.seed),
        "split_mode": args.split_mode,
        "mean_test_barrier": float(np.mean(barrier_spots[test_idx])),
        "mean_train_barrier": float(np.mean(barrier_spots[train_idx])),
        "mean_test_edge": float(np.mean(edge_spots[test_idx])),
        "mean_train_edge": float(np.mean(edge_spots[train_idx])),
        "mean_test_split_score": float(np.nanmean(split_score[test_idx])) if not np.all(np.isnan(split_score)) else None,
        "mean_train_split_score": float(np.nanmean(split_score[train_idx])) if not np.all(np.isnan(split_score)) else None,
        "graph_k": int(args.graph_k),
        "resistance_beta": float(args.resistance_beta),
        "metrics_csv": str(csv_path),
        "rows": rows,
    }
    with (output_dir / "barrier_aware_spot_interpolation_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
