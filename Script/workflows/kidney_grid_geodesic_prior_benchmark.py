"""Benchmark tissue-grid geodesic IDW priors on mouse kidney."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.metrics import field_roughness_metrics, mse, pearsonr_safe, sample_grid_at_spots
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


TARGETS = {
    "Umod": "Umod_proximal_barrier",
    "Slc34a1": "Slc34a1_TAL_CD_barrier",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tissue-grid geodesic prior benchmark.")
    parser.add_argument(
        "--sample-dir",
        default="codexAnalysis/processed_visium/mouse_kidney_10x/V1_Mouse_Kidney",
    )
    parser.add_argument(
        "--preflight-root",
        default="codexAnalysis/cross_tissue/mouse_kidney_10x/V1_Mouse_Kidney/preflight",
    )
    parser.add_argument(
        "--output-dir",
        default="codexAnalysis/barrier_split_anisonet/mouse_kidney_10x/V1_Mouse_Kidney/grid_geodesic_prior",
    )
    parser.add_argument("--targets", default="Umod,Slc34a1")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph-connectivity", choices=["4", "8"], default="8")
    parser.add_argument("--resistance-beta", type=float, default=4.0)
    parser.add_argument("--idw-k", type=int, default=32)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--max-sources", type=int, default=0, help="Optional cap for debugging; 0 uses all train source pixels.")
    return parser.parse_args()


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def robust01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = np.nanpercentile(arr, [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def edge_score_from_grid(grid: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(grid, dtype=np.float32))
    return robust01(np.sqrt(gx * gx + gy * gy))


def split_indices_by_score(score: np.ndarray, *, test_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(score, dtype=np.float32).reshape(-1)
    n_test = max(1, int(round(arr.size * test_fraction)))
    order = np.argsort(arr)[::-1]
    return np.sort(order[n_test:]), np.sort(order[:n_test])


def spot_grid_indices(coords_norm: np.ndarray, grid_size: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.clip((coords_norm[:, 0] * (grid_size - 1)).astype(int), 0, grid_size - 1)
    y = np.clip((coords_norm[:, 1] * (grid_size - 1)).astype(int), 0, grid_size - 1)
    return y, x


def build_tissue_grid_graph(tissue_mask: np.ndarray, resistance_grid: np.ndarray, *, beta: float, connectivity: str) -> tuple[csr_matrix, np.ndarray]:
    tissue = np.asarray(tissue_mask, dtype=bool)
    resistance = robust01(resistance_grid)
    grid_size = tissue.shape[0]
    node_ids = -np.ones(tissue.shape, dtype=np.int32)
    y_idx, x_idx = np.where(tissue)
    node_ids[y_idx, x_idx] = np.arange(y_idx.size, dtype=np.int32)

    if connectivity == "8":
        offsets = [(0, 1, 1.0), (1, 0, 1.0), (1, 1, np.sqrt(2.0)), (1, -1, np.sqrt(2.0))]
    else:
        offsets = [(0, 1, 1.0), (1, 0, 1.0)]

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    scale = 1.0 / max(grid_size - 1, 1)
    for y, x in zip(y_idx, x_idx):
        source = int(node_ids[y, x])
        for dy, dx, base_dist in offsets:
            yy = y + dy
            xx = x + dx
            if yy < 0 or yy >= grid_size or xx < 0 or xx >= grid_size or not tissue[yy, xx]:
                continue
            target = int(node_ids[yy, xx])
            cost = base_dist * scale * (1.0 + beta * 0.5 * (resistance[y, x] + resistance[yy, xx]))
            rows.extend([source, target])
            cols.extend([target, source])
            data.extend([float(cost), float(cost)])
    graph = csr_matrix((data, (rows, cols)), shape=(y_idx.size, y_idx.size))
    return graph, node_ids


def aggregate_source_pixels(
    *,
    train_idx: np.ndarray,
    coords_norm: np.ndarray,
    source_values: np.ndarray,
    node_ids: np.ndarray,
    max_sources: int,
) -> tuple[np.ndarray, np.ndarray]:
    grid_size = node_ids.shape[0]
    y, x = spot_grid_indices(coords_norm[train_idx], grid_size)
    nodes = node_ids[y, x]
    valid = nodes >= 0
    nodes = nodes[valid]
    values = source_values[train_idx][valid]
    accum: dict[int, list[float]] = {}
    for node, value in zip(nodes, values):
        accum.setdefault(int(node), []).append(float(value))
    source_nodes = np.array(sorted(accum), dtype=np.int32)
    source_vals = np.array([np.mean(accum[int(node)]) for node in source_nodes], dtype=np.float32)
    if max_sources > 0 and source_nodes.size > max_sources:
        order = np.argsort(source_vals)[::-1][:max_sources]
        source_nodes = source_nodes[order]
        source_vals = source_vals[order]
    return source_nodes, source_vals


def predict_from_distances(distance_matrix: np.ndarray, source_values: np.ndarray, *, k: int, power: float) -> np.ndarray:
    k_eff = min(k, source_values.size)
    dist = np.asarray(distance_matrix, dtype=np.float32)
    dist[~np.isfinite(dist)] = np.nan
    all_nan = np.all(np.isnan(dist), axis=0)
    if np.any(all_nan):
        dist[:, all_nan] = np.inf
    fill = np.nanmax(np.where(np.isfinite(dist), dist, np.nan))
    if not np.isfinite(fill):
        fill = 1.0
    dist = np.where(np.isfinite(dist), dist, fill * 10.0)
    order = np.argpartition(dist, kth=k_eff - 1, axis=0)[:k_eff, :]
    selected_dist = np.take_along_axis(dist, order, axis=0)
    selected_values = source_values[order]
    weights = 1.0 / np.maximum(selected_dist, 1e-8) ** power
    denom = np.sum(weights, axis=0)
    pred = np.sum(weights * selected_values, axis=0) / np.maximum(denom, 1e-8)
    pred[all_nan] = float(np.mean(source_values))
    return pred.astype(np.float32)


def line_distance_matrix(query_coords: np.ndarray, train_coords: np.ndarray, resistance_grid: np.ndarray, *, beta: float, line_samples: int = 7) -> np.ndarray:
    diff = train_coords[None, :, :] - query_coords[:, None, :]
    euclidean = np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)
    ts = np.linspace(0.0, 1.0, line_samples, dtype=np.float32)
    resistance_sum = np.zeros_like(euclidean, dtype=np.float32)
    grid_size = resistance_grid.shape[0]
    for t in ts:
        points = query_coords[:, None, :] + t * diff
        x = np.clip((points[..., 0] * (grid_size - 1)).astype(int), 0, grid_size - 1)
        y = np.clip((points[..., 1] * (grid_size - 1)).astype(int), 0, grid_size - 1)
        resistance_sum += resistance_grid[y, x]
    return euclidean * (1.0 + beta * resistance_sum / float(ts.size))


def idw_from_distance_rows(distance_matrix: np.ndarray, train_values: np.ndarray, *, k: int, power: float) -> np.ndarray:
    k_eff = min(k, train_values.size)
    order = np.argpartition(distance_matrix, kth=k_eff - 1, axis=1)[:, :k_eff]
    selected_dist = np.take_along_axis(distance_matrix, order, axis=1)
    selected_values = train_values[order]
    weights = 1.0 / np.maximum(selected_dist, 1e-8) ** power
    return (np.sum(weights * selected_values, axis=1) / np.sum(weights, axis=1)).astype(np.float32)


def metric_row(method: str, pred: np.ndarray, truth: np.ndarray, *, extra: dict[str, object]) -> dict[str, object]:
    return {
        "method": method,
        "test_pearson": pearsonr_safe(pred, truth),
        "test_mse": mse(pred, truth),
        "prediction_mean": float(np.mean(pred)),
        "prediction_sd": float(np.std(pred)),
        **extra,
    }


def run_target(args: argparse.Namespace, sample, target: str) -> dict[str, object]:
    output_dir = Path(args.output_dir) / target
    output_dir.mkdir(parents=True, exist_ok=True)
    preflight_dir = Path(args.preflight_root) / TARGETS[target]
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    resistance_grid = np.load(preflight_dir / "resistance_grid.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    source_values = clip_and_normalize(normalized_gene_vector(sample, target), percentile=99.0)
    barrier_spots = sample_grid_at_spots(barrier_grid, coords_norm)
    edge_spots = sample_grid_at_spots(edge_score_from_grid(barrier_grid), coords_norm)
    split_score = np.maximum(barrier_spots, edge_spots)
    train_idx, test_idx = split_indices_by_score(split_score, test_fraction=args.test_fraction)
    train_coords = coords_norm[train_idx]
    test_coords = coords_norm[test_idx]
    train_values = source_values[train_idx]
    truth = source_values[test_idx]

    rows = []
    line_dist = line_distance_matrix(
        test_coords,
        train_coords,
        robust01(resistance_grid),
        beta=args.resistance_beta,
    )
    rows.append(
        metric_row(
            "line_resistance_idw_spot",
            idw_from_distance_rows(line_dist, train_values, k=args.idw_k, power=args.idw_power),
            truth,
            extra={"field_type": "spot_prior", "distance": "line_resistance", "k": args.idw_k},
        )
    )

    graph, node_ids = build_tissue_grid_graph(
        tissue_mask,
        resistance_grid,
        beta=args.resistance_beta,
        connectivity=args.graph_connectivity,
    )
    source_nodes, source_node_values = aggregate_source_pixels(
        train_idx=train_idx,
        coords_norm=coords_norm,
        source_values=source_values,
        node_ids=node_ids,
        max_sources=args.max_sources,
    )
    distances = dijkstra(graph, directed=False, indices=source_nodes)
    grid_pred_flat = predict_from_distances(distances, source_node_values, k=args.idw_k, power=args.idw_power)

    grid = np.zeros_like(tissue_mask, dtype=np.float32)
    tissue_y, tissue_x = np.where(node_ids >= 0)
    node_order = node_ids[tissue_y, tissue_x]
    grid[tissue_y, tissue_x] = grid_pred_flat[node_order]
    np.save(output_dir / "grid_geodesic_idw_prior.npy", grid.astype(np.float32))

    test_y, test_x = spot_grid_indices(test_coords, tissue_mask.shape[0])
    test_nodes = node_ids[test_y, test_x]
    pred_test = np.zeros(test_idx.size, dtype=np.float32)
    valid = test_nodes >= 0
    pred_test[valid] = grid_pred_flat[test_nodes[valid]]
    if np.any(~valid):
        pred_test[~valid] = sample_grid_at_spots(grid, test_coords[~valid])
    rows.append(
        metric_row(
            "grid_geodesic_idw_prior",
            pred_test,
            truth,
            extra={
                "field_type": "continuous_prior",
                "distance": "grid_geodesic",
                "k": args.idw_k,
                "source_nodes": int(source_nodes.size),
                **field_roughness_metrics(grid, tissue_mask),
            },
        )
    )

    csv_path = output_dir / "grid_geodesic_prior_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "target_gene": target,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "source_nodes": int(source_nodes.size),
        "graph_nodes": int(graph.shape[0]),
        "graph_edges": int(graph.nnz),
        "metrics_csv": str(csv_path),
        "rows": rows,
    }
    with (output_dir / "grid_geodesic_prior_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample = load_visium_lite(args.sample_dir)
    summaries = [run_target(args, sample, target) for target in parse_csv(args.targets)]

    combined_rows = []
    for summary in summaries:
        for row in summary["rows"]:
            combined_rows.append({"target_gene": summary["target_gene"], **row})
    combined_csv = output_dir / "grid_geodesic_prior_combined_metrics.csv"
    with combined_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in combined_rows for key in row}))
        writer.writeheader()
        writer.writerows(combined_rows)

    import pandas as pd
    import matplotlib.pyplot as plt

    frame = pd.DataFrame(combined_rows)
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0], "test_pearson", "Held-out Pearson"),
        (axes[1], "test_mse", "Held-out MSE"),
    ]:
        pivot = frame.pivot_table(index="target_gene", columns="method", values=metric, aggfunc="first")
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", labelrotation=0)
        ax.legend(fontsize=6, frameon=False)
    fig.savefig(output_dir / "grid_geodesic_prior_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "grid_geodesic_prior_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Kidney Grid-Geodesic Prior Benchmark",
        "",
        "This benchmark compares the existing line-resistance IDW prior with a tissue-grid geodesic IDW continuous prior.",
        "",
    ]
    for summary in summaries:
        lines.append(f"## {summary['target_gene']}")
        lines.append("")
        lines.append(f"- Graph nodes: `{summary['graph_nodes']}`; source nodes: `{summary['source_nodes']}`.")
        for row in summary["rows"]:
            lines.append(f"- `{row['method']}`: Pearson `{row['test_pearson']:.4f}`, MSE `{row['test_mse']:.5f}`.")
        lines.append("")
    (output_dir / "grid_geodesic_prior_interpretation.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote grid-geodesic prior benchmark to {output_dir}")


if __name__ == "__main__":
    main()
