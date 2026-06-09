"""Benchmark liver zonation predictions against spot annotations.

This workflow is designed for the APAP liver Visium annotations. It keeps the
usual held-out expression metrics, but adds annotation-aware metrics such as
Central-vs-PeriPortal AUC and contrast preservation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "Script"
WORKFLOW_ROOT = SCRIPT_ROOT / "workflows"
for path in (SCRIPT_ROOT, WORKFLOW_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from anisonet.metrics import mse, pearsonr_safe, sample_grid_at_spots
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector
from barrier_aware_spot_interpolation_benchmark import (
    build_resistance_graph,
    edge_score_from_grid,
    euclidean_distance_matrix,
    predict_gaussian,
    predict_idw,
    split_indices,
    split_indices_by_score,
)
from scipy.sparse.csgraph import dijkstra


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run liver annotation-boundary interpolation benchmark.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--target-gene", required=True)
    parser.add_argument("--annotation-csv", required=True)
    parser.add_argument("--annotation-sample", required=True, help="Annotation sample label, e.g. ABU7 or ABU8.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--positive-group", default="PeriPortal")
    parser.add_argument("--negative-group", default="Central")
    parser.add_argument(
        "--split-mode",
        choices=["random", "high_barrier", "barrier_edge", "high_barrier_or_edge"],
        default="high_barrier_or_edge",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--graph-k", type=int, default=12)
    parser.add_argument("--idw-k", type=int, default=8)
    parser.add_argument("--gaussian-k", type=int, default=32)
    parser.add_argument("--gaussian-scale", type=float, default=3.0)
    parser.add_argument("--resistance-beta", type=float, default=4.0)
    return parser.parse_args()


def barcode_core(value: str) -> str:
    return str(value).split("-")[0]


def auc_score(values: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    valid = np.isfinite(values)
    values = values[valid]
    labels = labels[valid]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(values).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def annotation_metrics(
    values: np.ndarray,
    truth: np.ndarray,
    groups: np.ndarray,
    *,
    positive_group: str,
    negative_group: str,
) -> dict[str, object]:
    groups = np.asarray(groups, dtype=object)
    selected = (groups == positive_group) | (groups == negative_group)
    pos = groups[selected] == positive_group
    vals = np.asarray(values, dtype=np.float64)[selected]
    truth_vals = np.asarray(truth, dtype=np.float64)[selected]
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return {
            "annotation_auc": float("nan"),
            "truth_annotation_auc": float("nan"),
            "annotation_contrast": float("nan"),
            "truth_annotation_contrast": float("nan"),
            "contrast_retention": float("nan"),
            "negative_to_positive_ratio": float("nan"),
            "n_annotated_test": int(selected.sum()),
            "n_positive_group": n_pos,
            "n_negative_group": n_neg,
        }
    pred_pos_mean = float(np.mean(vals[pos]))
    pred_neg_mean = float(np.mean(vals[~pos]))
    truth_pos_mean = float(np.mean(truth_vals[pos]))
    truth_neg_mean = float(np.mean(truth_vals[~pos]))
    pred_contrast = pred_pos_mean - pred_neg_mean
    truth_contrast = truth_pos_mean - truth_neg_mean
    return {
        "annotation_auc": auc_score(vals, pos),
        "truth_annotation_auc": auc_score(truth_vals, pos),
        "annotation_contrast": pred_contrast,
        "truth_annotation_contrast": truth_contrast,
        "contrast_retention": float(pred_contrast / (truth_contrast + 1e-8)),
        "negative_to_positive_ratio": float(pred_neg_mean / (pred_pos_mean + 1e-8)),
        "n_annotated_test": int(selected.sum()),
        "n_positive_group": n_pos,
        "n_negative_group": n_neg,
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

    annotations = pd.read_csv(args.annotation_csv)
    annotations = annotations.loc[annotations["sample"].astype(str) == args.annotation_sample].copy()
    annotations["barcode_core"] = annotations["cell"].map(barcode_core)
    annotation_map = annotations.set_index("barcode_core")["zonationGroup"].to_dict()
    groups = np.asarray([annotation_map.get(barcode_core(barcode), "") for barcode in sample.barcodes], dtype=object)

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
    test_groups = groups[test_idx]

    euclidean = euclidean_distance_matrix(test_coords, train_coords)
    graph = build_resistance_graph(coords_norm, resistance_spots, graph_k=args.graph_k, beta=args.resistance_beta)
    resistance_all = dijkstra(graph, directed=False, indices=test_idx)
    resistance_distance = resistance_all[:, train_idx].astype(np.float32)
    unreachable = ~np.isfinite(resistance_distance)
    if np.any(unreachable):
        fallback = euclidean_distance_matrix(test_coords, train_coords)
        resistance_distance[unreachable] = fallback[unreachable] * (1.0 + args.resistance_beta)

    predictions: dict[str, np.ndarray] = {
        "euclidean_idw": predict_idw(euclidean, train_values, k=args.idw_k),
        "resistance_idw": predict_idw(resistance_distance, train_values, k=args.idw_k),
    }
    predictions["euclidean_gaussian"], euclidean_sigma = predict_gaussian(
        euclidean,
        train_values,
        k=args.gaussian_k,
        scale=args.gaussian_scale,
    )
    predictions["resistance_gaussian"], resistance_sigma = predict_gaussian(
        resistance_distance,
        train_values,
        k=args.gaussian_k,
        scale=args.gaussian_scale,
    )

    rows = []
    for method, pred in predictions.items():
        row = {
            "method": method,
            "target_gene": args.target_gene,
            "split_mode": args.split_mode,
            "test_pearson": pearsonr_safe(pred, truth),
            "test_mse": mse(pred, truth),
            "prediction_mean": float(np.mean(pred)),
            "prediction_sd": float(np.std(pred)),
            "positive_group": args.positive_group,
            "negative_group": args.negative_group,
            "sigma": euclidean_sigma if method == "euclidean_gaussian" else resistance_sigma if method == "resistance_gaussian" else "",
        }
        row.update(
            annotation_metrics(
                pred,
                truth,
                test_groups,
                positive_group=args.positive_group,
                negative_group=args.negative_group,
            )
        )
        rows.append(row)

    metrics_csv = output_dir / "liver_annotation_boundary_metrics.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    spot_rows = pd.DataFrame(
        {
            "barcode": np.asarray(sample.barcodes)[test_idx],
            "spot_index": test_idx,
            "zonationGroup": test_groups,
            "truth": truth,
            "barrier": barrier_spots[test_idx],
            "edge": edge_spots[test_idx],
            **{f"pred_{method}": pred for method, pred in predictions.items()},
        }
    )
    spot_csv = output_dir / "liver_annotation_boundary_test_predictions.csv"
    spot_rows.to_csv(spot_csv, index=False)

    manifest = {
        "sample_dir": str(Path(args.sample_dir).resolve()),
        "preflight_dir": str(Path(args.preflight_dir).resolve()),
        "annotation_csv": str(Path(args.annotation_csv).resolve()),
        "annotation_sample": args.annotation_sample,
        "target_gene": args.target_gene,
        "split_mode": args.split_mode,
        "n_train": int(train_idx.size),
        "n_test": int(test_idx.size),
        "n_annotated_test": int(np.sum(test_groups != "")),
        "metrics_csv": str(metrics_csv),
        "spot_csv": str(spot_csv),
    }
    with (output_dir / "liver_annotation_boundary_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"manifest": manifest, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
