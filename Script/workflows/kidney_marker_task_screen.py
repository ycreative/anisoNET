"""Systematically screen kidney marker genes and source/barrier tasks.

The screen is intentionally lightweight and annotation-free. It uses known
kidney compartment markers to build proxy compartment scores, then ranks
candidate source/barrier tasks by expression coverage, source-barrier
separation, top-spot overlap, and spatial adjacency.
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalize_pixel_coordinates, normalized_gene_vector


KIDNEY_COMPARTMENTS: dict[str, tuple[str, ...]] = {
    "proximal_tubule": ("Slc34a1", "Lrp2", "Slc5a2", "Kap", "Aqp1"),
    "tal": ("Umod", "Slc12a1", "Kcnj1", "Bsnd"),
    "dct_cnt": ("Slc12a3", "Pvalb", "Calb1"),
    "collecting_duct": ("Aqp2", "Aqp3", "Scnn1a", "Atp6v1b1", "Atp6v0d2", "Foxi1"),
    "glomerulus": ("Nphs1", "Nphs2", "Podxl"),
    "vascular": ("Pecam1", "Kdr", "Flt1"),
    "interstitial_stromal": ("Col1a1", "Col3a1", "Pdgfrb", "Acta2", "Myl9"),
    "epithelial_general": ("Cdh16", "Krt8", "Krt18"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen kidney marker genes and source/barrier tasks.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument("--min-source-p95", type=float, default=0.7)
    parser.add_argument("--min-barrier-p95", type=float, default=0.7)
    return parser.parse_args()


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


def top_mask(values: np.ndarray, fraction: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    n_top = max(1, int(round(values.size * fraction)))
    order = np.argsort(values)[::-1]
    mask = np.zeros(values.size, dtype=bool)
    mask[order[:n_top]] = True
    return mask


def nearest_distance_between_masks(coords: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    if not np.any(a) or not np.any(b):
        return float("nan")
    tree = cKDTree(coords[b])
    distances, _ = tree.query(coords[a], k=1)
    return float(np.median(distances))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_visium_lite(args.sample_dir)
    coords = normalize_pixel_coordinates(sample, flip_y=True)
    genes_present = set(sample.genes)

    gene_values: dict[str, np.ndarray] = {}
    gene_stats = []
    for compartment, genes in KIDNEY_COMPARTMENTS.items():
        for gene in genes:
            if gene in gene_values:
                continue
            if gene not in genes_present:
                gene_stats.append(
                    {
                        "gene": gene,
                        "compartment": compartment,
                        "present": False,
                        "nonzero_fraction": "",
                        "mean_lognorm": "",
                        "p95_lognorm": "",
                        "max_lognorm": "",
                    }
                )
                continue
            values = normalized_gene_vector(sample, gene)
            gene_values[gene] = values
            gene_stats.append(
                {
                    "gene": gene,
                    "compartment": compartment,
                    "present": True,
                    "nonzero_fraction": float(np.mean(values > 0)),
                    "mean_lognorm": float(np.mean(values)),
                    "p95_lognorm": float(np.percentile(values, 95)),
                    "max_lognorm": float(np.max(values)),
                }
            )

    group_scores: dict[str, np.ndarray] = {}
    group_rows = []
    for compartment, genes in KIDNEY_COMPARTMENTS.items():
        active = tuple(g for g in genes if g in gene_values)
        if not active:
            continue
        stacked = np.vstack([clip_and_normalize(gene_values[g], percentile=99.0) for g in active])
        score = np.mean(stacked, axis=0)
        group_scores[compartment] = score
        group_rows.append(
            {
                "compartment": compartment,
                "active_genes": ";".join(active),
                "n_active_genes": len(active),
                "score_mean": float(np.mean(score)),
                "score_p95": float(np.percentile(score, 95)),
                "top20_median_nearest_neighbor_distance": nearest_distance_between_masks(
                    coords,
                    top_mask(score, args.top_fraction),
                    top_mask(score, args.top_fraction),
                ),
            }
        )

    task_rows = []
    for source_compartment, barrier_compartment in product(group_scores, group_scores):
        if source_compartment == barrier_compartment:
            continue
        source_score = group_scores[source_compartment]
        barrier_score = group_scores[barrier_compartment]
        source_top = top_mask(source_score, args.top_fraction)
        barrier_top = top_mask(barrier_score, args.top_fraction)
        overlap = float(np.mean(source_top & barrier_top) / (np.mean(source_top) + 1e-8))
        adjacency = nearest_distance_between_masks(coords, source_top, barrier_top)
        score_corr = float(np.corrcoef(source_score, barrier_score)[0, 1])
        barrier_auc_vs_source_top = auc_score(barrier_score, barrier_top)

        for gene in KIDNEY_COMPARTMENTS[source_compartment]:
            if gene not in gene_values:
                continue
            values = gene_values[gene]
            gene_p95 = float(np.percentile(values, 95))
            source_auc = auc_score(values, source_top)
            barrier_leakage_ratio = float(np.mean(values[barrier_top]) / (np.mean(values[source_top]) + 1e-8))
            source_strength = min(gene_p95 / 3.0, 1.0)
            separation = max(0.0, 1.0 - overlap) * max(0.0, 1.0 - barrier_leakage_ratio)
            adjacency_score = float(np.exp(-adjacency / 0.08)) if np.isfinite(adjacency) else 0.0
            task_score = 0.35 * source_strength + 0.35 * separation + 0.20 * adjacency_score + 0.10 * max(source_auc, 0.0)
            task_rows.append(
                {
                    "target_gene": gene,
                    "source_compartment": source_compartment,
                    "barrier_compartment": barrier_compartment,
                    "barrier_genes": ";".join(g for g in KIDNEY_COMPARTMENTS[barrier_compartment] if g in gene_values),
                    "target_p95": gene_p95,
                    "source_auc_vs_source_top": source_auc,
                    "source_barrier_score_corr": score_corr,
                    "top20_overlap_fraction": overlap,
                    "source_to_barrier_top_median_distance": adjacency,
                    "barrier_leakage_ratio_in_target_expression": barrier_leakage_ratio,
                    "barrier_score_auc_vs_barrier_top": barrier_auc_vs_source_top,
                    "task_score": task_score,
                    "recommended": bool(gene_p95 >= args.min_source_p95 and overlap < 0.45 and barrier_leakage_ratio < 0.85),
                }
            )

    gene_csv = output_dir / "kidney_marker_screen_gene_stats.csv"
    group_csv = output_dir / "kidney_marker_screen_compartment_scores.csv"
    task_csv = output_dir / "kidney_marker_task_screen.csv"
    pd.DataFrame(gene_stats).to_csv(gene_csv, index=False)
    pd.DataFrame(group_rows).to_csv(group_csv, index=False)
    tasks = pd.DataFrame(task_rows).sort_values(["recommended", "task_score"], ascending=[False, False])
    tasks.to_csv(task_csv, index=False)

    top_csv = output_dir / "kidney_marker_task_screen_top_recommendations.csv"
    tasks.loc[tasks["recommended"]].head(30).to_csv(top_csv, index=False)
    write_interpretation(output_dir, tasks)
    print(f"Wrote {gene_csv}")
    print(f"Wrote {group_csv}")
    print(f"Wrote {task_csv}")
    print(f"Wrote {top_csv}")


def write_interpretation(output_dir: Path, tasks: pd.DataFrame) -> None:
    top = tasks.loc[tasks["recommended"]].head(10)
    lines = [
        "# Kidney Marker Task Screen Interpretation",
        "",
        "This is an annotation-free screening pass using known kidney compartment marker panels.",
        "It ranks source/barrier tasks by marker coverage, source-barrier separation, top-spot overlap, and adjacency.",
        "",
        "Important limitation: compartment labels are marker-derived proxies, so this is a task-prioritization screen rather than independent validation.",
        "",
        "## Top Recommended Tasks",
        "",
        "| Target | Source compartment | Barrier compartment | Barrier genes | Task score | Leakage ratio | Top overlap | Median distance |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in top.iterrows():
        lines.append(
            "| {target_gene} | {source_compartment} | {barrier_compartment} | {barrier_genes} | {task_score:.3f} | {barrier_leakage_ratio_in_target_expression:.3f} | {top20_overlap_fraction:.3f} | {source_to_barrier_top_median_distance:.3f} |".format(
                **row
            )
        )
    lines += [
        "",
        "## Suggested Use",
        "",
        "- Use the top-ranked tasks to choose kidney preflight and high-barrier/edge benchmarks.",
        "- Prefer tasks with low source-barrier overlap, low target leakage into barrier-top spots, and short source-to-barrier distance.",
        "- Do not treat this as independent anatomical annotation; it is a systematic replacement for manual marker picking.",
    ]
    (output_dir / "kidney_marker_task_screen_interpretation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
