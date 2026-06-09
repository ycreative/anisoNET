"""Prototype barrier-aware ligand-receptor scoring from anisoNET fields."""

from __future__ import annotations

import os

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.preprocessing import clip_and_normalize, spot_grid_indices
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


PROCESSED_ROOT = PROJECT_ROOT / "codexAnalysis" / "processed_visium" / "brain_aging_gse193107"
PREFLIGHT_ROOT = PROJECT_ROOT / "codexAnalysis" / "histology_prior" / "brain_aging_gse193107"
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "barrier_aware_lr" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a prototype barrier-aware LR scoring analysis.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument("--ligand", default="Apoe")
    parser.add_argument("--receptor", action="append", default=["Lrp1", "Ldlr", "Trem2", "Sorl1"])
    parser.add_argument("--barrier-name", default="CNS_Myelin")
    parser.add_argument("--histology-prior", default="brightness")
    parser.add_argument("--field-type", default="gauss07", choices=["masked", "gauss07"])
    parser.add_argument("--gaussian-sigma", type=float, default=3.0)
    parser.add_argument("--source-percentile", type=float, default=99.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / args.sample / f"{args.ligand}_{args.barrier_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_dir = PROCESSED_ROOT / args.sample
    preflight_dir = PREFLIGHT_ROOT / args.sample / f"{args.ligand}_{args.barrier_name}" / args.histology_prior
    pinn_dir = PREFLIGHT_ROOT / args.sample / f"{args.ligand}_{args.barrier_name}" / f"{args.histology_prior}_pinn"
    field_path = pinn_dir / (
        "pinn_grid_prediction_postprocessed.npy"
        if args.field_type == "gauss07"
        else "pinn_grid_prediction_clean_tissue_masked.npy"
    )

    sample = load_visium_lite(sample_dir)
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    source_grid = np.load(preflight_dir / "source_grid.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    anisonet_field = np.load(field_path)
    gaussian_field = normalize_grid(gaussian_filter(source_grid, sigma=args.gaussian_sigma) * tissue_mask)

    ligand_spot = clip_and_normalize(
        normalized_gene_vector(sample, args.ligand),
        percentile=args.source_percentile,
    )
    anisonet_spot = sample_grid_at_spots(anisonet_field, coords_norm)
    gaussian_spot = sample_grid_at_spots(gaussian_field, coords_norm)
    barrier_spot = sample_grid_at_spots(barrier_grid, coords_norm)

    rows = []
    spot_rows = []
    for receptor in args.receptor:
        if receptor not in sample.genes:
            rows.append({"receptor": receptor, "present": False})
            continue
        receptor_spot = clip_and_normalize(
            normalized_gene_vector(sample, receptor),
            percentile=args.source_percentile,
        )
        scores = {
            "direct_spot_product": normalize_vector(ligand_spot) * normalize_vector(receptor_spot),
            "gaussian_field_product": normalize_vector(gaussian_spot) * normalize_vector(receptor_spot),
            "anisoNET_field_product": normalize_vector(anisonet_spot) * normalize_vector(receptor_spot),
        }
        rows.append(summarize_receptor(receptor, receptor_spot, barrier_spot, scores))
        for index, barcode in enumerate(sample.barcodes):
            spot_rows.append(
                {
                    "barcode": barcode,
                    "x_norm": float(coords_norm[index, 0]),
                    "y_norm": float(coords_norm[index, 1]),
                    "ligand": args.ligand,
                    "receptor": receptor,
                    "ligand_spot": float(ligand_spot[index]),
                    "receptor_spot": float(receptor_spot[index]),
                    "anisonet_ligand_field": float(anisonet_spot[index]),
                    "gaussian_ligand_field": float(gaussian_spot[index]),
                    "barrier": float(barrier_spot[index]),
                    "direct_spot_product": float(scores["direct_spot_product"][index]),
                    "gaussian_field_product": float(scores["gaussian_field_product"][index]),
                    "anisoNET_field_product": float(scores["anisoNET_field_product"][index]),
                }
            )

    summary = pd.DataFrame(rows)
    spot_scores = pd.DataFrame(spot_rows)
    summary.to_csv(output_dir / "barrier_aware_lr_summary.csv", index=False)
    spot_scores.to_csv(output_dir / "barrier_aware_lr_spot_scores.csv", index=False)
    plot_summary(summary, spot_scores, args, output_dir)
    write_interpretation(summary, args, output_dir)
    print(summary.to_string(index=False))


def sample_grid_at_spots(grid: np.ndarray, coords_norm: np.ndarray) -> np.ndarray:
    grid_x, grid_y = spot_grid_indices(coords_norm, grid_size=grid.shape[0])
    return grid[grid_y, grid_x].astype(np.float32)


def normalize_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    vmax = np.percentile(arr, 99.0)
    vmin = np.min(arr)
    return (np.clip(arr, vmin, vmax) - vmin) / (vmax - vmin + 1e-8)


def normalize_grid(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    tissue_values = arr[arr > 0]
    if tissue_values.size == 0:
        return arr
    vmax = np.percentile(tissue_values, 99.5)
    return np.clip(arr, 0, vmax) / (vmax + 1e-8)


def summarize_receptor(
    receptor: str,
    receptor_spot: np.ndarray,
    barrier_spot: np.ndarray,
    scores: dict[str, np.ndarray],
) -> dict[str, object]:
    high_barrier = barrier_spot >= np.percentile(barrier_spot, 75)
    low_barrier = barrier_spot <= np.percentile(barrier_spot, 25)
    receptor_positive = receptor_spot > 0
    row: dict[str, object] = {
        "receptor": receptor,
        "present": True,
        "receptor_nonzero_fraction": float(np.mean(receptor_positive)),
        "receptor_mean": float(np.mean(receptor_spot)),
        "receptor_p95": float(np.percentile(receptor_spot, 95)),
    }
    for name, score in scores.items():
        high_mean = float(np.mean(score[high_barrier]))
        low_mean = float(np.mean(score[low_barrier]))
        receptor_pos_mean = float(np.mean(score[receptor_positive])) if np.any(receptor_positive) else float("nan")
        corr = spearmanr(score, barrier_spot, nan_policy="omit")
        row[f"{name}_mean"] = float(np.mean(score))
        row[f"{name}_p95"] = float(np.percentile(score, 95))
        row[f"{name}_high_barrier_mean"] = high_mean
        row[f"{name}_low_barrier_mean"] = low_mean
        row[f"{name}_high_to_low_barrier_ratio"] = high_mean / (low_mean + 1e-8)
        row[f"{name}_receptor_positive_mean"] = receptor_pos_mean
        row[f"{name}_spearman_barrier"] = float(corr.statistic)
    row["anisonet_vs_gaussian_score_spearman"] = float(
        spearmanr(scores["anisoNET_field_product"], scores["gaussian_field_product"], nan_policy="omit").statistic
    )
    return row


def plot_summary(summary: pd.DataFrame, spot_scores: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
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
    present = summary[summary["present"] == True].copy()
    receptors = present["receptor"].tolist()
    x = np.arange(len(receptors))
    width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), constrained_layout=True)
    axes[0].bar(
        x - width / 2,
        present["gaussian_field_product_p95"],
        width=width,
        color="#457b9d",
        label="Gaussian",
    )
    axes[0].bar(
        x + width / 2,
        present["anisoNET_field_product_p95"],
        width=width,
        color="#e76f51",
        label="anisoNET",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(receptors, rotation=35, ha="right")
    axes[0].set_ylabel("LR score p95")
    axes[0].set_title(f"{args.ligand} field x receptor", fontsize=7, pad=2)
    axes[0].legend(frameon=False, fontsize=6)

    axes[1].bar(
        x - width / 2,
        present["gaussian_field_product_high_to_low_barrier_ratio"],
        width=width,
        color="#457b9d",
        label="Gaussian",
    )
    axes[1].bar(
        x + width / 2,
        present["anisoNET_field_product_high_to_low_barrier_ratio"],
        width=width,
        color="#e76f51",
        label="anisoNET",
    )
    axes[1].axhline(1.0, color="black", linewidth=0.6, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(receptors, rotation=35, ha="right")
    axes[1].set_ylabel("High/low barrier score ratio")
    axes[1].set_title("Barrier localization", fontsize=7, pad=2)
    for ax in axes:
        ax.tick_params(width=0.3, length=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(output_dir / "barrier_aware_lr_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "barrier_aware_lr_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    plot_spatial_scores(spot_scores, args, output_dir)


def plot_spatial_scores(spot_scores: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    receptors = list(dict.fromkeys(spot_scores["receptor"].tolist()))[:4]
    fig, axes = plt.subplots(1, len(receptors), figsize=(2.2 * len(receptors), 2.4), constrained_layout=True)
    if len(receptors) == 1:
        axes = [axes]
    for ax, receptor in zip(axes, receptors):
        subset = spot_scores[spot_scores["receptor"] == receptor]
        values = subset["anisoNET_field_product"].to_numpy(dtype=float)
        image = ax.scatter(
            subset["x_norm"],
            subset["y_norm"],
            c=values,
            s=5,
            cmap="magma",
            linewidths=0,
            vmin=0,
            vmax=np.percentile(values, 99),
        )
        ax.set_title(f"{args.ligand}-{receptor}", fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig.savefig(output_dir / "barrier_aware_lr_spatial_scores.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "barrier_aware_lr_spatial_scores.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(summary: pd.DataFrame, args: argparse.Namespace, output_dir: Path) -> None:
    present = summary[summary["present"] == True]
    best_p95 = present.sort_values("anisoNET_field_product_p95", ascending=False).iloc[0]
    best_barrier = present.sort_values("anisoNET_field_product_high_to_low_barrier_ratio", ascending=False).iloc[0]
    mean_similarity = present["anisonet_vs_gaussian_score_spearman"].mean()
    lines = [
        "# Barrier-Aware Ligand-Receptor Prototype",
        "",
        f"Prototype scoring for ligand `{args.ligand}` using anisoNET `{args.field_type}` field and receptor expression.",
        "",
        "## Main Result",
        "",
        f"- Highest anisoNET LR score p95 occurred for `{args.ligand}-{best_p95.receptor}` with p95 `{best_p95.anisoNET_field_product_p95:.3f}`.",
        f"- Strongest anisoNET high/low barrier score ratio occurred for `{args.ligand}-{best_barrier.receptor}` with ratio `{best_barrier.anisoNET_field_product_high_to_low_barrier_ratio:.3f}`.",
        f"- Mean Spearman correlation between anisoNET and Gaussian LR scores across receptors was `{mean_similarity:.3f}`.",
        "",
        "## Interpretation",
        "",
        "This is a prototype scoring layer, not a replacement for CellChat, COMMOT, LIANA, SpaTalk, or Squidpy. The useful manuscript angle is that anisoNET can provide a barrier-aware ligand availability field, which can then be multiplied by receptor expression at receiving spots and compared with Euclidean or graph-neighborhood smoothing baselines. In this representative run, anisoNET and Gaussian scores are still highly correlated, so this result should be used as a feasibility prototype rather than a performance claim.",
        "",
    ]
    (output_dir / "barrier_aware_lr_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

