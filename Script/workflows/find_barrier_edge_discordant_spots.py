"""Find spots where traditional smoothing is high but anisoNET is low near barriers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.metrics import sample_grid_at_spots
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find barrier-edge discordant spots.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--anisonet-grid", required=True)
    parser.add_argument("--traditional-grid", required=True)
    parser.add_argument("--target-gene", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--traditional-label", default="Gaussian")
    parser.add_argument("--anisonet-label", default="anisoNET")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--barrier-quantile", type=float, default=0.75)
    parser.add_argument("--edge-quantile", type=float, default=0.75)
    parser.add_argument("--traditional-quantile", type=float, default=0.65)
    parser.add_argument("--delta-quantile", type=float, default=0.75)
    parser.add_argument("--no-normalize-fields", action="store_true", help="Compare raw field values instead of tissue-normalized values.")
    return parser.parse_args()


def robust01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    lo, hi = np.nanpercentile(arr, [1, 99])
    return np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def barrier_edge_grid(barrier_grid: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(barrier_grid, dtype=np.float32))
    return robust01(np.sqrt(gx * gx + gy * gy))


def normalize_grid_in_mask(grid: np.ndarray, tissue_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.float32)
    out = np.zeros_like(arr, dtype=np.float32)
    values = arr[np.asarray(tissue_mask, dtype=bool)]
    if values.size == 0:
        return robust01(arr)
    lo, hi = np.nanpercentile(values, [1, 99])
    out[:] = np.clip((arr - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    out[~np.asarray(tissue_mask, dtype=bool)] = 0.0
    return out


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    output_dir: Path,
    *,
    coords_norm: np.ndarray,
    barrier_grid: np.ndarray,
    edge_grid: np.ndarray,
    traditional_grid: np.ndarray,
    anisonet_grid: np.ndarray,
    delta_grid: np.ndarray,
    top_coords: np.ndarray,
    target_gene: str,
    traditional_label: str,
    anisonet_label: str,
) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("Barrier prior", barrier_grid, "Reds"),
        ("Barrier edge score", edge_grid, "viridis"),
        (traditional_label, traditional_grid, "magma"),
        (anisonet_label, anisonet_grid, "magma"),
        (f"{traditional_label} - {anisonet_label}", delta_grid, "coolwarm"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.0), constrained_layout=True)
    for ax, (title, grid, cmap) in zip(axes, panels):
        image = ax.imshow(grid, origin="lower", cmap=cmap)
        ax.scatter(coords_norm[:, 0] * (grid.shape[1] - 1), coords_norm[:, 1] * (grid.shape[0] - 1), s=1, c="white", alpha=0.12)
        if top_coords.size:
            ax.scatter(
                top_coords[:, 0] * (grid.shape[1] - 1),
                top_coords[:, 1] * (grid.shape[0] - 1),
                s=22,
                facecolors="none",
                edgecolors="cyan",
                linewidths=0.9,
            )
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6, length=2)
    fig.suptitle(f"Barrier-edge discordant spots | {target_gene}", fontsize=9)
    fig.savefig(output_dir / "barrier_edge_discordant_spots.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "barrier_edge_discordant_spots.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_visium_lite(args.sample_dir)
    preflight_dir = Path(args.preflight_dir)
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    anisonet_grid_raw = np.load(args.anisonet_grid)
    traditional_grid_raw = np.load(args.traditional_grid)
    if args.no_normalize_fields:
        anisonet_grid = anisonet_grid_raw
        traditional_grid = traditional_grid_raw
    else:
        anisonet_grid = normalize_grid_in_mask(anisonet_grid_raw, tissue_mask)
        traditional_grid = normalize_grid_in_mask(traditional_grid_raw, tissue_mask)
    source_values = clip_and_normalize(normalized_gene_vector(sample, args.target_gene), percentile=99.0)
    edge_grid = barrier_edge_grid(barrier_grid)

    barrier_spot = sample_grid_at_spots(barrier_grid, coords_norm)
    edge_spot = sample_grid_at_spots(edge_grid, coords_norm)
    anisonet_spot = sample_grid_at_spots(anisonet_grid, coords_norm)
    traditional_spot = sample_grid_at_spots(traditional_grid, coords_norm)
    delta_spot = traditional_spot - anisonet_spot
    barrier_or_edge = np.maximum(robust01(barrier_spot), robust01(edge_spot))

    candidate_mask = (
        (barrier_spot >= np.quantile(barrier_spot, args.barrier_quantile))
        | (edge_spot >= np.quantile(edge_spot, args.edge_quantile))
    )
    candidate_mask &= traditional_spot >= np.quantile(traditional_spot, args.traditional_quantile)
    candidate_mask &= delta_spot > 0
    candidate_mask &= delta_spot >= np.quantile(delta_spot, args.delta_quantile)

    score = robust01(delta_spot) + 0.5 * robust01(traditional_spot) + 0.5 * barrier_or_edge - 0.25 * robust01(anisonet_spot)
    candidate_indices = np.where(candidate_mask)[0]
    if candidate_indices.size == 0:
        positive = np.where(delta_spot > 0)[0]
        candidate_indices = positive[np.argsort(score[positive])[-args.top_n :]] if positive.size else np.argsort(score)[-args.top_n :]
    ranked = candidate_indices[np.argsort(score[candidate_indices])[::-1]]

    rows = []
    grid_size = barrier_grid.shape[0]
    for rank, idx in enumerate(ranked, start=1):
        rows.append(
            {
                "rank": rank,
                "barcode": sample.barcodes[idx],
                "spot_index": int(idx),
                "x_norm": float(coords_norm[idx, 0]),
                "y_norm": float(coords_norm[idx, 1]),
                "grid_x": int(np.clip(coords_norm[idx, 0] * (grid_size - 1), 0, grid_size - 1)),
                "grid_y": int(np.clip(coords_norm[idx, 1] * (grid_size - 1), 0, grid_size - 1)),
                "source_observed": float(source_values[idx]),
                "barrier_spot": float(barrier_spot[idx]),
                "barrier_edge_spot": float(edge_spot[idx]),
                "traditional_prediction": float(traditional_spot[idx]),
                "anisonet_prediction": float(anisonet_spot[idx]),
                "traditional_minus_anisonet": float(delta_spot[idx]),
                "discordance_score": float(score[idx]),
            }
        )

    write_rows(output_dir / "barrier_edge_discordant_spots_all_candidates.csv", rows)
    top_rows = rows[: args.top_n]
    write_rows(output_dir / "barrier_edge_discordant_spots_top.csv", top_rows)

    delta_grid = traditional_grid - anisonet_grid
    top_indices = [int(row["spot_index"]) for row in top_rows]
    top_coords = coords_norm[top_indices] if top_indices else np.empty((0, 2), dtype=np.float32)
    make_figure(
        output_dir,
        coords_norm=coords_norm,
        barrier_grid=barrier_grid,
        edge_grid=edge_grid,
        traditional_grid=traditional_grid,
        anisonet_grid=anisonet_grid,
        delta_grid=np.where(tissue_mask, delta_grid, 0.0),
        top_coords=top_coords,
        target_gene=args.target_gene,
        traditional_label=args.traditional_label,
        anisonet_label=args.anisonet_label,
    )

    summary = {
        "sample_dir": str(Path(args.sample_dir).resolve()),
        "preflight_dir": str(preflight_dir.resolve()),
        "anisonet_grid": str(Path(args.anisonet_grid).resolve()),
        "traditional_grid": str(Path(args.traditional_grid).resolve()),
        "target_gene": args.target_gene,
        "n_spots": int(len(source_values)),
        "n_candidates": int(len(rows)),
        "top_n": int(len(top_rows)),
        "candidate_definition": {
            "barrier_quantile": args.barrier_quantile,
            "edge_quantile": args.edge_quantile,
            "traditional_quantile": args.traditional_quantile,
            "delta_quantile": args.delta_quantile,
            "field_values_normalized_in_tissue": not args.no_normalize_fields,
        },
        "top_mean_traditional_minus_anisonet": float(np.mean([row["traditional_minus_anisonet"] for row in top_rows])),
        "top_mean_barrier": float(np.mean([row["barrier_spot"] for row in top_rows])),
        "top_mean_edge": float(np.mean([row["barrier_edge_spot"] for row in top_rows])),
    }
    with (output_dir / "barrier_edge_discordant_spots_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    interpretation = output_dir / "barrier_edge_discordant_spots_interpretation.md"
    best = top_rows[0]
    interpretation.write_text(
        "\n".join(
            [
                "# Barrier-Edge Discordant Spots Interpretation",
                "",
                f"Target gene: `{args.target_gene}`",
                "",
                "This diagnostic ranks real Visium spots where the traditional field is high, anisoNET is lower, and the spot lies in a high-barrier or barrier-edge region.",
                "",
                "## Top Spot",
                "",
                f"- Barcode: `{best['barcode']}`",
                f"- Grid coordinate: `({best['grid_x']}, {best['grid_y']})`",
                f"- Observed source: `{best['source_observed']:.4f}`",
                f"- Barrier score: `{best['barrier_spot']:.4f}`",
                f"- Barrier-edge score: `{best['barrier_edge_spot']:.4f}`",
                f"- Traditional prediction: `{best['traditional_prediction']:.4f}`",
                f"- anisoNET prediction: `{best['anisonet_prediction']:.4f}`",
                f"- Traditional minus anisoNET: `{best['traditional_minus_anisonet']:.4f}`",
                "",
                "## Interpretation",
                "",
                "These spots are candidate examples of barrier-sensitive suppression: a local smoothing baseline assigns a high value, while anisoNET assigns a lower value in a high-barrier or barrier-edge context.",
                "",
                "Because this is real tissue, these examples are diagnostic rather than definitive false-positive labels. They should be paired with synthetic or independently annotated barrier tests before making strong claims.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
