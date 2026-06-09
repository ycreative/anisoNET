"""Evaluate an anisoNET field prediction with quantitative metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.metrics import evaluate_field
from anisonet.preprocessing import clip_and_normalize, spot_grid_indices
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an anisoNET prediction grid.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--target-gene", required=True)
    parser.add_argument("--prediction-grid", required=True, help="Path to prediction .npy file.")
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--source-percentile", type=float, default=99.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight_dir = Path(args.preflight_dir)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    sample = load_visium_lite(args.sample_dir)
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    prediction_grid = np.load(args.prediction_grid)

    source_spot_values = clip_and_normalize(
        normalized_gene_vector(sample, args.target_gene),
        percentile=args.source_percentile,
    )

    metrics = evaluate_field(
        coords_norm=coords_norm,
        source_spot_values=source_spot_values,
        prediction_grid=prediction_grid,
        barrier_grid=barrier_grid,
        tissue_mask=tissue_mask,
    ).to_dict()
    payload = {
        "method": args.method_name,
        "target_gene": args.target_gene,
        "sample_dir": str(Path(args.sample_dir).resolve()),
        "preflight_dir": str(preflight_dir.resolve()),
        "prediction_grid": str(Path(args.prediction_grid).resolve()),
        "metrics": metrics,
    }
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


def build_gaussian_baseline_from_source_grid(source_grid: np.ndarray, *, sigma: float = 3.5) -> np.ndarray:
    """Helper used by notebooks or future workflows for a simple smoothing baseline."""

    return gaussian_filter(source_grid, sigma=sigma).astype(np.float32)


def build_spot_raster_grid(coords_norm: np.ndarray, spot_values: np.ndarray, *, grid_size: int = 200) -> np.ndarray:
    """Rasterize spot values onto a grid without smoothing."""

    grid_x, grid_y = spot_grid_indices(coords_norm, grid_size=grid_size)
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    grid[grid_y, grid_x] = np.asarray(spot_values, dtype=np.float32)
    return grid


if __name__ == "__main__":
    main()
