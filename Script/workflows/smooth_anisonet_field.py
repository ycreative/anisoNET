"""Apply tissue-aware smoothing to an anisoNET field prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from anisonet.postprocessing import normalize_masked_field, smooth_inside_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smooth an anisoNET field inside the tissue mask.")
    parser.add_argument("--prediction-grid", required=True)
    parser.add_argument("--tissue-mask", required=True)
    parser.add_argument("--output-grid", required=True)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--renormalize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    field = np.load(args.prediction_grid)
    tissue_mask = np.load(args.tissue_mask)
    smoothed = smooth_inside_mask(field, tissue_mask, sigma=args.sigma)
    if args.renormalize:
        smoothed = normalize_masked_field(smoothed, tissue_mask, percentile=99.5)
    output_grid = Path(args.output_grid)
    output_grid.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_grid, smoothed)
    print(f"Wrote smoothed field to {output_grid}")


if __name__ == "__main__":
    main()
