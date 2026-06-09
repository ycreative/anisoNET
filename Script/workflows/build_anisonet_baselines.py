"""Build simple baseline fields from an anisoNET preflight directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build baseline fields for anisoNET benchmarking.")
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gaussian-sigma", type=float, default=3.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight_dir = Path(args.preflight_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_grid = np.load(preflight_dir / "source_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")

    raw_source = source_grid.astype(np.float32)
    gaussian = gaussian_filter(source_grid, sigma=args.gaussian_sigma).astype(np.float32)
    gaussian_masked = np.where(tissue_mask, gaussian, 0.0).astype(np.float32)

    np.save(output_dir / "baseline_raw_source_grid.npy", raw_source)
    np.save(output_dir / "baseline_gaussian_grid.npy", gaussian)
    np.save(output_dir / "baseline_gaussian_masked_grid.npy", gaussian_masked)

    payload = {
        "preflight_dir": str(preflight_dir.resolve()),
        "gaussian_sigma": float(args.gaussian_sigma),
        "outputs": [
            "baseline_raw_source_grid.npy",
            "baseline_gaussian_grid.npy",
            "baseline_gaussian_masked_grid.npy",
        ],
    }
    with (output_dir / "baseline_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
