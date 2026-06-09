"""Post-process anisoNET field grids for evaluation and plotting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process an anisoNET prediction grid.")
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--prediction-grid", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="field_postprocessed.npy")
    parser.add_argument("--mask-tissue", action="store_true")
    parser.add_argument("--smooth-sigma", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preflight_dir = Path(args.preflight_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    field = np.load(args.prediction_grid).astype(np.float32)
    if args.smooth_sigma > 0:
        field = gaussian_filter(field, sigma=args.smooth_sigma).astype(np.float32)
    if args.mask_tissue:
        tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
        field = np.where(tissue_mask, field, 0.0).astype(np.float32)

    output_path = output_dir / args.output_name
    np.save(output_path, field)
    payload = {
        "prediction_grid": str(Path(args.prediction_grid).resolve()),
        "preflight_dir": str(preflight_dir.resolve()),
        "output": str(output_path.resolve()),
        "mask_tissue": bool(args.mask_tissue),
        "smooth_sigma": float(args.smooth_sigma),
    }
    with (output_dir / f"{Path(args.output_name).stem}_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
