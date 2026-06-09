"""Re-evaluate existing anisoNET sweep runs after metric updates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from anisonet.metrics import evaluate_field
from anisonet.preprocessing import clip_and_normalize
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refresh metrics for existing anisoNET sweep predictions.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--target-gene", required=True)
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--prediction-name", default="pinn_grid_prediction_clean_tissue_masked.npy")
    parser.add_argument("--source-percentile", type=float, default=99.0)
    return parser.parse_args()


def profile_metadata(profile_path: Path) -> dict[str, float | str]:
    if not profile_path.exists():
        return {}
    with profile_path.open("r", encoding="utf-8") as handle:
        profile = json.load(handle)
    return {
        "run": profile.get("name", profile_path.parent.name),
        "sigma": profile.get("fourier_sigma", ""),
        "smoothness_weight": profile.get("smoothness_weight", ""),
        "pde_weight": profile.get("pde_weight", ""),
        "data_weight": profile.get("data_weight", ""),
        "fourier_features": profile.get("fourier_features", ""),
        "iterations": profile.get("iterations", ""),
    }


def main() -> None:
    args = parse_args()
    preflight_dir = Path(args.preflight_dir)
    sweep_dir = Path(args.sweep_dir)

    sample = load_visium_lite(args.sample_dir)
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    barrier_grid = np.load(preflight_dir / "barrier_grid.npy")
    tissue_mask = np.load(preflight_dir / "tissue_mask.npy")
    source_values = clip_and_normalize(
        normalized_gene_vector(sample, args.target_gene),
        percentile=args.source_percentile,
    )

    rows = []
    for run_dir in sorted(path for path in sweep_dir.iterdir() if path.is_dir()):
        prediction_path = run_dir / args.prediction_name
        if not prediction_path.exists():
            continue
        prediction_grid = np.load(prediction_path)
        metrics = evaluate_field(
            coords_norm=coords_norm,
            source_spot_values=source_values,
            prediction_grid=prediction_grid,
            barrier_grid=barrier_grid,
            tissue_mask=tissue_mask,
        ).to_dict()
        row = {
            **profile_metadata(run_dir / "profile.json"),
            **metrics,
        }
        row.setdefault("run", run_dir.name)
        rows.append(row)
        with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(row, handle, indent=2)

    if not rows:
        raise SystemExit(f"No predictions named {args.prediction_name} found in {sweep_dir}")

    csv_path = sweep_dir / "sweep_metrics.csv"
    columns = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Refreshed {len(rows)} sweep rows at {csv_path}")


if __name__ == "__main__":
    main()
