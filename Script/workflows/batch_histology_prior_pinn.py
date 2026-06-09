"""Run representative PINN comparisons for H&E histology priors."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
PYTHON = Path(r"K:\software\miniconda\envs\scvi_env\python.exe")
PROCESSED_ROOT = PROJECT_ROOT / "codexAnalysis" / "processed_visium" / "brain_aging_gse193107"
HISTOLOGY_ROOT = PROJECT_ROOT / "codexAnalysis" / "histology_prior" / "brain_aging_gse193107"

DEFAULT_SAMPLES = [
    "GSM5773453_Young_mouse_brain_A1-1",
    "GSM5773454_Young_mouse_brain_B1-1",
    "GSM5773457_Old_mouse_brain_A1-2",
    "GSM5773458_Old_mouse_brain_B1-2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run representative histology-prior PINN comparisons.")
    parser.add_argument("--samples", nargs="*", default=DEFAULT_SAMPLES)
    parser.add_argument("--target-gene", action="append", default=["Apoe", "Gfap"])
    parser.add_argument("--profile", default="fourier_refined_16g")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--postprocess-sigma", type=float, default=0.7)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def condition(sample: str) -> str:
    return "Young" if "_Young_" in sample else "Old" if "_Old_" in sample else "Unknown"


def script(name: str) -> str:
    return str(PROJECT_ROOT / "Script" / "workflows" / name)


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    script_root = str(PROJECT_ROOT / "Script")
    env["PYTHONPATH"] = script_root + os.pathsep + env.get("PYTHONPATH", "")
    print("RUN " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_metric_json(path: Path, *, sample: str, histology_prior: str, field_type: str) -> dict[str, object]:
    payload = read_json(path)
    row: dict[str, object] = {
        "sample": sample,
        "condition": condition(sample),
        "histology_prior": histology_prior,
        "field_type": field_type,
        "method": payload.get("method", ""),
        "target_gene": payload.get("target_gene", ""),
    }
    row.update(payload.get("metrics", {}))
    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample",
        "condition",
        "target_gene",
        "histology_prior",
        "field_type",
        "method",
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "roughness_grad_mean",
        "roughness_laplacian_energy",
        "spot_pearson_barrier",
        "high_to_low_barrier_prediction_ratio",
        "background_to_tissue_ratio",
    ]
    extras = sorted({key for row in rows for key in row if key not in columns})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + extras)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = []
    for target_gene in args.target_gene:
        target_name = f"{target_gene}_CNS_Myelin"
        for sample in args.samples:
            sample_dir = PROCESSED_ROOT / sample
            for prior in ["brightness", "hematoxylin"]:
                preflight_dir = HISTOLOGY_ROOT / sample / target_name / prior
                pinn_dir = HISTOLOGY_ROOT / sample / target_name / f"{prior}_pinn"
                postprocessed_grid = pinn_dir / "pinn_grid_prediction_postprocessed.npy"
                if not args.skip_existing or not postprocessed_grid.exists():
                    run(
                        [
                            str(PYTHON),
                            script("run_anisonet_pinn.py"),
                            "--sample-dir",
                            str(sample_dir),
                            "--preflight-dir",
                            str(preflight_dir),
                            "--target-gene",
                            target_gene,
                            "--output-dir",
                            str(pinn_dir),
                            "--profile",
                            args.profile,
                            "--device",
                            args.device,
                            "--postprocess-sigma",
                            str(args.postprocess_sigma),
                        ]
                    )

                for field_type, prediction_grid in [
                    ("masked", pinn_dir / "pinn_grid_prediction_clean_tissue_masked.npy"),
                    ("gauss07", postprocessed_grid),
                ]:
                    metric_json = pinn_dir / f"field_metrics_{field_type}.json"
                    run(
                        [
                            str(PYTHON),
                            script("evaluate_anisonet_field.py"),
                            "--sample-dir",
                            str(sample_dir),
                            "--preflight-dir",
                            str(preflight_dir),
                            "--target-gene",
                            target_gene,
                            "--prediction-grid",
                            str(prediction_grid),
                            "--method-name",
                            f"histology_{prior}_{field_type}",
                            "--output-json",
                            str(metric_json),
                        ]
                    )
                    rows.append(
                        flatten_metric_json(
                            metric_json,
                            sample=sample,
                            histology_prior=prior,
                            field_type=field_type,
                        )
                    )

    summary_csv = HISTOLOGY_ROOT / "histology_prior_pinn_metrics_summary.csv"
    write_csv(rows, summary_csv)
    print(f"Wrote histology prior PINN summary to {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
