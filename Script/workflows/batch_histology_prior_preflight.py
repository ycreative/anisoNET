"""Batch-compare brightness and hematoxylin H&E priors on GSE193107."""

from __future__ import annotations

import os

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
PYTHON = Path(os.environ.get("ANISONET_PYTHON", sys.executable))
PROCESSED_ROOT = PROJECT_ROOT / "codexAnalysis" / "processed_visium" / "brain_aging_gse193107"
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "histology_prior" / "brain_aging_gse193107"

SAMPLES = [
    "GSM5773453_Young_mouse_brain_A1-1",
    "GSM5773454_Young_mouse_brain_B1-1",
    "GSM5773455_Young_mouse_brain_C1-1",
    "GSM5773456_Young_mouse_brain_D1-1",
    "GSM5773457_Old_mouse_brain_A1-2",
    "GSM5773458_Old_mouse_brain_B1-2",
    "GSM5773459_Old_mouse_brain_C1-2",
    "GSM5773460_Old_mouse_brain_D1-2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H&E prior preflight comparison across GSE193107.")
    parser.add_argument("--samples", nargs="*", default=SAMPLES)
    parser.add_argument("--target-gene", action="append", default=["Apoe", "Gfap"])
    parser.add_argument("--barrier-genes", nargs="*", default=["Mbp", "Plp1", "Mobp"])
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("RUN " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def script(name: str) -> str:
    return str(PROJECT_ROOT / "Script" / "workflows" / name)


def condition(sample: str) -> str:
    return "Young" if "_Young_" in sample else "Old" if "_Old_" in sample else "Unknown"


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    rows = []
    for target_gene in args.target_gene:
        target_name = f"{target_gene}_CNS_Myelin"
        for sample in args.samples:
            sample_dir = PROCESSED_ROOT / sample
            for prior in ["brightness", "hematoxylin"]:
                output_dir = OUTPUT_ROOT / sample / target_name / prior
                metrics_json = output_dir / "preflight_metrics.json"
                if not args.skip_existing or not metrics_json.exists():
                    run(
                        [
                            str(PYTHON),
                            script("run_anisonet_preflight.py"),
                            "--sample-dir",
                            str(sample_dir),
                            "--target-gene",
                            target_gene,
                            "--barrier-genes",
                            *args.barrier_genes,
                            "--output-dir",
                            str(output_dir),
                            "--histology-prior",
                            prior,
                        ]
                    )
                metrics = read_json(metrics_json)
                rows.append(
                    {
                        "sample": sample,
                        "condition": condition(sample),
                        "target_gene": target_gene,
                        "histology_prior": prior,
                        "n_spots": metrics["n_spots"],
                        "histology_resistance_mean_in_tissue": metrics["histology_resistance_mean_in_tissue"],
                        "histology_resistance_p95_in_tissue": metrics["histology_resistance_p95_in_tissue"],
                        "diffusion_min_in_tissue": metrics["diffusion_min_in_tissue"],
                        "diffusion_max_in_tissue": metrics["diffusion_max_in_tissue"],
                        "resistance_ratio_in_tissue": metrics["resistance_ratio_in_tissue"],
                        "barrier_grid_max": metrics["barrier_grid_max"],
                        "source_grid_max": metrics["source_grid_max"],
                    }
                )
    summary_csv = OUTPUT_ROOT / "histology_prior_preflight_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote histology prior preflight summary to {summary_csv}", flush=True)


if __name__ == "__main__":
    main()

