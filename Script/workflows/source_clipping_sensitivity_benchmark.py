"""Benchmark sensitivity to source-expression clipping percentile."""

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
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "source_clipping_sensitivity" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run source clipping sensitivity benchmark.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument("--target-gene", action="append", default=["Apoe", "Gfap"])
    parser.add_argument("--source-percentile", type=float, nargs="*", default=[95.0, 97.5, 99.0, 99.5])
    parser.add_argument("--eval-source-percentile", type=float, default=99.0)
    parser.add_argument("--profile", default="fourier_refined_16g")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--postprocess-sigma", type=float, default=0.7)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


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


def flatten_metric_json(
    path: Path,
    *,
    sample: str,
    train_source_percentile: float,
    eval_source_percentile: float,
    field_type: str,
) -> dict[str, object]:
    payload = read_json(path)
    row: dict[str, object] = {
        "sample": sample,
        "target_gene": payload.get("target_gene", ""),
        "train_source_percentile": float(train_source_percentile),
        "eval_source_percentile": float(eval_source_percentile),
        "field_type": field_type,
        "method": payload.get("method", ""),
    }
    row.update(payload.get("metrics", {}))
    return row


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    columns = [
        "sample",
        "target_gene",
        "train_source_percentile",
        "eval_source_percentile",
        "field_type",
        "method",
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "roughness_grad_mean",
        "spot_pearson_barrier",
        "high_to_low_barrier_prediction_ratio",
        "background_to_tissue_ratio",
    ]
    extras = sorted({key for row in rows for key in row if key not in columns})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + extras)
        writer.writeheader()
        writer.writerows(rows)


def label_percentile(value: float) -> str:
    return f"p{str(value).replace('.', 'p')}"


def main() -> None:
    args = parse_args()
    sample_dir = PROCESSED_ROOT / args.sample
    rows = []
    for target_gene in args.target_gene:
        target_name = f"{target_gene}_CNS_Myelin"
        for percentile in args.source_percentile:
            label = label_percentile(percentile)
            run_dir = OUTPUT_ROOT / args.sample / target_name / label
            preflight_dir = run_dir / "preflight"
            pinn_dir = run_dir / "pinn"
            postprocessed = pinn_dir / "pinn_grid_prediction_postprocessed.npy"
            if not args.skip_existing or not (preflight_dir / "preflight_metrics.json").exists():
                run(
                    [
                        str(PYTHON),
                        script("run_anisonet_preflight.py"),
                        "--sample-dir",
                        str(sample_dir),
                        "--target-gene",
                        target_gene,
                        "--output-dir",
                        str(preflight_dir),
                        "--source-percentile",
                        str(percentile),
                        "--histology-prior",
                        "brightness",
                        "--skip-figure",
                    ]
                )
            if not args.skip_existing or not postprocessed.exists():
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
                        "--source-percentile",
                        str(percentile),
                        "--postprocess-sigma",
                        str(args.postprocess_sigma),
                    ]
                )
            for field_type, grid in [
                ("masked", pinn_dir / "pinn_grid_prediction_clean_tissue_masked.npy"),
                ("gauss07", postprocessed),
            ]:
                metric_json = pinn_dir / f"field_metrics_{field_type}_evalp{str(args.eval_source_percentile).replace('.', 'p')}.json"
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
                        str(grid),
                        "--method-name",
                        f"source_clip_{label}_{field_type}",
                        "--output-json",
                        str(metric_json),
                        "--source-percentile",
                        str(args.eval_source_percentile),
                    ]
                )
                rows.append(
                    flatten_metric_json(
                        metric_json,
                        sample=args.sample,
                        train_source_percentile=percentile,
                        eval_source_percentile=args.eval_source_percentile,
                        field_type=field_type,
                    )
                )
    summary = OUTPUT_ROOT / args.sample / "source_clipping_sensitivity_summary.csv"
    write_csv(rows, summary)
    print(f"Wrote source clipping sensitivity summary to {summary}", flush=True)


if __name__ == "__main__":
    main()
