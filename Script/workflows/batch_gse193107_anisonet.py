"""Run anisoNET Apoe/CNS-myelin validation across GSE193107 brain sections."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
PYTHON = Path(r"K:\software\miniconda\envs\scvi_env\python.exe")
RAW_ROOT = PROJECT_ROOT / "dataset" / "分析数据集" / "GSE193107_RAW"
PROCESSED_ROOT = PROJECT_ROOT / "codexAnalysis" / "processed_visium" / "brain_aging_gse193107"
PREFLIGHT_ROOT = PROJECT_ROOT / "codexAnalysis" / "preflight" / "brain_aging_gse193107"
PINN_ROOT = PROJECT_ROOT / "codexAnalysis" / "pinn" / "brain_aging_gse193107"
METRICS_ROOT = PROJECT_ROOT / "codexAnalysis" / "metrics" / "brain_aging_gse193107"
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
    parser = argparse.ArgumentParser(description="Batch-run anisoNET on GSE193107 brain sections.")
    parser.add_argument("--samples", nargs="*", default=SAMPLES)
    parser.add_argument("--target-gene", default="Apoe")
    parser.add_argument("--barrier-genes", nargs="*", default=["Mbp", "Plp1", "Mobp"])
    parser.add_argument("--profile", default="fourier_refined_16g")
    parser.add_argument("--postprocess-sigma", type=float, default=0.7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite-standardized", action="store_true")
    parser.add_argument("--skip-standardize", action="store_true")
    parser.add_argument("--skip-existing-pinn", action="store_true")
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


def flatten_metric_json(path: Path, *, sample: str, field_type: str) -> dict[str, object]:
    payload = read_json(path)
    row: dict[str, object] = {
        "sample": sample,
        "condition": condition(sample),
        "field_type": field_type,
        "method": payload.get("method", ""),
        "target_gene": payload.get("target_gene", ""),
    }
    row.update(payload.get("metrics", {}))
    return row


def write_summary(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "sample",
        "condition",
        "field_type",
        "method",
        "target_gene",
        "spot_pearson_source",
        "spot_mse_source",
        "roughness_grad_p95",
        "roughness_grad_mean",
        "roughness_laplacian_energy",
        "background_to_tissue_ratio",
        "spot_pearson_barrier",
        "high_to_low_barrier_prediction_ratio",
    ]
    extras = sorted({key for row in rows for key in row if key not in columns})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + extras)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    target_name = f"{args.target_gene}_CNS_Myelin"
    summary_root = PROJECT_ROOT / "codexAnalysis" / "batch" / "brain_aging_gse193107" / target_name
    summary_root.mkdir(parents=True, exist_ok=True)
    rows = []

    for sample in args.samples:
        sample_dir = PROCESSED_ROOT / sample
        preflight_dir = PREFLIGHT_ROOT / sample / target_name
        pinn_dir = PINN_ROOT / sample / target_name / f"{args.profile}_gauss{int(args.postprocess_sigma * 10):02d}_batch"
        metrics_dir = METRICS_ROOT / sample / target_name
        metrics_dir.mkdir(parents=True, exist_ok=True)

        standardized_ready = (sample_dir / "filtered_feature_bc_matrix.h5").exists() and (
            sample_dir / "spatial" / "tissue_hires_image.png"
        ).exists()
        if not args.skip_standardize or not standardized_ready:
            cmd = [
                str(PYTHON),
                script("standardize_gse193107_visium.py"),
                "--raw-dir",
                str(RAW_ROOT),
                "--output-root",
                str(PROCESSED_ROOT),
                "--sample",
                sample,
            ]
            if args.overwrite_standardized:
                cmd.append("--overwrite")
            run(cmd)

        run(
            [
                str(PYTHON),
                script("run_anisonet_preflight.py"),
                "--sample-dir",
                str(sample_dir),
                "--target-gene",
                args.target_gene,
                "--barrier-genes",
                *args.barrier_genes,
                "--output-dir",
                str(preflight_dir),
            ]
        )

        if not args.skip_existing_pinn or not (pinn_dir / "pinn_grid_prediction_postprocessed.npy").exists():
            run(
                [
                    str(PYTHON),
                    script("run_anisonet_pinn.py"),
                    "--sample-dir",
                    str(sample_dir),
                    "--preflight-dir",
                    str(preflight_dir),
                    "--target-gene",
                    args.target_gene,
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

        metric_specs = [
            ("masked", pinn_dir / "pinn_grid_prediction_clean_tissue_masked.npy"),
            ("gauss07", pinn_dir / "pinn_grid_prediction_postprocessed.npy"),
        ]
        for field_type, prediction_path in metric_specs:
            metric_json = metrics_dir / f"batch_{args.profile}_{field_type}.json"
            run(
                [
                    str(PYTHON),
                    script("evaluate_anisonet_field.py"),
                    "--sample-dir",
                    str(sample_dir),
                    "--preflight-dir",
                    str(preflight_dir),
                    "--target-gene",
                    args.target_gene,
                    "--prediction-grid",
                    str(prediction_path),
                    "--method-name",
                    f"batch_{args.profile}_{field_type}",
                    "--output-json",
                    str(metric_json),
                ]
            )
            rows.append(flatten_metric_json(metric_json, sample=sample, field_type=field_type))

    summary_csv = summary_root / "batch_metrics_summary.csv"
    write_summary(rows, summary_csv)
    manifest = {
        "target_gene": args.target_gene,
        "barrier_genes": args.barrier_genes,
        "profile": args.profile,
        "postprocess_sigma": args.postprocess_sigma,
        "samples": args.samples,
        "summary_csv": str(summary_csv),
    }
    with (summary_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
