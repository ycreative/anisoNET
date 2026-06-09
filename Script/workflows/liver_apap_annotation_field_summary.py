"""Summarize liver APAP Visium fields against provided spot annotations."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.metrics import sample_grid_at_spots
from anisonet.visium_io import load_visium_lite, normalized_gene_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize APAP liver annotation groups and anisoNET preflight fields.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--annotation-csv", required=True)
    parser.add_argument("--annotation-sample", required=True, help="Annotation sample label, e.g. ABU7 or ABU8.")
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--genes", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def barcode_core(value: str) -> str:
    return str(value).split("-")[0]


def safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def safe_median(values: pd.Series) -> float:
    return float(values.median()) if len(values) else float("nan")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample = load_visium_lite(args.sample_dir)
    annotations = pd.read_csv(args.annotation_csv)
    annotations = annotations.loc[annotations["sample"].astype(str) == args.annotation_sample].copy()
    annotations["barcode_core"] = annotations["cell"].map(barcode_core)

    table = pd.DataFrame(
        {
            "barcode": sample.barcodes,
            "barcode_core": [barcode_core(b) for b in sample.barcodes],
            "spot_index": np.arange(len(sample.barcodes)),
        }
    )
    merged = table.merge(
        annotations[
            [
                "barcode_core",
                "cell",
                "sample",
                "nCount_Spatial",
                "nFeature_Spatial",
                "percent.mt",
                "seurat_clusters",
                "zonationGroup",
            ]
        ],
        on="barcode_core",
        how="left",
    )
    merged["has_annotation"] = merged["zonationGroup"].notna()

    for gene in args.genes:
        if gene in sample.genes:
            merged[f"gene_{gene}"] = normalized_gene_vector(sample, gene)
        else:
            merged[f"gene_{gene}"] = np.nan

    preflight_dir = Path(args.preflight_dir)
    coords_norm = np.load(preflight_dir / "coords_norm.npy")
    field_files = {
        "field_source": "source_grid.npy",
        "field_barrier": "barrier_grid.npy",
        "field_resistance": "resistance_grid.npy",
        "field_histology_resistance": "histology_resistance_grid.npy",
        "field_diffusion": "diffusion_grid.npy",
    }
    for column, filename in field_files.items():
        grid = np.load(preflight_dir / filename)
        merged[column] = sample_grid_at_spots(grid, coords_norm)

    spot_csv = output_dir / "liver_apap_annotation_spot_table.csv"
    merged.to_csv(spot_csv, index=False)

    annotated = merged.loc[merged["has_annotation"]].copy()
    group_rows = []
    value_columns = [c for c in merged.columns if c.startswith("gene_") or c.startswith("field_")]
    for group, group_frame in annotated.groupby("zonationGroup", dropna=False):
        row: dict[str, object] = {
            "zonationGroup": group,
            "n_spots": int(len(group_frame)),
            "annotation_sample": args.annotation_sample,
            "sample_dir": str(Path(args.sample_dir).resolve()),
            "preflight_dir": str(preflight_dir.resolve()),
        }
        for column in value_columns:
            row[f"{column}_mean"] = safe_mean(group_frame[column])
            row[f"{column}_median"] = safe_median(group_frame[column])
            row[f"{column}_p95"] = float(group_frame[column].quantile(0.95))
        group_rows.append(row)

    summary_csv = output_dir / "liver_apap_annotation_group_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(group_rows[0].keys()))
        writer.writeheader()
        writer.writerows(group_rows)

    save_summary_plot(annotated, output_dir, args.genes)
    print(f"Wrote {spot_csv}")
    print(f"Wrote {summary_csv}")


def save_summary_plot(frame: pd.DataFrame, output_dir: Path, genes: list[str]) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "Arial", "pdf.fonttype": 42, "ps.fonttype": 42})
    groups = ["Central", "Mid", "PeriPortal", "Portal"]
    columns = [f"gene_{gene}" for gene in genes[:4]] + ["field_source", "field_barrier", "field_resistance"]
    columns = [column for column in columns if column in frame.columns]
    n_cols = min(3, len(columns))
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.0 * n_cols, 2.4 * n_rows), constrained_layout=True)
    axes_arr = np.atleast_1d(axes).reshape(-1)
    for ax, column in zip(axes_arr, columns):
        values = [frame.loc[frame["zonationGroup"] == group, column].dropna().to_numpy() for group in groups]
        ax.boxplot(values, tick_labels=groups, showfliers=False, widths=0.55)
        ax.set_title(column.replace("gene_", "").replace("field_", "field "), fontsize=8)
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
    for ax in axes_arr[len(columns) :]:
        ax.axis("off")
    fig.savefig(output_dir / "liver_apap_annotation_group_summary.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "liver_apap_annotation_group_summary.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
