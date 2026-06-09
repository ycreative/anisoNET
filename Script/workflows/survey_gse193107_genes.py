"""Survey candidate gene expression coverage in standardized GSE193107 samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from anisonet.visium_io import load_visium_lite, normalized_gene_vector


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
PROCESSED_ROOT = PROJECT_ROOT / "codexAnalysis" / "processed_visium" / "brain_aging_gse193107"


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
    parser = argparse.ArgumentParser(description="Survey GSE193107 candidate genes.")
    parser.add_argument("--genes", nargs="+", required=True)
    parser.add_argument("--samples", nargs="*", default=SAMPLES)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def condition(sample: str) -> str:
    return "Young" if "_Young_" in sample else "Old" if "_Old_" in sample else "Unknown"


def main() -> None:
    args = parse_args()
    rows = []
    for sample_name in args.samples:
        sample = load_visium_lite(PROCESSED_ROOT / sample_name)
        for gene in args.genes:
            if gene not in sample.genes:
                rows.append(
                    {
                        "sample": sample_name,
                        "condition": condition(sample_name),
                        "gene": gene,
                        "present": False,
                        "nonzero_fraction": "",
                        "mean_lognorm": "",
                        "p95_lognorm": "",
                        "max_lognorm": "",
                    }
                )
                continue
            values = normalized_gene_vector(sample, gene)
            rows.append(
                {
                    "sample": sample_name,
                    "condition": condition(sample_name),
                    "gene": gene,
                    "present": True,
                    "nonzero_fraction": float(np.mean(values > 0)),
                    "mean_lognorm": float(np.mean(values)),
                    "p95_lognorm": float(np.percentile(values, 95)),
                    "max_lognorm": float(np.max(values)),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote gene survey to {output_csv}")


if __name__ == "__main__":
    main()
