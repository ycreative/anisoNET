"""Survey candidate gene expression coverage in one standardized Visium sample."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.visium_io import load_visium_lite, normalized_gene_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Survey candidate genes in a standardized Visium sample.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--genes", nargs="+", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample = load_visium_lite(args.sample_dir)
    rows = []
    for gene in args.genes:
        if gene not in sample.genes:
            rows.append(
                {
                    "gene": gene,
                    "present": False,
                    "nonzero_fraction": "",
                    "mean_lognorm": "",
                    "p50_lognorm": "",
                    "p95_lognorm": "",
                    "max_lognorm": "",
                }
            )
            continue
        values = normalized_gene_vector(sample, gene)
        rows.append(
            {
                "gene": gene,
                "present": True,
                "nonzero_fraction": float(np.mean(values > 0)),
                "mean_lognorm": float(np.mean(values)),
                "p50_lognorm": float(np.percentile(values, 50)),
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

