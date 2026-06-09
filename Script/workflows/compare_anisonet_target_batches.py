"""Compare grouped anisoNET batch summaries across target genes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare anisoNET target batch summaries.")
    parser.add_argument("--target-summary", action="append", required=True, help="Target=path/to/group_summary.csv")
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def parse_target_summary(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--target-summary must use Target=path")
    target, path = value.split("=", 1)
    return target, Path(path)


def main() -> None:
    args = parse_args()
    rows = []
    for target, path in map(parse_target_summary, args.target_summary):
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "target": target,
                        "condition": row["condition"],
                        "field_type": row["field_type"],
                        "n": row["n"],
                        "pearson_mean": row["spot_pearson_source_mean"],
                        "pearson_sem": row["spot_pearson_source_sem"],
                        "mse_mean": row["spot_mse_source_mean"],
                        "mse_sem": row["spot_mse_source_sem"],
                        "roughness_p95_mean": row["roughness_grad_p95_mean"],
                        "roughness_p95_sem": row["roughness_grad_p95_sem"],
                        "barrier_pearson_mean": row["spot_pearson_barrier_mean"],
                    }
                )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote target comparison to {output_csv}")


if __name__ == "__main__":
    main()
