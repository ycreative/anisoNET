"""Collect anisoNET JSON metric files into a CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect anisoNET metrics JSON files.")
    parser.add_argument("--metrics-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_dir = Path(args.metrics_dir)
    rows = []
    metric_names = set()
    for path in sorted(metrics_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        row = {
            "file": path.name,
            "method": payload.get("method", ""),
            "target_gene": payload.get("target_gene", ""),
        }
        metrics = payload.get("metrics", {})
        metric_names.update(metrics.keys())
        row.update(metrics)
        rows.append(row)

    metric_columns = sorted(metric_names)
    columns = ["file", "method", "target_gene"] + metric_columns
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
