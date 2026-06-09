"""Rank anisoNET candidate fields by fit, leakage, and roughness."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


POSITIVE_METRICS = {
    "spot_pearson_source": 0.35,
}

NEGATIVE_METRICS = {
    "spot_mse_source": 0.25,
    "roughness_grad_p95": 0.20,
    "roughness_grad_mean": 0.10,
    "background_to_tissue_ratio": 0.10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank anisoNET candidate metric rows.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md")
    parser.add_argument(
        "--require-roughness",
        action="store_true",
        help="Drop rows without roughness metrics. Useful when selecting PINN candidates.",
    )
    parser.add_argument(
        "--method-contains",
        default="",
        help="Optional substring filter applied to method or run names.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def finite_values(rows: Iterable[dict[str, str]], key: str) -> list[float]:
    values = [numeric(row, key) for row in rows]
    return [value for value in values if math.isfinite(value)]


def minmax_score(value: float, values: list[float], *, higher_is_better: bool) -> float:
    if not math.isfinite(value):
        return 0.0
    if not values:
        return 0.0
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-12:
        return 1.0
    score = (value - lo) / (hi - lo)
    if not higher_is_better:
        score = 1.0 - score
    return max(0.0, min(1.0, score))


def display_name(row: dict[str, str]) -> str:
    return row.get("method") or row.get("run") or row.get("file") or "candidate"


def filter_rows(rows: list[dict[str, str]], *, require_roughness: bool, method_contains: str) -> list[dict[str, str]]:
    out = []
    needle = method_contains.lower()
    for row in rows:
        name = display_name(row).lower()
        if needle and needle not in name:
            continue
        if require_roughness and not math.isfinite(numeric(row, "roughness_grad_p95")):
            continue
        out.append(row)
    return out


def rank_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    positive_values = {key: finite_values(rows, key) for key in POSITIVE_METRICS}
    negative_values = {key: finite_values(rows, key) for key in NEGATIVE_METRICS}
    ranked = []
    for row in rows:
        score = 0.0
        components: dict[str, float] = {}
        for key, weight in POSITIVE_METRICS.items():
            component = minmax_score(numeric(row, key), positive_values[key], higher_is_better=True)
            components[f"score_{key}"] = component
            score += component * weight
        for key, weight in NEGATIVE_METRICS.items():
            component = minmax_score(numeric(row, key), negative_values[key], higher_is_better=False)
            components[f"score_{key}"] = component
            score += component * weight

        ranked_row = dict(row)
        ranked_row["candidate"] = display_name(row)
        ranked_row["selection_score"] = f"{score:.6f}"
        for key, value in components.items():
            ranked_row[key] = f"{value:.6f}"
        ranked.append(ranked_row)

    ranked.sort(key=lambda row: float(row["selection_score"]), reverse=True)
    for index, row in enumerate(ranked, start=1):
        row["rank"] = str(index)
    return ranked


def write_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank",
        "candidate",
        "selection_score",
        "spot_pearson_source",
        "spot_mse_source",
        "background_to_tissue_ratio",
        "roughness_grad_mean",
        "roughness_grad_p95",
        "roughness_laplacian_energy",
        "spot_pearson_barrier",
        "high_to_low_barrier_prediction_ratio",
    ]
    extra = [key for key in rows[0].keys() if key not in columns] if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns + extra)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# anisoNET Candidate Ranking",
        "",
        "Selection score weights: source Pearson 0.35, source MSE 0.25, roughness p95 0.20, roughness mean 0.10, background/tissue leakage 0.10.",
        "All components are min-max scaled within the provided candidate table; higher score is better.",
        "",
        "| Rank | Candidate | Score | Pearson | MSE | Rough p95 | Rough mean | Leakage |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {rank} | {candidate} | {selection_score} | {pearson:.3f} | {mse:.5f} | {rp95:.4f} | {rmean:.4f} | {leak:.3f} |".format(
                rank=row.get("rank", ""),
                candidate=row.get("candidate", ""),
                selection_score=row.get("selection_score", ""),
                pearson=numeric(row, "spot_pearson_source"),
                mse=numeric(row, "spot_mse_source"),
                rp95=numeric(row, "roughness_grad_p95"),
                rmean=numeric(row, "roughness_grad_mean"),
                leak=numeric(row, "background_to_tissue_ratio"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = filter_rows(
        read_rows(Path(args.input_csv)),
        require_roughness=args.require_roughness,
        method_contains=args.method_contains,
    )
    if not rows:
        raise SystemExit("No candidate rows available after filtering.")
    ranked = rank_rows(rows)
    write_csv(ranked, Path(args.output_csv))
    if args.output_md:
        write_markdown(ranked, Path(args.output_md))
    print(f"Ranked {len(ranked)} candidates; best={ranked[0]['candidate']}")


if __name__ == "__main__":
    main()
