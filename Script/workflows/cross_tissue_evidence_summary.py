"""Build a manuscript-facing summary of cross-tissue evidence boundaries.

This script consolidates existing kidney, liver/APAP, and sagittal brain
outputs. It does not rerun preflight, interpolation, or PINN analyses.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = Path("codexAnalysis/cross_tissue/evidence_summary")
KIDNEY_TABLE = Path(
    "codexAnalysis/barrier_split_anisonet/mouse_kidney_10x/"
    "V1_Mouse_Kidney/evidence_boundary_diagnostics/kidney_supplementary_summary_table.csv"
)
LIVER_BOUNDARY = Path(
    "codexAnalysis/annotation_boundary_benchmark/mouse_liver_apap_gse280515/"
    "summary/liver_annotation_boundary_resistance_vs_euclidean_idw_summary.csv"
)


SAGITTAL_RESULTS = [
    {
        "dataset": "10x mouse brain sagittal",
        "task_or_target": "Section1 Apoe",
        "comparison": "resistance-IDW vs Euclidean-IDW, high-barrier/edge",
        "primary_metric": "Pearson delta",
        "candidate_value": 0.6195 - 0.6276,
        "secondary_metric": "MSE delta",
        "candidate_secondary_value": 0.01129 - 0.01042,
    },
    {
        "dataset": "10x mouse brain sagittal",
        "task_or_target": "Section1 Gfap",
        "comparison": "resistance-IDW vs Euclidean-IDW, high-barrier/edge",
        "primary_metric": "Pearson delta",
        "candidate_value": 0.3670 - 0.3900,
        "secondary_metric": "MSE delta",
        "candidate_secondary_value": 0.11763 - 0.11313,
    },
    {
        "dataset": "10x mouse brain sagittal",
        "task_or_target": "Section2 Apoe",
        "comparison": "resistance-IDW vs Euclidean-IDW, high-barrier/edge",
        "primary_metric": "Pearson delta",
        "candidate_value": 0.5481 - 0.5726,
        "secondary_metric": "MSE delta",
        "candidate_secondary_value": 0.01043 - 0.00979,
    },
    {
        "dataset": "10x mouse brain sagittal",
        "task_or_target": "Section2 Gfap",
        "comparison": "resistance-IDW vs Euclidean-IDW, high-barrier/edge",
        "primary_metric": "Pearson delta",
        "candidate_value": 0.2576 - 0.3389,
        "secondary_metric": "MSE delta",
        "candidate_secondary_value": 0.12407 - 0.11303,
    },
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def format_markdown_table(table: pd.DataFrame) -> str:
    display = table.copy()
    numeric_cols = ["baseline_value", "candidate_value", "baseline_secondary_value", "candidate_secondary_value"]
    for col in numeric_cols:
        display[col] = display[col].map(lambda x: f"{x:.5f}" if isinstance(x, (int, float)) else str(x))
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[col]).replace("|", "/") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def kidney_rows(kidney: pd.DataFrame) -> pd.DataFrame:
    rows = kidney.copy()
    rows.insert(0, "dataset", "10x mouse kidney")
    rows["summary_role"] = rows["claim_level"].map(
        {
            "strong baseline / claim-boundary": "claim-boundary",
            "method-development": "method-development",
            "method diagnostic": "method diagnostic",
            "supplementary": "supplementary",
        }
    ).fillna(rows["claim_level"])
    return rows[
        [
            "dataset",
            "evidence_block",
            "task_or_target",
            "comparison",
            "primary_metric",
            "baseline_value",
            "candidate_value",
            "secondary_metric",
            "baseline_secondary_value",
            "candidate_secondary_value",
            "summary_role",
            "interpretation",
        ]
    ]


def liver_rows(liver: pd.DataFrame) -> pd.DataFrame:
    keep = liver[liver["split_mode"] == "high_barrier_or_edge"].copy()
    rows = []
    for _, row in keep.iterrows():
        target = row["target_gene"]
        if target == "Cps1":
            role = "supplementary positive-leaning"
            interpretation = (
                "Best liver annotation-boundary candidate; small gains in Pearson, MSE, "
                "contrast retention, and leakage ratio."
            )
        elif target == "Cyp2f2":
            role = "supplementary mixed"
            interpretation = "Improves contrast retention/leakage but MSE is not consistently favorable."
        else:
            role = "claim-boundary"
            interpretation = "Improves annotation-boundary metrics but not Pearson/MSE."
        rows.append(
            {
                "dataset": "mouse liver APAP GSE280515",
                "evidence_block": "annotation_boundary",
                "task_or_target": target,
                "comparison": "resistance-IDW minus Euclidean-IDW, high-barrier/edge",
                "primary_metric": "Pearson delta",
                "baseline_value": 0.0,
                "candidate_value": row["mean_pearson_delta"],
                "secondary_metric": "MSE delta; contrast retention delta; leakage ratio delta",
                "baseline_secondary_value": 0.0,
                "candidate_secondary_value": (
                    f"MSE {row['mean_mse_delta']:+.5f}; "
                    f"retention {row['mean_contrast_retention_delta']:+.5f}; "
                    f"leakage {row['mean_leakage_ratio_delta']:+.5f}"
                ),
                "summary_role": role,
                "interpretation": interpretation,
            }
        )
    return pd.DataFrame(rows)


def sagittal_rows() -> pd.DataFrame:
    rows = pd.DataFrame(SAGITTAL_RESULTS)
    rows.insert(1, "evidence_block", "healthy_brain_portability")
    rows["baseline_value"] = 0.0
    rows["baseline_secondary_value"] = 0.0
    rows["summary_role"] = "claim-boundary / negative control"
    rows["interpretation"] = (
        "Preflight is portable, but resistance-IDW underperforms Euclidean-IDW in this healthy brain task."
    )
    return rows[
        [
            "dataset",
            "evidence_block",
            "task_or_target",
            "comparison",
            "primary_metric",
            "baseline_value",
            "candidate_value",
            "secondary_metric",
            "baseline_secondary_value",
            "candidate_secondary_value",
            "summary_role",
            "interpretation",
        ]
    ]


def write_interpretation(summary: pd.DataFrame, out_path: Path) -> None:
    kidney_umod = summary[(summary["dataset"] == "10x mouse kidney") & (summary["task_or_target"] == "Umod")].iloc[0]
    liver_cps1 = summary[(summary["dataset"] == "mouse liver APAP GSE280515") & (summary["task_or_target"] == "Cps1")].iloc[0]
    sagittal = summary[summary["dataset"] == "10x mouse brain sagittal"]
    sagittal_neg = int((pd.to_numeric(sagittal["candidate_value"], errors="coerce") < 0).sum())

    text = f"""# Cross-Tissue Evidence Summary

Date: 2026-06-08

## Purpose

This summary consolidates non-primary datasets into manuscript-facing evidence roles. It is intended to prevent overclaiming while preserving useful cross-tissue validation and claim-boundary evidence.

## Dataset Roles

- Mouse kidney: benchmark-development evidence. `Umod` is the strongest prior-development signal; `Slc34a1` is a near-ceiling resistance-IDW baseline; `Slc12a3` is supplementary.
- Mouse liver APAP: disease/injury validation-design evidence. The initial `Lyz2/Spp1` task was inappropriate because injury markers are Central-enriched; `Cps1` and `Cyp2f2` provide small annotation-boundary signals.
- 10x mouse brain sagittal: healthy portability and negative-control evidence. Preflight construction works, but resistance-aware interpolation is negative in all four Apoe/Gfap high-barrier/edge comparisons.

## Key Quantitative Anchors

- Kidney `Umod`: {kidney_umod['comparison']}; candidate held-out Pearson {float(kidney_umod['candidate_value']):.4f} versus baseline {float(kidney_umod['baseline_value']):.4f}.
- Liver `Cps1`: high-barrier/edge Pearson delta {float(liver_cps1['candidate_value']):+.4f}; {liver_cps1['candidate_secondary_value']}.
- Sagittal brain: resistance-IDW Pearson delta is negative in {sagittal_neg}/4 high-barrier/edge comparisons.

## Manuscript Implication

The cross-tissue analyses support portability of the preprocessing and field-construction workflow, and they show that resistance-aware priors are task-specific. They should be used mainly in Supplementary Results and Discussion to define evidence boundaries, not as a broad claim that anisoNET or resistance-aware interpolation always improves held-out prediction.
"""
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kidney = read_csv(KIDNEY_TABLE)
    liver = read_csv(LIVER_BOUNDARY)
    summary = pd.concat([kidney_rows(kidney), liver_rows(liver), sagittal_rows()], ignore_index=True)
    summary.to_csv(args.output_dir / "cross_tissue_evidence_summary_table.csv", index=False)
    (args.output_dir / "cross_tissue_evidence_summary_table.md").write_text(
        format_markdown_table(summary), encoding="utf-8"
    )
    write_interpretation(summary, args.output_dir / "cross_tissue_evidence_summary_interpretation.md")
    print(f"Wrote cross-tissue evidence summary to {args.output_dir}")


if __name__ == "__main__":
    main()
