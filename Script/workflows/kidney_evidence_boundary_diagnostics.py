"""Summarize mouse kidney evidence boundaries from existing result tables.

This workflow does not retrain or rerun anisoNET. It consolidates the current
mouse-kidney marker screen, high-barrier IDW follow-up, prior-hybrid sweep, and
grid-geodesic prior results into manuscript-facing diagnostic tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_ROOT = Path("codexAnalysis")
KIDNEY_ROOT = DEFAULT_ROOT / "barrier_split_anisonet" / "mouse_kidney_10x" / "V1_Mouse_Kidney"
MARKER_ROOT = DEFAULT_ROOT / "cross_tissue" / "mouse_kidney_10x" / "V1_Mouse_Kidney" / "marker_task_screen"
FOLLOWUP_ROOT = (
    DEFAULT_ROOT
    / "barrier_aware_interpolation"
    / "mouse_kidney_10x"
    / "V1_Mouse_Kidney"
    / "summary_marker_screen_followup"
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path)


def first_number(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[0])


def summarize_main_methods(combined: pd.DataFrame, grid_geo: pd.DataFrame, prior_delta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in ["Slc34a1", "Umod"]:
        sub = combined[combined["target_gene"] == target].copy()
        baseline = sub[sub["method"] == "resistance_idw"]
        baseline = baseline.sort_values(["profile"]).head(1)
        baseline_pearson = first_number(baseline["test_pearson"])
        baseline_mse = first_number(baseline["test_mse"])

        geodesic = grid_geo[(grid_geo["target_gene"] == target) & (grid_geo["method"] == "grid_geodesic_idw_prior")]
        line_prior = grid_geo[(grid_geo["target_gene"] == target) & (grid_geo["method"] == "line_resistance_idw_spot")]

        anisonet_candidates = sub[
            sub["field_type"].isin(["continuous_prior", "anisonet"])
            & (
                sub["method"].astype(str).str.contains("prior", case=False, na=False)
                | sub["method"].astype(str).str.contains("anisonet", case=False, na=False)
                | sub["prior_field"].notna()
            )
        ].copy()
        if anisonet_candidates.empty:
            best_hybrid = pd.DataFrame()
        else:
            best_hybrid = anisonet_candidates.sort_values("test_pearson", ascending=False).head(1)

        delta_sub = prior_delta[prior_delta["target_gene"] == target].copy()
        if delta_sub.empty:
            best_weight = pd.DataFrame()
        else:
            best_weight = delta_sub.sort_values("pearson_delta_mean", ascending=False).head(1)

        rows.append(
            {
                "target_gene": target,
                "resistance_idw_pearson": baseline_pearson,
                "resistance_idw_mse": baseline_mse,
                "line_prior_pearson": first_number(line_prior["test_pearson"]),
                "line_prior_mse": first_number(line_prior["test_mse"]),
                "grid_geodesic_pearson": first_number(geodesic["test_pearson"]),
                "grid_geodesic_mse": first_number(geodesic["test_mse"]),
                "best_hybrid_method": "" if best_hybrid.empty else str(best_hybrid["method"].iloc[0]),
                "best_hybrid_profile": "" if best_hybrid.empty else str(best_hybrid["profile"].iloc[0]),
                "best_hybrid_calibration": "" if best_hybrid.empty else str(best_hybrid["calibration"].iloc[0]),
                "best_hybrid_prior_weight": np.nan if best_hybrid.empty else first_number(best_hybrid["prior_weight"]),
                "best_hybrid_pearson": np.nan if best_hybrid.empty else first_number(best_hybrid["test_pearson"]),
                "best_hybrid_mse": np.nan if best_hybrid.empty else first_number(best_hybrid["test_mse"]),
                "best_prior_weight_by_mean_pearson_delta": np.nan
                if best_weight.empty
                else first_number(best_weight["prior_weight"]),
                "best_prior_weight_mean_pearson_delta": np.nan
                if best_weight.empty
                else first_number(best_weight["pearson_delta_mean"]),
                "best_prior_weight_mean_mse_delta": np.nan if best_weight.empty else first_number(best_weight["mse_delta_mean"]),
                "positive_seed_count_at_best_weight": np.nan
                if best_weight.empty
                else first_number(best_weight["pearson_delta_n_positive"]),
            }
        )

    return pd.DataFrame(rows)


def summarize_marker_tasks(marker: pd.DataFrame, followup: pd.DataFrame) -> pd.DataFrame:
    task_names = {
        "Slc12a3": "Slc12a3_TAL_barrier",
        "Calb1": "Calb1_TAL_barrier",
        "Aqp2": "Aqp2_proximal_barrier",
    }
    rows: list[dict[str, object]] = []
    for target, task in task_names.items():
        marker_sub = marker[(marker["target_gene"] == target) & (marker["recommended"] == True)].copy()  # noqa: E712
        if task.endswith("TAL_barrier"):
            marker_sub = marker_sub[marker_sub["barrier_compartment"] == "tal"]
        elif task.endswith("proximal_barrier"):
            marker_sub = marker_sub[marker_sub["barrier_compartment"] == "proximal_tubule"]
        marker_sub = marker_sub.sort_values("task_score", ascending=False).head(1)

        follow_sub = followup[(followup["target_gene"] == target) & (followup["split_mode"] == "high_barrier_or_edge")]

        rows.append(
            {
                "task": task,
                "target_gene": target,
                "source_compartment": "" if marker_sub.empty else str(marker_sub["source_compartment"].iloc[0]),
                "barrier_compartment": "" if marker_sub.empty else str(marker_sub["barrier_compartment"].iloc[0]),
                "task_score": np.nan if marker_sub.empty else first_number(marker_sub["task_score"]),
                "target_p95": np.nan if marker_sub.empty else first_number(marker_sub["target_p95"]),
                "source_barrier_score_corr": np.nan
                if marker_sub.empty
                else first_number(marker_sub["source_barrier_score_corr"]),
                "top20_overlap_fraction": np.nan if marker_sub.empty else first_number(marker_sub["top20_overlap_fraction"]),
                "barrier_leakage_ratio_in_target_expression": np.nan
                if marker_sub.empty
                else first_number(marker_sub["barrier_leakage_ratio_in_target_expression"]),
                "high_barrier_idw_pearson_delta": np.nan
                if follow_sub.empty
                else first_number(follow_sub["pearson_delta_resistance_minus_euclidean"]),
                "high_barrier_idw_mse_delta": np.nan
                if follow_sub.empty
                else first_number(follow_sub["mse_delta_resistance_minus_euclidean"]),
            }
        )
    return pd.DataFrame(rows)


def assign_claim_level(row: pd.Series) -> tuple[str, str]:
    target = row["target_gene"]
    if target == "Umod":
        return (
            "method-development",
            "Continuous resistance-aware prior improves Umod; hybrid PINN improves Pearson at selected weights but MSE is unstable.",
        )
    if target == "Slc34a1":
        return (
            "strong baseline / claim-boundary",
            "Resistance-IDW is already very strong; hybrid and geodesic variants do not justify a superiority claim.",
        )
    return (
        "supplementary",
        "New screened task has only a small or negative high-barrier/edge resistance-IDW effect.",
    )


def write_interpretation(main: pd.DataFrame, marker: pd.DataFrame, out_path: Path) -> None:
    umod = main[main["target_gene"] == "Umod"].iloc[0]
    slc = main[main["target_gene"] == "Slc34a1"].iloc[0]
    slc12a3 = marker[marker["target_gene"] == "Slc12a3"].iloc[0]

    text = f"""# Mouse Kidney Evidence Boundary Diagnostics

Date: 2026-06-08

## Purpose

This diagnostic consolidates existing kidney outputs after the systematic marker screen. It does not rerun PINN training or interpolation. The goal is to separate manuscript-grade claims from method-development signals.

## Main Kidney Tasks

| Target | Resistance-IDW Pearson | Resistance-IDW MSE | Best prior/hybrid Pearson | Best prior/hybrid MSE | Interpretation |
|---|---:|---:|---:|---:|---|
| `Umod` | {umod['resistance_idw_pearson']:.4f} | {umod['resistance_idw_mse']:.5f} | {umod['best_hybrid_pearson']:.4f} | {umod['best_hybrid_mse']:.5f} | method-development signal |
| `Slc34a1` | {slc['resistance_idw_pearson']:.4f} | {slc['resistance_idw_mse']:.5f} | {slc['best_hybrid_pearson']:.4f} | {slc['best_hybrid_mse']:.5f} | strong baseline / claim boundary |

For `Umod`, the line-resistance prior remains the cleanest amplitude-preserving improvement, while the PINN hybrid mainly improves ranking/Pearson. For `Slc34a1`, resistance-IDW is already near ceiling and remains the strongest benchmarked readout.

## Grid-Geodesic Prior Check

- `Umod`: grid-geodesic prior Pearson {umod['grid_geodesic_pearson']:.4f}, MSE {umod['grid_geodesic_mse']:.5f}; line-resistance prior Pearson {umod['line_prior_pearson']:.4f}, MSE {umod['line_prior_mse']:.5f}.
- `Slc34a1`: grid-geodesic prior Pearson {slc['grid_geodesic_pearson']:.4f}, MSE {slc['grid_geodesic_mse']:.5f}; line-resistance prior Pearson {slc['line_prior_pearson']:.4f}, MSE {slc['line_prior_mse']:.5f}.

The grid-geodesic prior is therefore mixed: it slightly helps `Slc34a1` but underperforms the line-resistance prior for `Umod`.

## New Marker-Screen Candidate

`Slc12a3_TAL_barrier` remains the best new supplementary task from the marker screen:

- task score {slc12a3['task_score']:.3f};
- high-barrier/edge resistance-IDW Pearson delta {slc12a3['high_barrier_idw_pearson_delta']:+.4f};
- high-barrier/edge MSE delta {slc12a3['high_barrier_idw_mse_delta']:+.4f}.

This is directionally favorable but too small for a main performance claim.

## Manuscript Implication

Kidney should be framed as systematic cross-tissue benchmark-development evidence:

1. The marker screen reduces ad hoc target-selection risk.
2. `Slc34a1` shows that resistance-aware IDW can be a very strong anatomy-aware baseline.
3. `Umod` shows that continuous resistance-aware priors can improve difficult kidney tasks, but current PINN hybrids still need better amplitude preservation.
4. `Slc12a3` is a supplementary candidate only.

Do not promote kidney as a broad anisoNET superiority result under the current scalar PINN objective.
"""
    out_path.write_text(text, encoding="utf-8")


def build_supplementary_summary_table(main: pd.DataFrame, marker: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for _, row in main.iterrows():
        rows.append(
            {
                "evidence_block": "primary_kidney_task",
                "task_or_target": row["target_gene"],
                "comparison": "resistance-IDW vs best prior/hybrid",
                "primary_metric": "held-out Pearson",
                "baseline_value": row["resistance_idw_pearson"],
                "candidate_value": row["best_hybrid_pearson"],
                "secondary_metric": "held-out MSE",
                "baseline_secondary_value": row["resistance_idw_mse"],
                "candidate_secondary_value": row["best_hybrid_mse"],
                "claim_level": row["claim_level"],
                "interpretation": row["claim_reason"],
            }
        )
        rows.append(
            {
                "evidence_block": "prior_geometry_check",
                "task_or_target": row["target_gene"],
                "comparison": "line-resistance prior vs grid-geodesic prior",
                "primary_metric": "held-out Pearson",
                "baseline_value": row["line_prior_pearson"],
                "candidate_value": row["grid_geodesic_pearson"],
                "secondary_metric": "held-out MSE",
                "baseline_secondary_value": row["line_prior_mse"],
                "candidate_secondary_value": row["grid_geodesic_mse"],
                "claim_level": "method diagnostic",
                "interpretation": "Grid-geodesic prior is mixed and is not a drop-in replacement for line resistance.",
            }
        )

    for _, row in marker.iterrows():
        rows.append(
            {
                "evidence_block": "marker_screen_followup",
                "task_or_target": row["task"],
                "comparison": "resistance-IDW minus Euclidean-IDW in high-barrier/edge split",
                "primary_metric": "Pearson delta",
                "baseline_value": 0.0,
                "candidate_value": row["high_barrier_idw_pearson_delta"],
                "secondary_metric": "MSE delta",
                "baseline_secondary_value": 0.0,
                "candidate_secondary_value": row["high_barrier_idw_mse_delta"],
                "claim_level": row["claim_level"],
                "interpretation": row["claim_reason"],
            }
        )
    return pd.DataFrame(rows)


def write_markdown_table(table: pd.DataFrame, out_path: Path) -> None:
    display = table.copy()
    numeric_cols = [
        "baseline_value",
        "candidate_value",
        "baseline_secondary_value",
        "candidate_secondary_value",
    ]
    for col in numeric_cols:
        display[col] = pd.to_numeric(display[col], errors="coerce").map(lambda x: "" if pd.isna(x) else f"{x:.5f}")
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        values = [str(row[col]).replace("|", "/") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=KIDNEY_ROOT / "evidence_boundary_diagnostics",
        help="Directory for consolidated kidney diagnostic outputs.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combined = read_csv(KIDNEY_ROOT / "summary_with_prior" / "generic_barrier_split_combined_metrics.csv")
    grid_geo = read_csv(KIDNEY_ROOT / "grid_geodesic_prior" / "grid_geodesic_prior_combined_metrics.csv")
    prior_delta = read_csv(KIDNEY_ROOT / "prior_hybrid_sweep" / "kidney_prior_hybrid_sweep_delta_summary.csv")
    marker = read_csv(MARKER_ROOT / "kidney_marker_task_screen.csv")
    followup = read_csv(FOLLOWUP_ROOT / "kidney_marker_screen_followup_idw_deltas.csv")

    main_summary = summarize_main_methods(combined, grid_geo, prior_delta)
    claim_info = main_summary.apply(assign_claim_level, axis=1, result_type="expand")
    main_summary["claim_level"] = claim_info[0]
    main_summary["claim_reason"] = claim_info[1]

    marker_summary = summarize_marker_tasks(marker, followup)
    marker_summary["claim_level"] = "supplementary"
    marker_summary["claim_reason"] = np.where(
        marker_summary["target_gene"] == "Slc12a3",
        "Best new marker-screen candidate, but effect size is small.",
        "Marker-screen ranking did not translate into high-barrier/edge improvement.",
    )

    main_summary.to_csv(args.output_dir / "kidney_main_task_evidence_boundary.csv", index=False)
    marker_summary.to_csv(args.output_dir / "kidney_marker_screen_evidence_boundary.csv", index=False)
    supp_table = build_supplementary_summary_table(main_summary, marker_summary)
    supp_table.to_csv(args.output_dir / "kidney_supplementary_summary_table.csv", index=False)
    write_markdown_table(supp_table, args.output_dir / "kidney_supplementary_summary_table.md")
    write_interpretation(main_summary, marker_summary, args.output_dir / "kidney_evidence_boundary_interpretation.md")

    print(f"Wrote kidney evidence-boundary diagnostics to {args.output_dir}")


if __name__ == "__main__":
    main()
