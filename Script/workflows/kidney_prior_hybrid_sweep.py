"""Sweep mouse kidney prior-hybrid anisoNET settings."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
BENCHMARK = PROJECT_ROOT / "Script" / "workflows" / "generic_barrier_split_anisonet_benchmark.py"


TARGETS = {
    "Umod": "Umod_proximal_barrier",
    "Slc34a1": "Slc34a1_TAL_CD_barrier",
}


def parse_csv_numbers(text: str, cast):
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run kidney prior-hybrid sweep.")
    parser.add_argument(
        "--sample-dir",
        default="codexAnalysis/processed_visium/mouse_kidney_10x/V1_Mouse_Kidney",
    )
    parser.add_argument(
        "--preflight-root",
        default="codexAnalysis/cross_tissue/mouse_kidney_10x/V1_Mouse_Kidney/preflight",
    )
    parser.add_argument(
        "--output-root",
        default="codexAnalysis/barrier_split_anisonet/mouse_kidney_10x/V1_Mouse_Kidney/prior_hybrid_sweep",
    )
    parser.add_argument("--targets", default="Umod,Slc34a1")
    parser.add_argument("--prior-weights", default="5,10,20,40")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--num-domain", type=int, default=1800)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_one(
    *,
    args: argparse.Namespace,
    target: str,
    prior_weight: float,
    seed: int,
) -> Path:
    preflight_dir = Path(args.preflight_root) / TARGETS[target]
    output_dir = Path(args.output_root) / target / f"priorw{prior_weight:g}_seed{seed}"
    metrics_csv = output_dir / "generic_barrier_split_metrics.csv"
    if args.skip_existing and metrics_csv.exists():
        return metrics_csv

    command = [
        str(PYTHON),
        str(BENCHMARK),
        "--sample-dir",
        args.sample_dir,
        "--preflight-dir",
        str(preflight_dir),
        "--target-gene",
        target,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(seed),
        "--split-mode",
        "high_barrier_or_edge",
        "--profile",
        "fourier_refined_low_pde_16g",
        "--iterations",
        str(args.iterations),
        "--num-domain",
        str(args.num_domain),
        "--device",
        args.device,
        "--fourier-sigma",
        "4.0",
        "--smoothness-weight",
        "0.004",
        "--pde-weight",
        "0.04",
        "--data-weight",
        "8.0",
        "--prior-field",
        "line_resistance_idw",
        "--prior-weight",
        str(prior_weight),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return metrics_csv


def summarize(output_root: Path, metrics_paths: list[Path]) -> None:
    frames = []
    for path in metrics_paths:
        target = path.parents[1].name
        run_name = path.parent.name
        weight_text, seed_text = run_name.replace("priorw", "").split("_seed")
        frame = pd.read_csv(path)
        if "prior_weight" in frame.columns:
            frame = frame.rename(columns={"prior_weight": "row_prior_weight"})
        frame.insert(0, "target_gene", target)
        frame.insert(1, "prior_weight", float(weight_text))
        frame.insert(2, "seed", int(seed_text))
        frame.insert(3, "source_csv", str(path))
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_root / "kidney_prior_hybrid_sweep_metrics.csv", index=False)

    key_methods = [
        "euclidean_idw",
        "resistance_idw",
        "prior_line_resistance_idw_grid",
        "prior_line_resistance_idw_grid_traincal",
        "anisonet_prior_gauss07",
        "anisonet_prior_gauss07_traincal",
        "anisonet_prior_masked",
        "anisonet_prior_masked_traincal",
    ]
    key = combined[combined["method"].isin(key_methods)].copy()
    key.to_csv(output_root / "kidney_prior_hybrid_sweep_key_metrics.csv", index=False)

    summaries = []
    for (target, method, weight), group in key.groupby(["target_gene", "method", "prior_weight"], sort=True):
        summaries.append(
            {
                "target_gene": target,
                "method": method,
                "prior_weight": weight,
                "n": int(group.shape[0]),
                "test_pearson_mean": float(group["test_pearson"].mean()),
                "test_pearson_sd": float(group["test_pearson"].std(ddof=1)) if group.shape[0] > 1 else 0.0,
                "test_mse_mean": float(group["test_mse"].mean()),
                "test_mse_sd": float(group["test_mse"].std(ddof=1)) if group.shape[0] > 1 else 0.0,
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_root / "kidney_prior_hybrid_sweep_summary.csv", index=False)

    baseline = (
        key[key["method"] == "resistance_idw"][["target_gene", "seed", "test_pearson", "test_mse"]]
        .drop_duplicates(["target_gene", "seed"])
        .rename(columns={"test_pearson": "baseline_pearson", "test_mse": "baseline_mse"})
    )
    hybrid = key[key["method"] == "anisonet_prior_gauss07_traincal"].merge(
        baseline,
        on=["target_gene", "seed"],
        how="left",
    )
    hybrid["pearson_delta_vs_resistance_idw"] = hybrid["test_pearson"] - hybrid["baseline_pearson"]
    hybrid["mse_delta_vs_resistance_idw"] = hybrid["test_mse"] - hybrid["baseline_mse"]
    hybrid.to_csv(output_root / "kidney_prior_hybrid_sweep_hybrid_deltas.csv", index=False)

    delta_summary = (
        hybrid.groupby(["target_gene", "prior_weight"], as_index=False)
        .agg(
            n=("pearson_delta_vs_resistance_idw", "size"),
            pearson_delta_mean=("pearson_delta_vs_resistance_idw", "mean"),
            pearson_delta_sd=("pearson_delta_vs_resistance_idw", "std"),
            pearson_delta_n_positive=("pearson_delta_vs_resistance_idw", lambda x: int((x > 0).sum())),
            mse_delta_mean=("mse_delta_vs_resistance_idw", "mean"),
            mse_delta_sd=("mse_delta_vs_resistance_idw", "std"),
            mse_delta_n_negative=("mse_delta_vs_resistance_idw", lambda x: int((x < 0).sum())),
        )
    )
    delta_summary.to_csv(output_root / "kidney_prior_hybrid_sweep_delta_summary.csv", index=False)

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0], "pearson_delta_vs_resistance_idw", "Delta Pearson vs resistance-IDW"),
        (axes[1], "mse_delta_vs_resistance_idw", "Delta MSE vs resistance-IDW"),
    ]:
        for target in sorted(hybrid["target_gene"].unique()):
            subset = hybrid[hybrid["target_gene"] == target]
            grouped = subset.groupby("prior_weight")[metric]
            means = grouped.mean()
            sds = grouped.std(ddof=1).fillna(0.0)
            ax.errorbar(means.index, means.values, yerr=sds.values, marker="o", capsize=3, label=target)
        ax.axhline(0.0, color="0.25", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks(sorted(hybrid["prior_weight"].unique()))
        ax.set_xticklabels([f"{value:g}" for value in sorted(hybrid["prior_weight"].unique())])
        ax.set_xlabel("Prior weight")
        ax.set_ylabel(ylabel)
    axes[0].legend(frameon=False)
    fig.savefig(output_root / "kidney_prior_hybrid_sweep_delta_summary.pdf", bbox_inches="tight")
    fig.savefig(output_root / "kidney_prior_hybrid_sweep_delta_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    best_rows = []
    for target in sorted(hybrid["target_gene"].unique()):
        target_summary = delta_summary[delta_summary["target_gene"] == target].copy()
        best = target_summary.sort_values(["pearson_delta_mean", "mse_delta_mean"], ascending=[False, True]).iloc[0]
        best_rows.append(best.to_dict())

    prior_anchor_rows = []
    for target in sorted(key["target_gene"].unique()):
        target_key = key[key["target_gene"] == target]
        ridw = target_key[target_key["method"] == "resistance_idw"].drop_duplicates(["target_gene", "seed"])
        prior = target_key[target_key["method"] == "prior_line_resistance_idw_grid"].drop_duplicates(["target_gene", "seed"])
        if ridw.empty or prior.empty:
            continue
        prior_anchor_rows.append(
            {
                "target_gene": target,
                "resistance_idw_pearson_mean": float(ridw["test_pearson"].mean()),
                "resistance_idw_mse_mean": float(ridw["test_mse"].mean()),
                "prior_grid_pearson_mean": float(prior["test_pearson"].mean()),
                "prior_grid_mse_mean": float(prior["test_mse"].mean()),
            }
        )

    lines = [
        "# Mouse Kidney Prior-Hybrid Sweep Interpretation",
        "",
        "This sweep evaluates prior-hybrid anisoNET stability across prior weights and random seeds.",
        "",
        "## Continuous Prior Anchor",
        "",
    ]
    for row in prior_anchor_rows:
        lines.append(
            f"- `{row['target_gene']}` continuous line-resistance prior grid: Pearson "
            f"`{row['prior_grid_pearson_mean']:.4f}` vs resistance-IDW "
            f"`{row['resistance_idw_pearson_mean']:.4f}`; MSE "
            f"`{row['prior_grid_mse_mean']:.5f}` vs resistance-IDW "
            f"`{row['resistance_idw_mse_mean']:.5f}`."
        )
    lines.extend(
        [
            "",
        "## Best Prior Weights",
        "",
        ]
    )
    for row in best_rows:
        lines.append(
            f"- `{row['target_gene']}` best by mean Pearson delta: prior weight `{row['prior_weight']:g}`, "
            f"mean delta `{row['pearson_delta_mean']:+.4f}` "
            f"({int(row['pearson_delta_n_positive'])}/{int(row['n'])} seeds positive), "
            f"mean MSE delta `{row['mse_delta_mean']:+.4f}`."
        )
    lines.extend(["", "## Delta Summary", ""])
    for _, row in delta_summary.sort_values(["target_gene", "prior_weight"]).iterrows():
        lines.append(
            f"- `{row.target_gene}` prior weight `{row.prior_weight:g}`: "
            f"Pearson delta `{row.pearson_delta_mean:+.4f}` +/- `{row.pearson_delta_sd:.4f}`, "
            f"{int(row.pearson_delta_n_positive)}/{int(row.n)} positive; "
            f"MSE delta `{row.mse_delta_mean:+.4f}` +/- `{row.mse_delta_sd:.4f}`, "
            f"{int(row.mse_delta_n_negative)}/{int(row.n)} lower MSE."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use these results to decide whether prior-hybrid anisoNET is stable enough for manuscript evidence. A positive Pearson delta with non-increasing MSE across seeds supports promoting the hybrid result; mixed seed behavior should remain optimization evidence only.",
            "",
        ]
    )
    (output_root / "kidney_prior_hybrid_sweep_interpretation.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "metrics": [str(path) for path in metrics_paths],
        "best_rows": best_rows,
    }
    with (output_root / "kidney_prior_hybrid_sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    targets = [target for target in parse_csv_numbers(args.targets, str) if target]
    weights = parse_csv_numbers(args.prior_weights, float)
    seeds = parse_csv_numbers(args.seeds, int)

    metrics_paths = []
    for target in targets:
        if target not in TARGETS:
            raise ValueError(f"Unknown target '{target}'. Available: {', '.join(TARGETS)}")
        for prior_weight in weights:
            for seed in seeds:
                metrics_paths.append(run_one(args=args, target=target, prior_weight=prior_weight, seed=seed))
    summarize(output_root, metrics_paths)
    print(f"Wrote kidney prior-hybrid sweep to {output_root}")


if __name__ == "__main__":
    main()
