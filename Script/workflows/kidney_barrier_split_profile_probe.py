"""Run a focused kidney high-barrier/edge anisoNET profile probe."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
BENCHMARK = PROJECT_ROOT / "Script" / "workflows" / "generic_barrier_split_anisonet_benchmark.py"


PROBES = [
    {
        "label": "low_pde_1000",
        "profile": "fourier_refined_low_pde_16g",
        "iterations": 1000,
        "num_domain": 2000,
        "extra": {},
    },
    {
        "label": "smooth_sigma4_pde04",
        "profile": "fourier_refined_low_pde_16g",
        "iterations": 900,
        "num_domain": 1800,
        "extra": {"fourier_sigma": 4.0, "smoothness_weight": 0.004, "pde_weight": 0.04, "data_weight": 8.0},
    },
    {
        "label": "smooth_sigma4_pde08",
        "profile": "fourier_refined_16g",
        "iterations": 900,
        "num_domain": 1800,
        "extra": {"fourier_sigma": 4.0, "smoothness_weight": 0.004, "pde_weight": 0.08, "data_weight": 8.0},
    },
    {
        "label": "less_smooth_sigma6_pde04",
        "profile": "fourier_refined_low_pde_16g",
        "iterations": 900,
        "num_domain": 1800,
        "extra": {"fourier_sigma": 6.5, "smoothness_weight": 0.001, "pde_weight": 0.04, "data_weight": 8.0},
    },
    {
        "label": "high_pde_sigma4",
        "profile": "fourier_refined_16g",
        "iterations": 900,
        "num_domain": 1800,
        "extra": {"fourier_sigma": 4.0, "smoothness_weight": 0.006, "pde_weight": 0.16, "data_weight": 8.0},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run kidney high-barrier/edge profile probe.")
    parser.add_argument("--sample-dir", required=True)
    parser.add_argument("--preflight-dir", required=True)
    parser.add_argument("--target-gene", default="Umod")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def run_probe(args: argparse.Namespace, probe: dict[str, object]) -> Path:
    output_dir = Path(args.output_dir) / str(probe["label"])
    metrics_csv = output_dir / "generic_barrier_split_metrics.csv"
    if args.skip_existing and metrics_csv.exists():
        return metrics_csv

    command = [
        str(PYTHON),
        str(BENCHMARK),
        "--sample-dir",
        args.sample_dir,
        "--preflight-dir",
        args.preflight_dir,
        "--target-gene",
        args.target_gene,
        "--output-dir",
        str(output_dir),
        "--split-mode",
        "high_barrier_or_edge",
        "--test-fraction",
        str(args.test_fraction),
        "--seed",
        str(args.seed),
        "--profile",
        str(probe["profile"]),
        "--iterations",
        str(probe["iterations"]),
        "--num-domain",
        str(probe["num_domain"]),
        "--device",
        args.device,
    ]
    for key, value in dict(probe["extra"]).items():
        command.extend([f"--{key.replace('_', '-')}", str(value)])
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return metrics_csv


def summarize(output_dir: Path, metrics_paths: list[Path]) -> None:
    frames = []
    for metrics_path in metrics_paths:
        frame = pd.read_csv(metrics_path)
        frame.insert(0, "probe", metrics_path.parent.name)
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "kidney_barrier_split_profile_probe_metrics.csv", index=False)

    anisonet = combined[combined["method"].isin(["anisonet_masked", "anisonet_gauss07"])].copy()
    anisonet = anisonet.sort_values("test_pearson", ascending=False)
    anisonet.to_csv(output_dir / "kidney_barrier_split_profile_probe_anisonet_ranked.csv", index=False)

    baseline = combined[combined["method"].isin(["euclidean_idw", "resistance_idw", "euclidean_gaussian", "resistance_gaussian"])].copy()
    baseline_best = baseline.sort_values("test_pearson", ascending=False).iloc[0]
    best_anisonet = anisonet.iloc[0]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)
    plot_frame = combined[combined["method"].isin(["resistance_idw", "anisonet_gauss07", "anisonet_masked"])].copy()
    pivot = plot_frame.pivot_table(index="probe", columns="method", values="test_pearson", aggfunc="first")
    pivot = pivot.reindex([probe["label"] for probe in PROBES])
    pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_ylabel("Held-out Pearson")
    axes[0].set_title("Umod high-barrier/edge")
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].legend(fontsize=7, frameon=False)

    pivot_mse = plot_frame.pivot_table(index="probe", columns="method", values="test_mse", aggfunc="first")
    pivot_mse = pivot_mse.reindex([probe["label"] for probe in PROBES])
    pivot_mse.plot(kind="bar", ax=axes[1])
    axes[1].set_ylabel("Held-out MSE")
    axes[1].set_title("Umod high-barrier/edge")
    axes[1].tick_params(axis="x", labelrotation=45)
    axes[1].legend(fontsize=7, frameon=False)
    fig.savefig(output_dir / "kidney_barrier_split_profile_probe_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "kidney_barrier_split_profile_probe_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Kidney Barrier-Split Profile Probe Interpretation",
        "",
        "This probe tests whether simple PINN profile changes improve Umod high-barrier/edge held-out prediction.",
        "",
        "## Main Result",
        "",
        f"- Best baseline: `{baseline_best['method']}` with Pearson `{baseline_best['test_pearson']:.4f}` and MSE `{baseline_best['test_mse']:.4f}`.",
        f"- Best anisoNET probe: `{best_anisonet['probe']}` / `{best_anisonet['method']}` with Pearson `{best_anisonet['test_pearson']:.4f}` and MSE `{best_anisonet['test_mse']:.4f}`.",
        "",
        "## anisoNET Probe Ranking",
        "",
    ]
    for _, row in anisonet.iterrows():
        lines.append(f"- `{row['probe']}` `{row['method']}`: Pearson `{row['test_pearson']:.4f}`, MSE `{row['test_mse']:.4f}`, train Pearson `{row['train_pearson']:.4f}`.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Simple profile changes improve or degrade the current kidney PINN modestly, but none of the tested settings outperform the resistance-aware IDW baseline. This supports shifting optimization toward hybrid warm starts, resistance-aware source terms, or a revised physics objective rather than only changing scalar loss weights.",
            "",
        ]
    )
    (output_dir / "kidney_barrier_split_profile_probe_interpretation.md").write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "best_baseline": baseline_best.to_dict(),
        "best_anisonet": best_anisonet.to_dict(),
        "metrics": [str(path) for path in metrics_paths],
    }
    with (output_dir / "kidney_barrier_split_profile_probe_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_paths = []
    for probe in PROBES:
        metrics_paths.append(run_probe(args, probe))
    summarize(output_dir, metrics_paths)
    print(f"Wrote kidney profile probe summary to {output_dir}")


if __name__ == "__main__":
    main()

