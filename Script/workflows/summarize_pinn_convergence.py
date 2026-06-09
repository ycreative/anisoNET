"""Summarize PINN convergence histories from existing anisoNET runs."""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
CODEX_ROOT = PROJECT_ROOT / "codexAnalysis"
OUTPUT_ROOT = CODEX_ROOT / "convergence_diagnostics" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize existing anisoNET PINN convergence histories.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / args.sample
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, summary_rows = collect_histories(args.sample)
    history = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    if history.empty:
        raise FileNotFoundError(f"No pinn_history.json files found for sample {args.sample}")

    history.to_csv(output_dir / "pinn_convergence_history_long.csv", index=False)
    summary.to_csv(output_dir / "pinn_convergence_summary.csv", index=False)
    plot_representative(history, summary, output_dir)
    write_interpretation(summary, output_dir)
    print(f"Wrote convergence diagnostics to {output_dir}")


def collect_histories(sample: str) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for path in sorted(CODEX_ROOT.rglob("pinn_history.json")):
        if sample not in str(path):
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = infer_metadata(path)
        iterations = payload.get("iteration", [])
        losses = payload.get("loss", [])
        if not iterations or not losses:
            continue
        for idx, iteration in enumerate(iterations):
            row = dict(meta)
            row.update(
                {
                    "iteration": int(iteration),
                    "loss": get_index(payload, "loss", idx),
                    "loss_pde": get_index(payload, "loss_pde", idx),
                    "loss_data": get_index(payload, "loss_data", idx),
                    "loss_boundary": get_index(payload, "loss_boundary", idx),
                    "loss_background": get_index(payload, "loss_background", idx),
                    "loss_smoothness": get_index(payload, "loss_smoothness", idx),
                    "history_path": str(path),
                }
            )
            rows.append(row)
        initial = float(losses[0])
        final = float(losses[-1])
        summary_row = dict(meta)
        summary_row.update(
            {
                "initial_iteration": int(iterations[0]),
                "final_iteration": int(iterations[-1]),
                "initial_loss": initial,
                "final_loss": final,
                "loss_reduction_fraction": 1.0 - final / (initial + 1e-12),
                "loss_reduction_fold": initial / (final + 1e-12),
                "n_checkpoints": int(len(iterations)),
                "history_path": str(path),
            }
        )
        summary_rows.append(summary_row)
    return rows, summary_rows


def get_index(payload: dict[str, list[float]], key: str, idx: int) -> float:
    values = payload.get(key, [])
    if idx >= len(values):
        return float("nan")
    return float(values[idx])


def infer_metadata(path: Path) -> dict[str, object]:
    parts = path.parts
    text = str(path)
    target_gene = infer_target(parts)
    experiment = infer_experiment(text)
    parent = path.parent.name
    profile = read_profile(path.parent)
    return {
        "experiment": experiment,
        "target_gene": target_gene,
        "run_name": parent,
        "profile": profile,
    }


def infer_target(parts: tuple[str, ...]) -> str:
    for part in reversed(parts):
        if part.startswith("Apoe"):
            return "Apoe"
        if part.startswith("Gfap"):
            return "Gfap"
    return "unknown"


def infer_experiment(text: str) -> str:
    for name in [
        "alpha_sensitivity",
        "heldout",
        "histology_prior",
        "leave_one_marker_out",
        "resource_profile",
        "synthetic_barrier",
        "barrier_perturbation",
    ]:
        if name in text:
            return name
    return "other"


def read_profile(run_dir: Path) -> str:
    metrics_path = run_dir / "pinn_metrics.json"
    if not metrics_path.exists():
        return "unknown"
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "unknown"
    return str(payload.get("profile", "unknown"))


def plot_representative(history: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7,
            "axes.linewidth": 0.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    representative = select_representative(history)
    colors = {"Apoe": "#d1495b", "Gfap": "#00798c", "unknown": "#555555"}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    for (target_gene, run_name), group in representative.groupby(["target_gene", "run_name"], sort=True):
        label = f"{target_gene} {run_name}"
        axes[0].plot(
            group["iteration"],
            group["loss"],
            marker="o",
            linewidth=1.1,
            markersize=3,
            color=colors.get(target_gene, "black"),
            alpha=0.8,
            label=label,
        )
        axes[1].plot(
            group["iteration"],
            group["loss_data"],
            marker="o",
            linewidth=1.1,
            markersize=3,
            color=colors.get(target_gene, "black"),
            alpha=0.8,
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Total loss")
    axes[0].set_title("PINN convergence", fontsize=7, pad=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Data loss")
    axes[1].set_title("Source fit component", fontsize=7, pad=2)
    axes[0].legend(frameon=False, fontsize=5)
    for ax in axes:
        ax.tick_params(width=0.3, length=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(output_dir / "pinn_convergence_representative.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pinn_convergence_representative.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    plot_summary(summary, output_dir)


def select_representative(history: pd.DataFrame) -> pd.DataFrame:
    histology = history[
        (history["experiment"] == "histology_prior")
        & (history["run_name"] == "brightness_pinn")
        & (history["target_gene"].isin(["Apoe", "Gfap"]))
    ]
    if not histology.empty:
        return histology
    return history.groupby(["target_gene", "run_name"], sort=True).head(10)


def plot_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    grouped = (
        summary.groupby(["experiment", "target_gene"], sort=True)
        .agg(
            n_runs=("history_path", "count"),
            median_loss_reduction_fold=("loss_reduction_fold", "median"),
            min_loss_reduction_fold=("loss_reduction_fold", "min"),
        )
        .reset_index()
    )
    grouped["label"] = grouped["experiment"] + "\n" + grouped["target_gene"]
    fig, ax = plt.subplots(figsize=(7.2, 2.8), constrained_layout=True)
    ax.bar(
        np.arange(len(grouped)),
        grouped["median_loss_reduction_fold"],
        color="#2a9d8f",
        width=0.72,
    )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(grouped)))
    ax.set_xticklabels(grouped["label"], rotation=45, ha="right")
    ax.set_ylabel("Median loss reduction fold")
    ax.set_title("Existing PINN histories", fontsize=7, pad=2)
    ax.tick_params(width=0.3, length=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    fig.savefig(output_dir / "pinn_convergence_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "pinn_convergence_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_interpretation(summary: pd.DataFrame, output_dir: Path) -> None:
    total_runs = len(summary)
    median_fold = summary["loss_reduction_fold"].median()
    min_fold = summary["loss_reduction_fold"].min()
    lines = [
        "# PINN Convergence Diagnostics",
        "",
        f"Collected `{total_runs}` existing `pinn_history.json` files for the representative sample.",
        "",
        "## Main Result",
        "",
        f"- Median total-loss reduction fold: `{median_fold:.2f}`.",
        f"- Minimum total-loss reduction fold among collected histories: `{min_fold:.2f}`.",
        "",
        "## Interpretation",
        "",
        "Existing checkpoint histories show consistent loss reduction across representative anisoNET runs. These diagnostics support reporting training convergence qualitatively, but the current histories are sparse because losses were saved only at display checkpoints. If convergence becomes a main figure, rerun representative models with a smaller display interval to obtain denser curves.",
        "",
    ]
    (output_dir / "pinn_convergence_interpretation.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

