"""Summarize anisoNET resource profiling outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "resource_profile" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize anisoNET resource profile.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--output-suffix", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = OUTPUT_ROOT / args.sample
    input_csv = Path(args.input_csv) if args.input_csv else output_dir / "anisonet_resource_profile.csv"
    suffix = sanitize_suffix(args.output_suffix)
    frame = pd.read_csv(input_csv)
    plot_summary(frame, output_dir, suffix=suffix)
    write_table(frame, output_dir, suffix=suffix)
    print(f"Wrote resource profile summary to {output_dir}")


def sanitize_suffix(value: str) -> str:
    if not value:
        return ""
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value.strip())
    return f"_{safe}" if safe else ""


def plot_summary(frame: pd.DataFrame, output_dir: Path, *, suffix: str = "") -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.4), constrained_layout=True)
    colors = ["#2a9d8f", "#e76f51", "#457b9d", "#8d99ae"]
    labels = frame["target_gene"].tolist()

    axes[0].bar(labels, frame["elapsed_seconds"], color=colors[: len(frame)], width=0.65)
    axes[0].set_ylabel("Runtime (seconds)")
    axes[0].set_title("PINN runtime", fontsize=7, pad=2)

    axes[1].bar(labels, frame["peak_cuda_reserved_gb"], color=colors[: len(frame)], width=0.65)
    total = float(frame["gpu_total_memory_gb"].iloc[0])
    axes[1].axhline(total, color="black", linewidth=0.7, linestyle="--", label="GPU total")
    axes[1].set_ylabel("CUDA reserved memory (GB)")
    axes[1].set_title("Peak GPU memory", fontsize=7, pad=2)
    axes[1].legend(frameon=False, fontsize=6)

    for ax in axes:
        ax.tick_params(width=0.3, length=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(output_dir / f"anisonet_resource_profile_summary{suffix}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"anisonet_resource_profile_summary{suffix}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def write_table(frame: pd.DataFrame, output_dir: Path, *, suffix: str = "") -> None:
    columns = [
        "target_gene",
        "profile",
        "n_spots",
        "grid_size",
        "num_domain",
        "iterations",
        "elapsed_seconds",
        "elapsed_minutes",
        "peak_cuda_allocated_gb",
        "peak_cuda_reserved_gb",
        "reserved_fraction_of_gpu",
    ]
    table = frame[columns].copy()
    table.to_csv(output_dir / f"anisonet_resource_profile_table{suffix}.csv", index=False)


if __name__ == "__main__":
    main()
