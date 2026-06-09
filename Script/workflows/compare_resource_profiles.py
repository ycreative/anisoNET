"""Compare resource profiles for multiple anisoNET PINN profiles."""

from __future__ import annotations

import os

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RESOURCE_ROOT = PROJECT_ROOT / "codexAnalysis" / "resource_profile" / "brain_aging_gse193107"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare anisoNET resource profile tables.")
    parser.add_argument("--sample", default="GSM5773457_Old_mouse_brain_A1-2")
    parser.add_argument(
        "--tables",
        nargs="+",
        default=[
            "anisonet_resource_profile_table.csv",
            "anisonet_resource_profile_table_low_pde_candidate.csv",
        ],
    )
    parser.add_argument("--output-prefix", default="anisonet_resource_profile_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = RESOURCE_ROOT / args.sample
    frames = []
    for table in args.tables:
        frames.append(pd.read_csv(output_dir / table))
    frame = pd.concat(frames, ignore_index=True)
    comparison = summarize(frame)
    comparison.to_csv(output_dir / f"{args.output_prefix}.csv", index=False)
    write_markdown(comparison, output_dir / f"{args.output_prefix}.md")
    print(f"Wrote resource profile comparison to {output_dir}", flush=True)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile, group in frame.groupby("profile", sort=True):
        rows.append(
            {
                "profile": profile,
                "n_targets": int(len(group)),
                "mean_elapsed_seconds": group["elapsed_seconds"].mean(),
                "max_elapsed_seconds": group["elapsed_seconds"].max(),
                "mean_peak_cuda_reserved_gb": group["peak_cuda_reserved_gb"].mean(),
                "max_peak_cuda_reserved_gb": group["peak_cuda_reserved_gb"].max(),
                "mean_reserved_fraction_of_gpu": group["reserved_fraction_of_gpu"].mean(),
                "n_spots": int(group["n_spots"].iloc[0]),
                "grid_size": int(group["grid_size"].iloc[0]),
                "num_domain": int(group["num_domain"].iloc[0]),
                "iterations": int(group["iterations"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(comparison: pd.DataFrame, path: Path) -> None:
    lines = [
        "# anisoNET Resource Profile Comparison",
        "",
        "| Profile | Targets | Mean runtime (s) | Max runtime (s) | Max CUDA reserved (GB) | GPU fraction |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            "| "
            f"`{row.profile}` | {row.n_targets} | "
            f"{row.mean_elapsed_seconds:.1f} | {row.max_elapsed_seconds:.1f} | "
            f"{row.max_peak_cuda_reserved_gb:.3f} | {row.mean_reserved_fraction_of_gpu:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The conservative default and low-PDE candidate profiles have matched architecture, collocation count, and iteration count, so their runtime and memory footprints are effectively the same on the representative Visium-scale section. The low-PDE profile changes the loss balance rather than computational scale.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

