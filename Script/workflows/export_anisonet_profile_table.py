"""Export anisoNET PINN profile parameter tables for manuscript reporting."""

from __future__ import annotations

import os

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SCRIPT_ROOT = PROJECT_ROOT / "Script"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from anisonet.pinn import get_profile


OUTPUT_ROOT = PROJECT_ROOT / "codexAnalysis" / "profile_definitions"

PROFILE_NOTES = {
    "fourier_refined_16g": {
        "role": "Conservative default",
        "selection_basis": "Refined Fourier profile selected for source fit, tissue restriction, and physics regularization balance.",
        "manuscript_use": "Primary conservative physics-regularized configuration unless a source-fit-optimized comparison is explicitly shown.",
    },
    "fourier_refined_low_pde_16g": {
        "role": "Source-fit-optimized candidate",
        "selection_basis": "All-section low-PDE validation improved source Pearson in 16/16 GSE193107 Apoe/Gfap sample-target pairs; seed-stability check improved 12/12 representative seed-level comparisons.",
        "manuscript_use": "Sensitivity or candidate optimized profile; report Gfap roughness trade-off and do not silently replace the default.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export anisoNET profile parameter table.")
    parser.add_argument(
        "--profiles",
        nargs="*",
        default=["fourier_refined_16g", "fourier_refined_low_pde_16g"],
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for profile_name in args.profiles:
        profile = get_profile(profile_name)
        row = asdict(profile)
        row.update(PROFILE_NOTES.get(profile_name, {}))
        row["data_to_pde_weight_ratio"] = row["data_weight"] / row["pde_weight"]
        rows.append(row)
    write_csv(rows, output_dir / "anisonet_pinn_profile_table.csv")
    write_markdown(rows, output_dir / "anisonet_pinn_profile_table.md")
    print(f"Wrote profile tables to {output_dir}", flush=True)


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    columns = [
        "name",
        "role",
        "network",
        "hidden_width",
        "hidden_depth",
        "fourier_features",
        "fourier_sigma",
        "num_domain",
        "num_boundary",
        "iterations",
        "learning_rate",
        "data_weight",
        "pde_weight",
        "data_to_pde_weight_ratio",
        "boundary_weight",
        "background_weight",
        "smoothness_weight",
        "display_every",
        "selection_basis",
        "manuscript_use",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: list[dict[str, object]], path: Path) -> None:
    lines = [
        "# anisoNET PINN Profile Definitions",
        "",
        "This table documents the conservative default and the low-PDE source-fit-optimized candidate profile.",
        "",
        "| Profile | Role | Data weight | PDE weight | Data/PDE ratio | Fourier features | Sigma | Iterations | Manuscript use |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"`{row['name']}` | {row.get('role', '')} | "
            f"{row['data_weight']:.3g} | {row['pde_weight']:.3g} | "
            f"{row['data_to_pde_weight_ratio']:.3g} | {row['fourier_features']} | "
            f"{row['fourier_sigma']:.3g} | {row['iterations']} | "
            f"{row.get('manuscript_use', '')} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### `{row['name']}`",
                "",
                f"- Role: {row.get('role', '')}",
                f"- Selection basis: {row.get('selection_basis', '')}",
                f"- Manuscript use: {row.get('manuscript_use', '')}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()

