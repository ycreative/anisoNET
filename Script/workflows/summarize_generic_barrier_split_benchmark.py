"""Summarize generic barrier-split benchmark runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize generic barrier-split benchmark outputs.")
    parser.add_argument("--input-csv", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def infer_run(path: Path) -> tuple[str, str]:
    name = path.parent.name
    path_text = str(path)
    if "Slc34a1" in path_text:
        target = "Slc34a1"
    elif "Umod" in path_text:
        target = "Umod"
    else:
        target = "unknown"
    if "prior" in path_text:
        profile = "prior_hybrid"
    elif name.endswith("_sourcefit_probe"):
        profile = "sourcefit_probe"
    elif name.endswith("_low_pde"):
        profile = "low_pde"
    else:
        profile = "default"
    return target, profile


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for csv_path in args.input_csv:
        path = Path(csv_path)
        frame = pd.read_csv(path)
        target, profile = infer_run(path)
        frame.insert(0, "target_gene", target)
        frame.insert(1, "profile", profile)
        frame.insert(2, "source_csv", str(path))
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "generic_barrier_split_combined_metrics.csv", index=False)

    key = combined[
        combined["method"].isin(
            [
                "euclidean_idw",
                "resistance_idw",
                "euclidean_gaussian",
                "resistance_gaussian",
                "prior_line_resistance_idw_grid",
                "prior_line_resistance_idw_grid_traincal",
                "anisonet_gauss07",
                "anisonet_masked",
                "anisonet_prior_gauss07",
                "anisonet_prior_gauss07_traincal",
                "anisonet_prior_masked",
                "anisonet_prior_masked_traincal",
            ]
        )
    ].copy()
    key = key.sort_values(["target_gene", "profile", "test_pearson"], ascending=[True, True, False])
    key.to_csv(output_dir / "generic_barrier_split_ranked_metrics.csv", index=False)

    best_rows = key.groupby(["target_gene", "profile"], as_index=False).first()
    best_rows.to_csv(output_dir / "generic_barrier_split_best_by_target_profile.csv", index=False)

    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42})
    methods = [
        "euclidean_idw",
        "resistance_idw",
        "euclidean_gaussian",
        "resistance_gaussian",
        "prior_line_resistance_idw_grid",
        "prior_line_resistance_idw_grid_traincal",
        "anisonet_masked",
        "anisonet_gauss07",
        "anisonet_prior_masked",
        "anisonet_prior_gauss07",
        "anisonet_prior_masked_traincal",
        "anisonet_prior_gauss07_traincal",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), constrained_layout=True)
    for ax, metric, ylabel in [
        (axes[0], "test_pearson", "Held-out Pearson"),
        (axes[1], "test_mse", "Held-out MSE"),
    ]:
        pivot = key.pivot_table(index=["target_gene", "profile"], columns="method", values=metric, aggfunc="first")
        labels = [f"{target}\n{profile}" for target, profile in pivot.index]
        x = range(len(labels))
        width = 0.13
        for i, method in enumerate(methods):
            if method not in pivot:
                continue
            offsets = [v + (i - (len(methods) - 1) / 2) * width for v in x]
            ax.bar(offsets, pivot[method].to_numpy(), width=width, label=method)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
    axes[0].legend(fontsize=6, ncols=2, frameon=False)
    fig.savefig(output_dir / "generic_barrier_split_summary.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "generic_barrier_split_summary.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    interpretation = output_dir / "generic_barrier_split_interpretation.md"
    lines = [
        "# Generic Barrier-Split Benchmark Interpretation",
        "",
        "This summary compares Euclidean interpolation, resistance-aware interpolation, and anisoNET on mouse kidney high-barrier/edge held-out spots.",
        "",
        "## Main Findings",
        "",
    ]
    for target in sorted(key["target_gene"].unique()):
        subset = key[key["target_gene"] == target]
        lines.append(f"### {target}")
        lines.append("")
        for profile in sorted(subset["profile"].unique()):
            profile_frame = subset[subset["profile"] == profile].sort_values("test_pearson", ascending=False)
            top = profile_frame.iloc[0]
            aniso_frame = profile_frame[
                profile_frame["method"].isin(
                    [
                        "anisonet_gauss07",
                        "anisonet_prior_gauss07",
                        "anisonet_prior_gauss07_traincal",
                    ]
                )
            ]
            ridw = profile_frame[profile_frame["method"] == "resistance_idw"].iloc[0]
            eidw = profile_frame[profile_frame["method"] == "euclidean_idw"].iloc[0]
            lines.extend(
                [
                    f"- `{profile}` best method by Pearson: `{top['method']}` (Pearson `{top['test_pearson']:.4f}`, MSE `{top['test_mse']:.4f}`).",
                    f"- Resistance-aware IDW vs Euclidean IDW: Pearson `{eidw['test_pearson']:.4f}` to `{ridw['test_pearson']:.4f}`.",
                ]
            )
            if not aniso_frame.empty:
                aniso = aniso_frame.iloc[0]
                lines.append(f"- Best anisoNET-like gauss07 row: `{aniso['method']}` Pearson `{aniso['test_pearson']:.4f}`, MSE `{aniso['test_mse']:.4f}`.")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "High-barrier/edge splits are more informative than random splits for anatomy-aware modeling. In the no-prior kidney tests, resistance-aware IDW improved over Euclidean IDW, whereas the scalar anisoNET PINN did not yet outperform the stronger spot-interpolation baselines.",
            "",
            "The prior-hybrid probe is mixed but useful: it improves Umod high-barrier/edge prediction beyond resistance-aware IDW, while Slc34a1 remains slightly below the strongest resistance-aware IDW baseline. This suggests the algorithm can be improved by injecting resistance-aware continuous priors, but the result is not yet broad enough for a cross-tissue superiority claim.",
            "",
        ]
    )
    interpretation.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote summary to {output_dir}")


if __name__ == "__main__":
    main()
