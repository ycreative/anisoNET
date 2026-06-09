"""Generate separated Figure 3 panel assets for benchmark and synthetic validation."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from PIL import Image


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("ANISONET_ANALYSIS_ROOT", PROJECT_ROOT / "codexAnalysis"))
OUT_DIR = ROOT / "manuscript_figures" / "Figure3_benchmark_and_synthetic_validation"
SYN_ROOT = ROOT / "synthetic_barrier" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin"
SEED0 = SYN_ROOT / "seed0"
PREFLIGHT = ROOT / "preflight" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin"
HELDOUT = ROOT / "heldout" / "brain_aging_gse193107" / "summary" / "heldout_group_summary.csv"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.5,
            "axes.linewidth": 0.55,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_panel(fig: plt.Figure, stem: str) -> None:
    ensure_dir(OUT_DIR)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.08) -> None:
    return None


def clean_label(label: str) -> str:
    return {
        "nearest": "Near",
        "idw_k8": "IDW",
        "gaussian_sigma1p5": "Gauss1.5",
        "gaussian_sigma3": "Gauss3",
        "anisoNET_masked": "aNET\nmasked",
        "anisoNET_gauss07": "aNET\nsmooth",
        "graph_smooth_k6_iter3": "Graph6",
        "graph_smooth_k12_iter5": "Graph12",
        "anisoNET_original_barrier": "aNET+",
        "anisoNET_no_transcript_barrier": "aNET-",
    }.get(str(label), str(label))


def norm(arr: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if vmin is None or vmax is None:
        valid = data[np.isfinite(data)]
        vmin, vmax = np.percentile(valid, [1, 99]) if valid.size else (0, 1)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return np.clip((data - vmin) / (vmax - vmin), 0, 1)


def fig3a_heldout_benchmark() -> None:
    df = pd.read_csv(HELDOUT)
    keep = ["nearest", "idw_k8", "gaussian_sigma1p5", "gaussian_sigma3", "anisoNET_masked", "anisoNET_gauss07"]
    df = df[df["method_label"].isin(keep)].copy()
    grouped = df.groupby(["target_gene", "method_label"], as_index=False).agg(
        mean=("test_pearson_mean", "mean"),
        sem=("test_pearson_sem", lambda x: float(np.sqrt(np.sum(np.square(x))) / len(x))),
    )
    x_base = np.arange(len(keep))
    width = 0.36
    colors = {"Apoe": "#4C78A8", "Gfap": "#B279A2"}
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    panel_label(ax, "A", x=-0.12, y=1.12)
    for i, gene in enumerate(["Apoe", "Gfap"]):
        sub = grouped[grouped["target_gene"] == gene].set_index("method_label").reindex(keep)
        ax.bar(x_base + (i - 0.5) * width, sub["mean"], yerr=sub["sem"], width=width, label=gene, color=colors[gene], edgecolor="#222222", linewidth=0.5)
    ax.set_xticks(x_base)
    ax.set_xticklabels([clean_label(x) for x in keep], rotation=35, ha="right")
    ax.set_ylabel("Held-out Pearson")
    ax.set_ylim(0, 0.86)
    ax.set_title("Generic held-out expression prediction")
    ax.legend(frameon=False, ncols=2, loc="upper left")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    save_panel(fig, "Fig3A_generic_heldout_benchmark")


def fig3b_synthetic_design() -> None:
    truth = np.load(SEED0 / "truth.npy")
    source = np.load(PREFLIGHT / "source_grid.npy")
    barrier = np.load(PREFLIGHT / "barrier_grid.npy")
    mask = np.load(PREFLIGHT / "tissue_mask.npy") > 0
    fig = plt.figure(figsize=(5.8, 2.2))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1, 1.15], wspace=0.08)
    panels = [("Source prior", source, "magma"), ("Barrier prior", barrier, "YlOrBr"), ("Synthetic truth", truth, "magma")]
    for i, (title, arr, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            panel_label(ax, "B", x=-0.20, y=1.10)
        ax.imshow(np.where(mask, norm(arr), np.nan), cmap=cmap, origin="lower", interpolation="bilinear")
        ax.set_title(title, fontsize=7.3, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    ax = fig.add_subplot(gs[0, 3])
    ax.axis("off")
    ax.text(0.02, 0.82, "Synthetic barrier stress test", fontsize=8.4, fontweight="bold")
    ax.text(0.02, 0.60, "Known truth field\nTrain/test splits: 20/80, 80/20\nCompare full barrier vs no barrier", fontsize=7.2, va="top", linespacing=1.25)
    ax.text(0.02, 0.22, "Primary endpoint:\nhigh/low barrier leakage", fontsize=7.0, color="#8a1c1c", va="top")
    save_panel(fig, "Fig3B_synthetic_design")


def fig3c_synthetic_maps() -> None:
    specs = [
        ("Truth", "truth.npy"),
        ("aNET+\nbarrier", "anisoNET_original_barrier.npy"),
        ("aNET-\nno barrier", "anisoNET_no_transcript_barrier.npy"),
        ("Gauss3", "gaussian_sigma3.npy"),
        ("IDW", "idw_k8.npy"),
    ]
    arrays = [(label, np.load(SEED0 / name)) for label, name in specs]
    valid = np.concatenate([arr[np.isfinite(arr)].reshape(-1) for _, arr in arrays])
    vmin, vmax = np.percentile(valid, [1, 99])
    fig = plt.figure(figsize=(6.3, 1.75))
    gs = gridspec.GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.04], wspace=0.04)
    image = None
    for i, (label, arr) in enumerate(arrays):
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            panel_label(ax, "C", x=-0.20, y=1.10)
        image = ax.imshow(norm(arr, vmin, vmax), cmap="magma", origin="lower", interpolation="bilinear")
        ax.set_title(label, fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax = fig.add_subplot(gs[0, 5])
    cbar = fig.colorbar(image, cax=cax)
    cbar.outline.set_linewidth(0.3)
    cbar.ax.tick_params(labelsize=5.5, length=1.4, width=0.3)
    fig.suptitle("Representative synthetic field predictions", fontsize=9, fontweight="bold", y=1.04)
    save_panel(fig, "Fig3C_synthetic_prediction_maps")


def fig3d_error_leakage_maps() -> None:
    truth = np.load(SEED0 / "truth.npy")
    barrier = np.load(PREFLIGHT / "barrier_grid.npy")
    full = np.load(SEED0 / "anisoNET_original_barrier.npy")
    none = np.load(SEED0 / "anisoNET_no_transcript_barrier.npy")
    err_full = np.abs(full - truth)
    err_none = np.abs(none - truth)
    improvement = err_none - err_full
    high = barrier >= np.nanpercentile(barrier, 80)
    maps = [
        ("Error\naNET+", err_full, "magma"),
        ("Error\naNET-", err_none, "magma"),
        ("Error reduction\naNET- minus aNET+", improvement, "coolwarm"),
        ("High-barrier\nmask", high.astype(float), "Greys"),
    ]
    fig = plt.figure(figsize=(5.3, 1.8))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.06)
    for i, (title, arr, cmap) in enumerate(maps):
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            panel_label(ax, "D", x=-0.20, y=1.10)
        if "reduction" in title:
            vmax = float(np.nanpercentile(np.abs(arr), 98))
            ax.imshow(arr, cmap=cmap, origin="lower", vmin=-vmax, vmax=vmax, interpolation="bilinear")
        else:
            ax.imshow(norm(arr), cmap=cmap, origin="lower", interpolation="bilinear")
        ax.set_title(title, fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Synthetic error and high-barrier context", fontsize=9, fontweight="bold", y=1.04)
    save_panel(fig, "Fig3D_synthetic_error_and_leakage_maps")


def fig3e_metric_summary() -> None:
    df = pd.read_csv(SYN_ROOT / "synthetic_barrier_summary.csv")
    keep = ["nearest", "idw_k8", "gaussian_sigma3", "graph_smooth_k6_iter3", "graph_smooth_k12_iter5", "anisoNET_original_barrier", "anisoNET_no_transcript_barrier"]
    df = df[df["method"].isin(keep)].copy()
    df["split"] = df["train_fraction"].map({0.2: "20/80", 0.8: "80/20"})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.45), gridspec_kw={"wspace": 0.30})
    panel_label(axes[0], "E", x=-0.20, y=1.10)
    metrics = [
        ("grid_pearson_truth_mean", "grid_pearson_truth_sd", "Truth Pearson", False),
        ("grid_mse_truth_mean", "grid_mse_truth_sd", "Truth MSE", True),
        ("high_to_low_barrier_ratio_mean", "high_to_low_barrier_ratio_sd", "Leakage ratio", True),
    ]
    x = np.arange(len(keep))
    width = 0.36
    colors = {"20/80": "#E15759", "80/20": "#59A14F"}
    for ax, (mean_col, sd_col, title, lower) in zip(axes, metrics):
        for i, split in enumerate(["80/20", "20/80"]):
            sub = df[df["split"] == split].set_index("method").reindex(keep)
            ax.bar(x + (i - 0.5) * width, sub[mean_col], yerr=sub[sd_col], width=width, color=colors[split], edgecolor="#222222", linewidth=0.45, label=f"train/test {split}")
        ax.set_xticks(x)
        ax.set_xticklabels([clean_label(m) for m in keep], rotation=45, ha="right", fontsize=6.0)
        ax.set_title(title)
        ax.grid(axis="y", color="#dddddd", linewidth=0.5)
        if lower:
            ax.text(0.98, 0.04, "lower better", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8, color="#555555")
    axes[0].legend(frameon=False, fontsize=5.8, loc="lower right")
    fig.suptitle("Synthetic benchmark metrics", fontsize=9, fontweight="bold", y=1.05)
    save_panel(fig, "Fig3E_synthetic_metric_summary")


def fig3f_paired_ablation() -> None:
    df = pd.read_csv(SYN_ROOT / "synthetic_barrier_paired_ablation_stats.csv")
    order = [("grid_pearson_truth", "Truth\nPearson"), ("grid_mse_truth", "Truth\nMSE"), ("high_to_low_barrier_ratio", "Leakage\nratio")]
    rows = []
    for split, label in [(0.2, "20/80"), (0.8, "80/20")]:
        for metric, metric_label in order:
            row = df[(df["train_fraction"] == split) & (df["metric"] == metric)].iloc[0]
            rows.append({"split": label, "metric": metric_label, "delta": row["paired_difference_mean"], "sd": row["paired_difference_sd"]})
    out = pd.DataFrame(rows)
    x = np.arange(len(order))
    width = 0.36
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    panel_label(ax, "F", x=-0.18, y=1.10)
    for i, split in enumerate(["20/80", "80/20"]):
        sub = out[out["split"] == split].set_index("metric").reindex([label for _, label in order])
        ax.bar(x + (i - 0.5) * width, sub["delta"], yerr=sub["sd"], width=width, label=f"train/test {split}", color={"20/80": "#E15759", "80/20": "#59A14F"}[split], edgecolor="#222222", linewidth=0.5)
    ax.axhline(0, color="#222222", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in order])
    ax.set_ylabel("Barrier - no barrier")
    ax.set_title("Paired barrier ablation")
    ax.text(0.03, 0.04, "Pearson higher;\nMSE/leakage lower.", transform=ax.transAxes, fontsize=6, color="#555555")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    save_panel(fig, "Fig3F_paired_barrier_ablation")


def fig3g_attenuation_index() -> None:
    df = pd.read_csv(SYN_ROOT / "synthetic_barrier_paired_ablation_stats.csv")
    rows = []
    for split, label in [(0.2, "20/80"), (0.8, "80/20")]:
        row = df[(df["train_fraction"] == split) & (df["metric"] == "high_to_low_barrier_ratio")].iloc[0]
        attenuation = (row["no_barrier_mean"] - row["barrier_mean"]) / row["no_barrier_mean"]
        rows.append({"split": label, "attenuation": attenuation})
    out = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(2.7, 2.35))
    panel_label(ax, "G", x=-0.20, y=1.10)
    ax.bar(np.arange(len(out)), out["attenuation"], color=["#E15759", "#59A14F"], edgecolor="#222222", linewidth=0.5)
    ax.set_xticks(np.arange(len(out)))
    ax.set_xticklabels([f"train/test\n{x}" for x in out["split"]])
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Barrier attenuation index")
    ax.set_title("Leakage reduction")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    for i, val in enumerate(out["attenuation"]):
        ax.text(i, val + 0.025, f"{val:.2f}", ha="center", fontsize=7)
    save_panel(fig, "Fig3G_barrier_attenuation_index")


def rough_assembly() -> None:
    files = [
        "Fig3A_generic_heldout_benchmark.png",
        "Fig3B_synthetic_design.png",
        "Fig3C_synthetic_prediction_maps.png",
        "Fig3D_synthetic_error_and_leakage_maps.png",
        "Fig3E_synthetic_metric_summary.png",
        "Fig3F_paired_barrier_ablation.png",
        "Fig3G_barrier_attenuation_index.png",
    ]
    fig = plt.figure(figsize=(8.2, 9.2))
    gs = gridspec.GridSpec(5, 2, figure=fig, height_ratios=[1.05, 0.78, 0.78, 1.0, 0.9], hspace=0.18, wspace=0.08)
    layout = [
        (files[0], gs[0, :]),
        (files[1], gs[1, :]),
        (files[2], gs[2, :]),
        (files[3], gs[3, 0]),
        (files[4], gs[3, 1]),
        (files[5], gs[4, 0]),
        (files[6], gs[4, 1]),
    ]
    for file_name, spec in layout:
        ax = fig.add_subplot(spec)
        ax.imshow(Image.open(OUT_DIR / file_name).convert("RGB"))
        ax.axis("off")
    fig.suptitle("Figure 3 rough assembly: benchmark and synthetic barrier validation", fontsize=12, fontweight="bold", y=0.99)
    save_panel(fig, "Figure3_rough_assembly")


def main() -> None:
    setup()
    ensure_dir(OUT_DIR)
    fig3a_heldout_benchmark()
    fig3b_synthetic_design()
    fig3c_synthetic_maps()
    fig3d_error_leakage_maps()
    fig3e_metric_summary()
    fig3f_paired_ablation()
    fig3g_attenuation_index()
    rough_assembly()
    print(f"Wrote Figure 3 panel assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
