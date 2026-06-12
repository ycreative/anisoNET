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
from scipy.ndimage import gaussian_filter, zoom


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("ANISONET_ANALYSIS_ROOT", PROJECT_ROOT / "codexAnalysis"))
OUT_DIR = ROOT / "manuscript_figures" / "Figure3_benchmark_and_synthetic_validation"
SYN_ROOT = ROOT / "synthetic_barrier" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin"
SEED0 = SYN_ROOT / "seed0"
PREFLIGHT = ROOT / "preflight" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "Apoe_CNS_Myelin"
HELDOUT = ROOT / "heldout" / "brain_aging_gse193107" / "summary" / "heldout_group_summary.csv"
SYN_ALL_RUNS = SYN_ROOT / "synthetic_barrier_all_runs.csv"

PAPER_COLORS = {
    "teal": "#4C9A91",
    "blue": "#5E81AC",
    "terracotta": "#C36F5B",
    "lavender": "#8C7BB7",
    "ochre": "#C7A35A",
    "sage": "#7BA05B",
    "slate": "#7E8796",
    "dark": "#2F3A45",
}
SPLIT_COLORS = {"20/80": PAPER_COLORS["blue"], "80/20": PAPER_COLORS["teal"]}
GENE_COLORS = {"Apoe": PAPER_COLORS["teal"], "Gfap": PAPER_COLORS["lavender"]}


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
    fig.savefig(OUT_DIR / f"{stem}.pdf", dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.10, y: float = 1.08) -> None:
    # Panel letters are added manually during Inkscape layout.
    return


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


def smooth_masked_display(panel: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    valid = mask & np.isfinite(panel)
    if not np.any(valid):
        return panel
    values = np.where(valid, panel, 0.0)
    weights = valid.astype(float)
    smooth = gaussian_filter(values, sigma=sigma, mode="nearest")
    norm_weights = gaussian_filter(weights, sigma=sigma, mode="nearest")
    return np.where(mask, smooth / np.maximum(norm_weights, 1e-6), np.nan)


def upsample_display(panel: np.ndarray, min_short_side: int = 620) -> np.ndarray:
    """Upsample display rasters so PDF export does not preserve tiny Visium grids."""
    data = np.asarray(panel, dtype=float)
    if data.ndim != 2 or data.size == 0:
        return data
    short_side = min(data.shape)
    if short_side <= 0 or short_side >= min_short_side:
        return data
    scale = min_short_side / short_side
    valid = np.isfinite(data)
    values = np.where(valid, data, 0.0)
    weights = valid.astype(float)
    values_z = zoom(values, scale, order=3, mode="nearest")
    weights_z = zoom(weights, scale, order=1, mode="nearest")
    out = values_z / np.maximum(weights_z, 1e-6)
    return np.where(weights_z > 0.35, out, np.nan)


def display_input_grid(arr: np.ndarray, mask: np.ndarray, *, sigma: float = 5.0) -> np.ndarray:
    panel = np.where(mask, norm(arr), np.nan)
    return upsample_display(smooth_masked_display(panel, mask, sigma))


def display_synthetic_grid(arr: np.ndarray, mask: np.ndarray, *, sigma: float = 3.8) -> np.ndarray:
    panel = np.where(mask, norm(arr), np.nan)
    return upsample_display(smooth_masked_display(panel, mask, sigma))


def display_synthetic_grid_shared(arr: np.ndarray, mask: np.ndarray, vmin: float, vmax: float, *, sigma: float = 3.8) -> np.ndarray:
    panel = np.where(mask, norm(arr, vmin=vmin, vmax=vmax), np.nan)
    return upsample_display(smooth_masked_display(panel, mask, sigma))


def display_synthetic_diverging(arr: np.ndarray, mask: np.ndarray, *, sigma: float = 3.0) -> tuple[np.ndarray, float]:
    data = np.where(mask, arr, np.nan)
    valid = np.abs(data[np.isfinite(data)])
    limit = float(np.nanpercentile(valid, 98)) if valid.size else 1.0
    if limit <= 0:
        limit = 1.0
    smooth = smooth_masked_display(data, mask, sigma)
    return upsample_display(smooth), limit


def opaque_cmap_image(panel: np.ndarray, cmap: str, vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """Map a scalar display panel to opaque RGB with white outside tissue.

    Matplotlib/PDF/Inkscape can otherwise preserve NaN edges as transparent image
    masks, which may render as dark halos after PDF export.
    """
    data = np.asarray(panel, dtype=float)
    valid = np.isfinite(data)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    mapper = mpl.colormaps[cmap]
    normalized = np.clip((np.where(valid, data, vmin) - vmin) / (vmax - vmin), 0, 1)
    rgb = mapper(normalized)[..., :3]
    rgb[~valid] = 1.0
    return rgb


def fig3a_heldout_benchmark() -> None:
    df = pd.read_csv(HELDOUT)
    keep = ["nearest", "idw_k8", "gaussian_sigma1p5", "gaussian_sigma3", "anisoNET_masked", "anisoNET_gauss07"]
    df = df[df["method_label"].isin(keep)].copy()
    grouped = df.groupby(["target_gene", "method_label"], as_index=False).agg(
        mean=("test_pearson_mean", "mean"),
        sem=("test_pearson_sem", lambda x: float(np.sqrt(np.sum(np.square(x))) / len(x))),
    )
    x_base = np.arange(len(keep))
    width = 0.30
    fig, ax = plt.subplots(figsize=(5.9, 2.7))
    panel_label(ax, "A", x=-0.12, y=1.12)
    for i, gene in enumerate(["Apoe", "Gfap"]):
        sub = grouped[grouped["target_gene"] == gene].set_index("method_label").reindex(keep)
        ax.bar(x_base + (i - 0.5) * width, sub["mean"], yerr=sub["sem"], width=width, label=gene, color=GENE_COLORS[gene], edgecolor=PAPER_COLORS["dark"], linewidth=0.45)
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
    fig = plt.figure(figsize=(4.6, 2.0))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1], wspace=0.08)
    panels = [("Source prior", source, "magma"), ("Barrier prior", barrier, "YlOrBr"), ("Synthetic truth", truth, "magma")]
    for i, (title, arr, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            panel_label(ax, "B", x=-0.20, y=1.10)
        if i < 2:
            panel = display_input_grid(arr, mask)
            interpolation = "bicubic"
        else:
            panel = display_synthetic_grid(arr, mask, sigma=2.0)
            interpolation = "nearest"
        ax.imshow(opaque_cmap_image(panel, cmap, 0, 1), origin="lower", interpolation=interpolation)
        ax.set_title(title, fontsize=7.3, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Synthetic source-barrier benchmark inputs", fontsize=8.8, fontweight="bold", y=1.03)
    save_panel(fig, "Fig3B_synthetic_design")


def fig3c_synthetic_maps() -> None:
    run_label = "train 20% / test 80%"
    run_dir = SYN_ROOT / "seed0_train20_test80"
    mask = np.load(PREFLIGHT / "tissue_mask.npy") > 0
    barrier = np.load(PREFLIGHT / "barrier_grid.npy")
    truth = np.load(run_dir / "truth.npy")
    plus = np.load(run_dir / "anisoNET_original_barrier.npy")
    minus = np.load(run_dir / "anisoNET_no_transcript_barrier.npy")
    excess = minus - plus
    field_arrays = [truth, plus, minus]
    diff_arrays = [excess]
    field_values = np.concatenate([np.asarray(x)[mask & np.isfinite(x)].ravel() for x in field_arrays])
    field_vmin, field_vmax = np.percentile(field_values, [1, 99]) if field_values.size else (0.0, 1.0)
    diff_values = np.concatenate([np.abs(np.asarray(x)[mask & np.isfinite(x)]).ravel() for x in diff_arrays])
    diff_limit = float(np.percentile(diff_values, 95)) if diff_values.size else 1.0
    if diff_limit <= 0:
        diff_limit = 1.0
    barrier_display = display_input_grid(barrier, mask, sigma=5.0)
    barrier_level = float(np.nanpercentile(barrier_display[np.isfinite(barrier_display)], 78))

    metrics = pd.read_csv(SYN_ALL_RUNS)
    run_metrics = metrics[(metrics["run"] == run_dir.name) & metrics["method"].isin(["anisoNET_original_barrier", "anisoNET_no_transcript_barrier"])]
    leak_plus = float(run_metrics.loc[run_metrics["method"] == "anisoNET_original_barrier", "high_to_low_barrier_ratio"].iloc[0])
    leak_minus = float(run_metrics.loc[run_metrics["method"] == "anisoNET_no_transcript_barrier", "high_to_low_barrier_ratio"].iloc[0])

    fig = plt.figure(figsize=(6.7, 2.25))
    gs = gridspec.GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.045], wspace=0.05)
    diff_image = None
    panels = [
        ("Truth", truth, "field"),
        (f"aNET+\nbarrier\nleak {leak_plus:.2f}", plus, "field"),
        (f"aNET-\nno barrier\nleak {leak_minus:.2f}", minus, "field"),
        ("No-barrier\nexcess", excess, "diff"),
        ("High-barrier\nprior", barrier, "barrier"),
    ]
    for col_idx, (label, arr, kind) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col_idx])
        if col_idx == 0:
            panel_label(ax, "C", x=-0.20, y=1.10)
        if kind == "field":
            panel = display_synthetic_grid_shared(arr, mask, field_vmin, field_vmax)
            ax.imshow(opaque_cmap_image(panel, "magma", 0, 1), origin="lower", interpolation="nearest")
        elif kind == "diff":
            panel = upsample_display(smooth_masked_display(np.where(mask, arr, np.nan), mask, sigma=3.0))
            ax.imshow(opaque_cmap_image(panel, "coolwarm", -diff_limit, diff_limit), origin="lower", interpolation="nearest")
            ax.contour(barrier_display, levels=[barrier_level], colors=["#17A6B8"], linewidths=0.7, origin="lower")
        else:
            ax.imshow(opaque_cmap_image(barrier_display, "YlOrBr", 0, 1), origin="lower", interpolation="nearest")
        ax.set_title(label, fontsize=7, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    cax_diff = fig.add_subplot(gs[0, 5])
    diff_image = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-diff_limit, vmax=diff_limit), cmap="coolwarm")
    cbar_diff = fig.colorbar(diff_image, cax=cax_diff)
    cbar_diff.outline.set_linewidth(0.3)
    cbar_diff.ax.tick_params(labelsize=5.2, length=1.4, width=0.3)
    cbar_diff.set_label("No-barrier excess", fontsize=5.2, labelpad=1)
    fig.text(0.55, 0.06, f"Representative stress case: {run_label}. Excess = aNET no-barrier field minus aNET barrier field; cyan contour marks high barrier.", ha="center", va="center", fontsize=5.6, color=PAPER_COLORS["dark"])
    fig.suptitle("Synthetic barrier prior suppresses no-barrier field excess", fontsize=9, fontweight="bold", y=1.02)
    save_panel(fig, "Fig3C_synthetic_prediction_maps")


def fig3d_error_leakage_maps() -> None:
    truth = np.load(SEED0 / "truth.npy")
    barrier = np.load(PREFLIGHT / "barrier_grid.npy")
    mask = np.load(PREFLIGHT / "tissue_mask.npy") > 0
    full = np.load(SEED0 / "anisoNET_original_barrier.npy")
    none = np.load(SEED0 / "anisoNET_no_transcript_barrier.npy")
    err_full = np.abs(full - truth)
    err_none = np.abs(none - truth)
    improvement = err_none - err_full
    high_context = display_input_grid(barrier, mask, sigma=5.0)
    maps = [
        ("Error\naNET+", err_full, "magma"),
        ("Error\naNET-", err_none, "magma"),
        ("Error reduction\naNET- minus aNET+", improvement, "coolwarm"),
        ("High-barrier\ncontext", high_context, "Greys"),
    ]
    fig = plt.figure(figsize=(5.3, 1.8))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.06)
    for i, (title, arr, cmap) in enumerate(maps):
        ax = fig.add_subplot(gs[0, i])
        if i == 0:
            panel_label(ax, "D", x=-0.20, y=1.10)
        if "reduction" in title:
            panel, vmax = display_synthetic_diverging(arr, mask)
            ax.imshow(opaque_cmap_image(panel, cmap, -vmax, vmax), origin="lower", interpolation="nearest")
        else:
            if "context" in title:
                panel = arr
                vmin, vmax = 0, 1
            else:
                panel = display_synthetic_grid(arr, mask)
                vmin, vmax = 0, 1
            ax.imshow(opaque_cmap_image(panel, cmap, vmin, vmax), origin="lower", interpolation="nearest")
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
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"wspace": 0.34})
    panel_label(axes[0], "E", x=-0.20, y=1.10)
    metrics = [
        ("grid_pearson_truth_mean", "Truth\nPearson", True, "{:.2f}"),
        ("grid_mse_truth_mean", "Truth\nMSE", False, "{:.3f}"),
        ("high_to_low_barrier_ratio_mean", "Leakage\nratio", False, "{:.2f}"),
    ]
    score_df = df.copy()
    for value_col, _, higher_better, _ in metrics:
        values = score_df[value_col].astype(float)
        oriented = values if higher_better else -values
        lo = float(oriented.min())
        hi = float(oriented.max())
        score_df[f"{value_col}_score"] = 0.5 if hi <= lo else (oriented - lo) / (hi - lo)

    image = None
    for ax, split in zip(axes, ["80/20", "20/80"]):
        sub = score_df[score_df["split"] == split].set_index("method").reindex(keep)
        score_matrix = np.column_stack([sub[f"{value_col}_score"].to_numpy(dtype=float) for value_col, _, _, _ in metrics])
        value_matrix = np.column_stack([sub[value_col].to_numpy(dtype=float) for value_col, _, _, _ in metrics])
        image = ax.imshow(score_matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
        ax.set_title(f"train/test {split}", fontsize=7.4, pad=3)
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_xticklabels([label for _, label, _, _ in metrics], fontsize=6.2)
        if ax is axes[0]:
            ax.set_yticks(np.arange(len(keep)))
            ax.set_yticklabels([clean_label(method).replace("\n", " ") for method in keep], fontsize=6.1)
        else:
            ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, len(metrics), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(keep), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        for row in range(score_matrix.shape[0]):
            for col, (_, _, _, formatter) in enumerate(metrics):
                score = score_matrix[row, col]
                color = "white" if score > 0.62 else PAPER_COLORS["dark"]
                ax.text(col, row, formatter.format(value_matrix[row, col]), ha="center", va="center", fontsize=5.2, color=color)
        for spine in ax.spines.values():
            spine.set_visible(False)
    if image is not None:
        cbar = fig.colorbar(image, ax=axes, fraction=0.030, pad=0.025)
        cbar.outline.set_linewidth(0.3)
        cbar.ax.tick_params(labelsize=5.4, length=1.4, width=0.3)
        cbar.ax.set_ylabel("relative score", fontsize=5.8)
    axes[1].text(
        0.98,
        -0.18,
        "MSE and leakage are score-reversed so darker cells indicate better performance.",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color="#555555",
    )
    fig.suptitle("Synthetic benchmark method-by-metric heatmap", fontsize=9, fontweight="bold", y=1.03)
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
    fig, ax = plt.subplots(figsize=(3.4, 2.45))
    panel_label(ax, "F", x=-0.18, y=1.10)
    offsets = {"20/80": -0.12, "80/20": 0.12}
    for split in ["20/80", "80/20"]:
        sub = out[out["split"] == split].set_index("metric").reindex([label for _, label in order])
        xpos = x + offsets[split]
        color = SPLIT_COLORS[split]
        for xp, val, sd in zip(xpos, sub["delta"], sub["sd"]):
            ax.plot([xp, xp], [0, val], color=color, linewidth=1.0, alpha=0.90, zorder=2)
            ax.errorbar(xp, val, yerr=sd, fmt="o", markersize=4.2, color=color, markeredgecolor=PAPER_COLORS["dark"], markeredgewidth=0.45, elinewidth=0.75, capsize=2.0, label=f"train/test {split}" if xp == xpos[0] else None, zorder=3)
    ax.axhline(0, color=PAPER_COLORS["dark"], linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in order])
    ax.set_ylabel("Barrier - no barrier")
    ax.set_title("Paired barrier ablation")
    ax.text(0.03, 0.04, "Pearson higher;\nMSE/leakage lower.", transform=ax.transAxes, fontsize=6, color="#555555")
    ax.grid(axis="y", color="#dddddd", linewidth=0.5)
    ax.legend(frameon=False, fontsize=6, loc="upper right")
    save_panel(fig, "Fig3F_paired_barrier_ablation")


def fig3g_synthetic_seed_split_robustness() -> None:
    df = pd.read_csv(SYN_ALL_RUNS)
    rows = []
    for (run, seed, train_fraction), sub in df.groupby(["run", "seed", "train_fraction"]):
        methods = sub.set_index("method")
        barrier = methods.loc["anisoNET_original_barrier"]
        no_barrier = methods.loc["anisoNET_no_transcript_barrier"]
        split = "20/80" if float(train_fraction) == 0.2 else "80/20"
        rows.append(
            {
                "split": split,
                "seed": int(seed),
                "label": f"{split} seed {int(seed)}",
                "truth_pearson_adv": barrier["grid_pearson_truth"] - no_barrier["grid_pearson_truth"],
                "truth_mse_adv": no_barrier["grid_mse_truth"] - barrier["grid_mse_truth"],
                "leakage_adv": no_barrier["high_to_low_barrier_ratio"] - barrier["high_to_low_barrier_ratio"],
            }
        )
    out = pd.DataFrame(rows).sort_values(["split", "seed"], ascending=[False, True]).reset_index(drop=True)
    y = np.arange(len(out))
    colors = out["split"].map(SPLIT_COLORS).tolist()

    fig, axes = plt.subplots(1, 3, figsize=(5.9, 2.55), sharey=True, gridspec_kw={"wspace": 0.22})
    panel_label(axes[0], "G", x=-0.22, y=1.10)
    metrics = [
        ("truth_pearson_adv", "Truth Pearson\nadvantage"),
        ("truth_mse_adv", "Truth MSE\nadvantage"),
        ("leakage_adv", "Leakage ratio\nadvantage"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.axvline(0, color=PAPER_COLORS["dark"], linewidth=0.65)
        for yi, val, color in zip(y, out[col], colors):
            ax.plot([0, val], [yi, yi], color=color, linewidth=1.0, alpha=0.85)
            ax.scatter(val, yi, s=28, color=color, edgecolor=PAPER_COLORS["dark"], linewidth=0.35, zorder=3)
        ax.set_title(title, fontsize=7.0)
        ax.grid(axis="x", color="#dddddd", linewidth=0.45)
        ax.tick_params(axis="x", labelsize=5.7)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(out["label"], fontsize=6.0)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Barrier - no barrier")
    axes[1].set_xlabel("No barrier - barrier")
    axes[2].set_xlabel("No barrier - barrier")
    axes[2].text(0.98, 0.04, "positive = barrier helps", transform=axes[2].transAxes, ha="right", va="bottom", fontsize=5.6, color="#555555")
    fig.suptitle("Synthetic barrier effect across seeds and split stress", fontsize=8.9, fontweight="bold", y=1.04)
    save_panel(fig, "Fig3G_synthetic_seed_split_robustness")


def rough_assembly() -> None:
    files = [
        "Fig3A_generic_heldout_benchmark.png",
        "Fig3B_synthetic_design.png",
        "Fig3C_synthetic_prediction_maps.png",
        "Fig3D_synthetic_error_and_leakage_maps.png",
        "Fig3E_synthetic_metric_summary.png",
        "Fig3F_paired_barrier_ablation.png",
        "Fig3G_synthetic_seed_split_robustness.png",
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
    fig3g_synthetic_seed_split_robustness()
    rough_assembly()
    print(f"Wrote Figure 3 panel assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
