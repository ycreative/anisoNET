"""Generate separated Figure 4 panel assets for robustness and targeted-extension evidence."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from PIL import Image
from scipy.ndimage import gaussian_filter


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("ANISONET_ANALYSIS_ROOT", PROJECT_ROOT / "codexAnalysis"))
OUT_DIR = ROOT / "manuscript_figures" / "Figure4_robustness_reproducibility"
EXPORT_DPI = 600

SPATIAL_ROOT = ROOT / "processed_visium" / "brain_aging_gse193107"
PREFLIGHT_ROOT = ROOT / "preflight" / "brain_aging_gse193107"
HISTO = ROOT / "histology_prior" / "brain_aging_gse193107" / "histology_prior_pinn_group_summary.csv"
LOW_PDE = ROOT / "loss_weight_sensitivity" / "brain_aging_gse193107" / "multi_section_low_pde" / "low_pde_profile_validation_group_summary.csv"
SEED = ROOT / "loss_weight_sensitivity" / "brain_aging_gse193107" / "low_pde_seed_stability" / "low_pde_seed_stability_paired_summary.csv"
CLIP = ROOT / "source_clipping_sensitivity" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "source_clipping_sensitivity_summary.csv"
TARGETED_DATASET = ROOT / "targeted_gene_extension" / "full_metrics_by_dataset.csv"
TARGETED_DATASET_GENE = ROOT / "targeted_gene_extension" / "full_metrics_by_dataset_gene.csv"
TARGETED_FULL = ROOT / "targeted_gene_extension" / "full_metrics_summary.csv"

HISTO_REP_SAMPLE = "GSM5773457_Old_mouse_brain_A1-2"
HISTO_REP_TASK = "Apoe_CNS_Myelin"
HISTO_REP_ROOT = ROOT / "histology_prior" / "brain_aging_gse193107" / HISTO_REP_SAMPLE / HISTO_REP_TASK

LOWPDE_REP_SAMPLE = "GSM5773453_Young_mouse_brain_A1-1"
LOWPDE_REP_TASK = "Apoe_CNS_Myelin"
LOWPDE_REP_ROOT = (
    ROOT
    / "loss_weight_sensitivity"
    / "brain_aging_gse193107"
    / "low_pde_seed_stability"
    / LOWPDE_REP_SAMPLE
    / LOWPDE_REP_TASK
    / "brightness"
)


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

COLORS = {
    "Apoe": PAPER_COLORS["teal"],
    "Gfap": PAPER_COLORS["terracotta"],
    "brightness": PAPER_COLORS["blue"],
    "hematoxylin": PAPER_COLORS["lavender"],
    "masked": PAPER_COLORS["sage"],
    "gauss07": PAPER_COLORS["ochre"],
    "default": PAPER_COLORS["slate"],
    "low_pde": PAPER_COLORS["teal"],
}
DATASET_LABELS = {
    "brain_aging_gse193107": "Brain aging",
    "mouse_brain_sagittal_10x": "Sagittal brain",
    "mouse_kidney_10x": "Kidney",
    "mouse_liver_apap_gse280515": "Liver APAP",
}
DATASET_COLORS = {
    "brain_aging_gse193107": PAPER_COLORS["blue"],
    "mouse_brain_sagittal_10x": PAPER_COLORS["lavender"],
    "mouse_kidney_10x": PAPER_COLORS["teal"],
    "mouse_liver_apap_gse280515": PAPER_COLORS["terracotta"],
}


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
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_panel(fig: plt.Figure, stem: str) -> None:
    ensure_dir(OUT_DIR)
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", dpi=EXPORT_DPI, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.10) -> None:
    # Panel letters are added manually during Inkscape layout.
    return


def clean_field(value: str) -> str:
    return {"gauss07": "post", "masked": "mask"}.get(value, value)


def brain_sample_label(sample: str) -> str:
    condition = "Y" if "_Young_" in sample else "O"
    section = sample.split("_brain_")[-1]
    gsm = sample.split("_")[0].replace("GSM577", "G")
    return f"{condition} {section}\n{gsm}"


def crop_to_mask(arr: np.ndarray, mask: np.ndarray, pad: int = 5) -> np.ndarray:
    yy, xx = np.where(mask)
    if yy.size == 0:
        return arr
    y0 = max(int(yy.min()) - pad, 0)
    y1 = min(int(yy.max()) + pad + 1, arr.shape[0])
    x0 = max(int(xx.min()) - pad, 0)
    x1 = min(int(xx.max()) + pad + 1, arr.shape[1])
    return arr[y0:y1, x0:x1]


def normalized_panel(arr: np.ndarray, mask: np.ndarray | None = None, *, log: bool = False) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if log:
        finite = data[np.isfinite(data)]
        floor = np.nanpercentile(finite, 1) * 0.1 if finite.size else 1e-6
        data = np.log10(np.maximum(data, floor))
    valid = np.isfinite(data)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return np.zeros_like(data)
    lo, hi = np.nanpercentile(data[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out = np.clip((data - lo) / (hi - lo), 0, 1)
    if mask is not None:
        out = np.where(mask, out, np.nan)
    return out


def smooth_masked_display(panel: np.ndarray, mask: np.ndarray, sigma: float) -> np.ndarray:
    valid = mask & np.isfinite(panel)
    if not np.any(valid):
        return panel
    values = np.where(valid, panel, 0.0)
    weights = valid.astype(float)
    smooth = gaussian_filter(values, sigma=sigma, mode="nearest")
    norm = gaussian_filter(weights, sigma=sigma, mode="nearest")
    out = smooth / np.maximum(norm, 1e-6)
    return np.where(mask, out, np.nan)


def gradient_magnitude(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.where(mask, arr, np.nan))
    grad = np.sqrt(gx * gx + gy * gy)
    return np.where(mask, grad, np.nan)


def show_grid(
    ax: plt.Axes,
    arr: np.ndarray,
    mask: np.ndarray,
    title: str,
    cmap: str,
    *,
    log: bool = False,
    diverging: bool = False,
    display_smooth_sigma: float | None = None,
) -> None:
    if diverging:
        data = np.where(mask, arr, np.nan)
        valid = np.abs(data[np.isfinite(data)])
        limit = float(np.nanpercentile(valid, 98)) if valid.size else 1.0
        if limit <= 0:
            limit = 1.0
        panel = crop_to_mask(data, mask)
        ax.imshow(panel, cmap=cmap, interpolation="bilinear", origin="lower", vmin=-limit, vmax=limit)
    else:
        data = normalized_panel(arr, mask, log=log)
        if display_smooth_sigma is not None and display_smooth_sigma > 0:
            data = smooth_masked_display(data, mask, display_smooth_sigma)
        panel = crop_to_mask(data, mask)
        interpolation = "bicubic" if display_smooth_sigma is not None and display_smooth_sigma > 0 else "bilinear"
        ax.imshow(panel, cmap=cmap, interpolation=interpolation, origin="lower")
    ax.set_title(title, fontsize=7.1, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def zoom_bounds_from_peak(arr: np.ndarray, mask: np.ndarray, window: int = 46) -> tuple[int, int, int, int]:
    score = np.where(mask & np.isfinite(arr), np.abs(arr), np.nan)
    if np.all(~np.isfinite(score)):
        yy, xx = np.where(mask)
        cy = int(np.nanmean(yy)) if yy.size else arr.shape[0] // 2
        cx = int(np.nanmean(xx)) if xx.size else arr.shape[1] // 2
    else:
        cy, cx = np.unravel_index(np.nanargmax(score), score.shape)
    half = window // 2
    y0 = max(int(cy) - half, 0)
    y1 = min(y0 + window, arr.shape[0])
    x0 = max(int(cx) - half, 0)
    x1 = min(x0 + window, arr.shape[1])
    return y0, y1, x0, x1


def show_grid_zoom(
    ax: plt.Axes,
    arr: np.ndarray,
    mask: np.ndarray,
    bounds: tuple[int, int, int, int],
    title: str,
    cmap: str,
    *,
    diverging: bool = False,
) -> None:
    y0, y1, x0, x1 = bounds
    sub = np.where(mask[y0:y1, x0:x1], arr[y0:y1, x0:x1], np.nan)
    if diverging:
        valid = np.abs(sub[np.isfinite(sub)])
        limit = float(np.nanpercentile(valid, 98)) if valid.size else 1.0
        if limit <= 0:
            limit = 1.0
        ax.imshow(sub, cmap=cmap, interpolation="nearest", origin="lower", vmin=-limit, vmax=limit)
    else:
        sub_mask = mask[y0:y1, x0:x1]
        ax.imshow(normalized_panel(sub, sub_mask), cmap=cmap, interpolation="nearest", origin="lower")
    ax.set_title(title, fontsize=6.9, pad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#5fbf6a")


def show_surface(ax: plt.Axes, arr: np.ndarray, mask: np.ndarray, title: str, cmap: str = "viridis") -> None:
    data = crop_to_mask(normalized_panel(arr, mask), mask)
    data = np.where(np.isfinite(data), data, 0)
    max_side = 75
    step = max(int(np.ceil(max(data.shape) / max_side)), 1)
    z = data[::step, ::step]
    yy, xx = np.mgrid[0 : z.shape[0], 0 : z.shape[1]]
    ax.plot_surface(xx, yy, z, cmap=cmap, linewidth=0, antialiased=True, shade=True, rstride=1, cstride=1)
    ax.view_init(elev=35, azim=-55)
    ax.set_title(title, fontsize=6.9, pad=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 0.35))
    ax.grid(False)


def compact_slope_axis(
    ax: plt.Axes,
    labels: list[str],
    left_values: list[float],
    right_values: list[float],
    left_label: str,
    right_label: str,
    ylabel: str,
    *,
    color_left: str = PAPER_COLORS["blue"],
    color_right: str = PAPER_COLORS["lavender"],
) -> None:
    y = np.arange(len(labels))[::-1]
    for yi, lv, rv in zip(y, left_values, right_values):
        ax.plot([lv, rv], [yi, yi], color="#9aa1a8", linewidth=0.9, zorder=1)
        ax.scatter(lv, yi, s=22, color=color_left, edgecolor="#333333", linewidth=0.35, zorder=2)
        ax.scatter(rv, yi, s=22, color=color_right, edgecolor="#333333", linewidth=0.35, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel(ylabel, fontsize=6.7)
    ax.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    ax.tick_params(axis="x", labelsize=6.2)


def paired_metric_axis(
    ax: plt.Axes,
    labels: list[str],
    left_values: list[float],
    right_values: list[float],
    left_label: str,
    right_label: str,
    ylabel: str,
    *,
    color_left: str = PAPER_COLORS["blue"],
    color_right: str = PAPER_COLORS["lavender"],
) -> None:
    x = np.arange(len(labels))
    for xi, lv, rv in zip(x, left_values, right_values):
        ax.plot([xi, xi], [lv, rv], color="#9aa1a8", linewidth=0.9, zorder=1)
        ax.scatter(xi - 0.055, lv, s=22, color=color_left, edgecolor="#333333", linewidth=0.35, zorder=2)
        ax.scatter(xi + 0.055, rv, s=22, color=color_right, edgecolor="#333333", linewidth=0.35, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.4)
    ax.set_ylabel(ylabel, fontsize=6.5)
    ax.grid(axis="y", color="#dddddd", linewidth=0.45, alpha=0.75)
    ax.tick_params(axis="y", labelsize=6.0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def show_he_with_spots(ax: plt.Axes, sample: str, coords_norm: np.ndarray, title: str) -> None:
    image_path = SPATIAL_ROOT / sample / "spatial" / "tissue_hires_image.png"
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((900, 900), Image.Resampling.LANCZOS)
    arr = np.asarray(image)
    ax.imshow(arr)
    h, w = arr.shape[:2]
    x = np.clip(coords_norm[:, 0], 0, 1) * w
    y = (1 - np.clip(coords_norm[:, 1], 0, 1)) * h
    ax.scatter(x, y, s=2.0, facecolor="none", edgecolor="#3f3f3f", linewidth=0.12, alpha=0.28)
    ax.set_title(title, fontsize=7.1, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_spot_field(
    ax: plt.Axes,
    sample: str,
    coords_norm: np.ndarray,
    arr: np.ndarray,
    mask: np.ndarray,
    title: str,
    cmap: str,
    *,
    log: bool = False,
) -> None:
    image_path = SPATIAL_ROOT / sample / "spatial" / "tissue_hires_image.png"
    image = Image.open(image_path).convert("L")
    image.thumbnail((900, 900), Image.Resampling.LANCZOS)
    gray = np.asarray(image)
    ax.imshow(gray, cmap="gray", alpha=0.34)
    h, w = gray.shape[:2]
    x = np.clip(coords_norm[:, 0], 0, 1) * w
    y = (1 - np.clip(coords_norm[:, 1], 0, 1)) * h
    panel = normalized_panel(arr, mask, log=log)
    gx = np.clip(np.rint(coords_norm[:, 0] * (arr.shape[1] - 1)).astype(int), 0, arr.shape[1] - 1)
    gy = np.clip(np.rint(coords_norm[:, 1] * (arr.shape[0] - 1)).astype(int), 0, arr.shape[0] - 1)
    vals = panel[gy, gx]
    valid = np.isfinite(vals)
    ax.scatter(x[valid], y[valid], s=6.8, c=vals[valid], cmap=cmap, vmin=0, vmax=1, edgecolor="none", alpha=0.92)
    ax.set_title(title, fontsize=7.1, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def fig4a_spatial_prior_comparison() -> None:
    bright = HISTO_REP_ROOT / "brightness"
    hema = HISTO_REP_ROOT / "hematoxylin"
    mask = np.load(bright / "tissue_mask.npy") > 0
    coords = np.load(bright / "coords_norm.npy")
    source = np.load(bright / "source_grid.npy")
    barrier = np.load(bright / "barrier_grid.npy")
    bright_res = np.load(bright / "histology_resistance_grid.npy")
    hema_res = np.load(hema / "histology_resistance_grid.npy")
    bright_field = np.load(HISTO_REP_ROOT / "brightness_pinn" / "pinn_grid_prediction_postprocessed.npy")
    hema_field = np.load(HISTO_REP_ROOT / "hematoxylin_pinn" / "pinn_grid_prediction_postprocessed.npy")
    delta = hema_field - bright_field

    fig = plt.figure(figsize=(8.9, 3.15))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.08, hspace=0.22)
    axes = [fig.add_subplot(gs[i // 4, i % 4]) for i in range(8)]
    panel_label(axes[0], "A", x=-0.18, y=1.12)
    show_he_with_spots(axes[0], HISTO_REP_SAMPLE, coords, "H&E + spots")
    show_grid(axes[1], source, mask, "Apoe source", "magma", display_smooth_sigma=5.0)
    show_grid(axes[2], barrier, mask, "CNS-myelin barrier", "YlOrBr", display_smooth_sigma=5.0)
    show_grid(axes[3], bright_res, mask, "Brightness\nstructural prior", "viridis")
    show_grid(axes[4], hema_res, mask, "Hematoxylin\nstructural prior", "viridis")
    show_grid(axes[5], bright_field, mask, "Brightness-prior\nfield", "plasma")
    show_grid(axes[6], hema_field, mask, "Hematoxylin-prior\nfield", "plasma")
    show_grid(axes[7], delta, mask, "Hematoxylin -\nbrightness field", "coolwarm", diverging=True)
    fig.suptitle("Representative spatial robustness of histology-derived structural priors", fontsize=9, fontweight="bold", y=1.02)
    save_panel(fig, "Fig4A_spatial_histology_prior_comparison")


def fig4b_spatial_profile_sensitivity() -> None:
    preflight = PREFLIGHT_ROOT / LOWPDE_REP_SAMPLE / LOWPDE_REP_TASK
    mask = np.load(preflight / "tissue_mask.npy") > 0
    coords = np.load(preflight / "coords_norm.npy")
    source = np.load(preflight / "source_grid.npy")
    barrier = np.load(preflight / "barrier_grid.npy")
    default = np.load(LOWPDE_REP_ROOT / "seed0" / "default" / "field_gauss07.npy")
    low_pde = np.load(LOWPDE_REP_ROOT / "seed0" / "low_pde" / "field_gauss07.npy")
    delta = low_pde - default
    grad_delta = gradient_magnitude(low_pde, mask) - gradient_magnitude(default, mask)

    fig = plt.figure(figsize=(8.0, 2.65))
    gs = gridspec.GridSpec(1, 6, figure=fig, wspace=0.08)
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]
    panel_label(axes[0], "B", x=-0.20, y=1.11)
    show_grid(axes[0], source, mask, "Apoe source", "magma", display_smooth_sigma=5.0)
    show_grid(axes[1], barrier, mask, "CNS-myelin\nbarrier", "YlOrBr", display_smooth_sigma=5.0)
    show_grid(axes[2], default, mask, "Default\nfield", "plasma")
    show_grid(axes[3], low_pde, mask, "Low-PDE\nfield", "plasma")
    show_grid(axes[4], delta, mask, "Low-PDE -\ndefault", "coolwarm", diverging=True)
    show_grid(axes[5], grad_delta, mask, "Gradient\nchange", "coolwarm", diverging=True)
    fig.suptitle("Spatial profile sensitivity on a seed-paired section", fontsize=9, fontweight="bold", y=1.04)
    save_panel(fig, "Fig4B_spatial_low_pde_profile_sensitivity")


def fig4c_targeted_extension_multigene() -> None:
    gene_df = pd.read_csv(TARGETED_DATASET_GENE).copy()
    examples = [
        ("Brain\nC1qa", "brain_aging_gse193107", "GSM5773457_Old_mouse_brain_A1-2", "C1qa_CNS_Myelin"),
        ("Brain\nTyrobp", "brain_aging_gse193107", "GSM5773457_Old_mouse_brain_A1-2", "Tyrobp_CNS_Myelin"),
        ("Brain\nTrem2", "brain_aging_gse193107", "GSM5773457_Old_mouse_brain_A1-2", "Trem2_CNS_Myelin"),
        ("Brain\nAif1", "brain_aging_gse193107", "GSM5773457_Old_mouse_brain_A1-2", "Aif1_CNS_Myelin"),
    ]

    fig = plt.figure(figsize=(7.85, 2.85))
    gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 1.75], wspace=0.38)
    map_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_brain = fig.add_subplot(gs[0, 4])
    panel_label(map_axes[0], "C", x=-0.20, y=1.10)

    for ax, (title, dataset, sample, task) in zip(map_axes, examples):
        preflight = ROOT / "targeted_gene_extension" / "preflight" / dataset / sample / task / "brightness"
        pinn = ROOT / "targeted_gene_extension" / "pinn" / dataset / sample / task / "fourier_refined_low_pde_16g_gauss07"
        mask = np.load(preflight / "tissue_mask.npy") > 0
        field = np.load(pinn / "pinn_grid_prediction_postprocessed.npy")
        show_grid(ax, field, mask, title, "plasma")

    brain = gene_df[gene_df["dataset"].eq("brain_aging_gse193107")].sort_values("spot_pearson_source_mean", ascending=True)
    y = np.arange(len(brain))
    for yi, row in zip(y, brain.itertuples(index=False)):
        color = PAPER_COLORS["blue"] if row.target_gene in {"Apoe", "Gfap"} else PAPER_COLORS["teal"]
        ax_brain.plot([row.spot_pearson_source_min, row.spot_pearson_source_max], [yi, yi], color=color, linewidth=2.0, alpha=0.25)
        ax_brain.plot([0.55, row.spot_pearson_source_mean], [yi, yi], color=color, linewidth=1.25, alpha=0.78)
        ax_brain.scatter(row.spot_pearson_source_mean, yi, s=24, color=color, edgecolor=PAPER_COLORS["dark"], linewidth=0.35, zorder=3)
    ax_brain.axvline(0.55, color="#777777", linewidth=0.55, linestyle=":")
    ax_brain.set_yticks(y)
    ax_brain.set_yticklabels(brain["target_gene"], fontsize=5.8)
    ax_brain.set_xlim(0.55, 0.90)
    ax_brain.set_xlabel("Source Pearson", fontsize=6.4)
    ax_brain.set_title("Brain aging targeted extension\n8 genes x 8 sections", fontsize=7.3)
    ax_brain.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    ax_brain.tick_params(axis="x", labelsize=5.7)
    fig.suptitle("Primary brain targeted gene extension: added spatial tasks and fitted-source QC", fontsize=8.8, fontweight="bold", y=1.03)
    save_panel(fig, "Fig4C_targeted_extension_multigene_spatial_summary")


def fig4b_low_pde_profile() -> None:
    df = pd.read_csv(LOW_PDE)
    delta = df[df["summary_type"].eq("delta_low_pde_minus_default")].copy()
    delta["label"] = delta["target_gene"] + "\n" + delta["field_type"].map(clean_field)
    preflight = PREFLIGHT_ROOT / LOWPDE_REP_SAMPLE / LOWPDE_REP_TASK
    mask = np.load(preflight / "tissue_mask.npy") > 0
    default = np.load(LOWPDE_REP_ROOT / "seed0" / "default" / "field_gauss07.npy")
    low_pde = np.load(LOWPDE_REP_ROOT / "seed0" / "low_pde" / "field_gauss07.npy")
    spatial_delta = low_pde - default
    bounds = zoom_bounds_from_peak(spatial_delta, mask, window=50)

    fig = plt.figure(figsize=(8.9, 3.05))
    gs = gridspec.GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 1, 0.95, 1.18, 1.18], wspace=0.26, hspace=0.24)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axes.append(fig.add_subplot(gs[0, 3], projection="3d"))
    zoom_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    ax_metric = fig.add_subplot(gs[:, 4])
    ax_trade = fig.add_subplot(gs[:, 5])
    panel_label(axes[0], "D", x=-0.22, y=1.10)
    show_grid(axes[0], default, mask, "Default\nfield", "plasma")
    show_grid(axes[1], low_pde, mask, "Low-PDE\nfield", "plasma")
    show_grid(axes[2], spatial_delta, mask, "Low-PDE -\ndefault", "coolwarm", diverging=True)
    show_surface(axes[3], np.abs(spatial_delta), mask, "3D delta\nheight", "magma")
    show_grid_zoom(zoom_axes[0], default, mask, bounds, "Default zoom", "plasma")
    show_grid_zoom(zoom_axes[1], low_pde, mask, bounds, "Low-PDE zoom", "plasma")
    show_grid_zoom(zoom_axes[2], spatial_delta, mask, bounds, "Delta zoom", "coolwarm", diverging=True)
    zoom_axes[3].axis("off")
    zoom_axes[3].text(
        0.03,
        0.78,
        "Local inset:\nprofile change is\nspatially concentrated.",
        fontsize=6.1,
        color="#35424d",
        ha="left",
        va="top",
    )

    y = np.arange(len(delta))[::-1]
    colors = [PAPER_COLORS["teal"] if v >= 0 else PAPER_COLORS["terracotta"] for v in delta["spot_pearson_source_mean"]]
    ax_metric.axvline(0, color="#333333", linewidth=0.55)
    for yi, value, color in zip(y, delta["spot_pearson_source_mean"], colors):
        ax_metric.plot([0, value], [yi, yi], color=color, linewidth=1.0)
        ax_metric.scatter(value, yi, s=24, color=color, edgecolor="#333333", linewidth=0.35)
    ax_metric.set_yticks(y)
    ax_metric.set_yticklabels(delta["label"], fontsize=6.2)
    ax_metric.set_xlabel("Delta Pearson", fontsize=6.7)
    ax_metric.set_title("Source-fit gain", fontsize=7.4)
    ax_metric.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    ax_metric.tick_params(axis="y", pad=1)

    ax_trade.axvline(0, color="#333333", linewidth=0.55)
    for yi, mse, rough in zip(y, delta["spot_mse_source_mean"], delta["roughness_grad_p95_mean"]):
        ax_trade.scatter(mse, yi + 0.12, s=22, color=PAPER_COLORS["slate"], edgecolor=PAPER_COLORS["dark"], linewidth=0.30, label="Delta MSE" if yi == y[0] else None)
        ax_trade.scatter(rough, yi - 0.12, s=22, color=PAPER_COLORS["ochre"], edgecolor=PAPER_COLORS["dark"], linewidth=0.30, label="Delta rough" if yi == y[0] else None)
    ax_trade.set_yticks(y)
    ax_trade.set_yticklabels([])
    ax_trade.set_xlabel("Delta value", fontsize=6.7)
    ax_trade.set_title("Trade-off", fontsize=7.4)
    ax_trade.legend(frameon=False, fontsize=5.8, loc="upper right", bbox_to_anchor=(1.02, 1.02), borderaxespad=0)
    ax_trade.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    fig.suptitle("Low-PDE profile: local spatial differences with compact metric deltas", fontsize=8.8, fontweight="bold", y=1.03)
    save_panel(fig, "Fig4D_low_pde_profile_delta")


def fig4c_source_clipping() -> None:
    df = pd.read_csv(CLIP)
    fig, axes = plt.subplots(2, 1, figsize=(2.55, 5.15), sharex=True, sharey=False)
    fig.subplots_adjust(left=0.27, right=0.97, top=0.88, bottom=0.12, hspace=0.38)
    for ax, gene in zip(axes, ["Apoe", "Gfap"]):
        sub = df[df["target_gene"].eq(gene)]
        for field_type in ["masked", "gauss07"]:
            line = sub[sub["field_type"].eq(field_type)].sort_values("train_source_percentile")
            ax.plot(
                line["train_source_percentile"],
                line["spot_pearson_source"],
                marker="o",
                markersize=3.2,
                linewidth=1.1,
                color=COLORS[field_type],
                label=clean_field(field_type),
            )
        y_values = sub["spot_pearson_source"].to_numpy(dtype=float)
        y_pad = max((np.nanmax(y_values) - np.nanmin(y_values)) * 0.20, 0.001)
        ax.set_ylim(np.nanmin(y_values) - y_pad, np.nanmax(y_values) + y_pad)
        ax.set_title(gene, fontsize=8.1, loc="left", pad=2)
        ax.set_ylabel("Spot-source\nPearson", fontsize=6.8)
        ax.grid(color="#dddddd", linewidth=0.45, alpha=0.75)
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        ax.tick_params(axis="both", labelsize=6.4, pad=1)
    axes[-1].set_xticks([95, 97.5, 99, 99.5])
    axes[-1].set_xticklabels(["95", "97.5", "99", "99.5"])
    axes[-1].set_xlabel("Training source\nclipping percentile", fontsize=6.8)
    axes[0].legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.32),
        ncol=2,
        fontsize=6.2,
        handlelength=1.7,
        columnspacing=0.9,
    )
    fig.suptitle("Source clipping sensitivity", x=0.27, y=0.99, ha="left", fontsize=8.4, fontweight="bold")
    panel_label(axes[0], "E")
    save_panel(fig, "Fig4E_source_clipping_sensitivity")


def fig4d_seed_stability() -> None:
    df = pd.read_csv(SEED)
    df["label"] = df["target_gene"] + "\n" + df["field_type"].map(clean_field)
    preflight = PREFLIGHT_ROOT / LOWPDE_REP_SAMPLE / LOWPDE_REP_TASK
    mask = np.load(preflight / "tissue_mask.npy") > 0
    seed_deltas = []
    for seed in [0, 1, 2]:
        default = np.load(LOWPDE_REP_ROOT / f"seed{seed}" / "default" / "field_gauss07.npy")
        low_pde = np.load(LOWPDE_REP_ROOT / f"seed{seed}" / "low_pde" / "field_gauss07.npy")
        seed_deltas.append(low_pde - default)

    fig = plt.figure(figsize=(8.4, 2.55))
    gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1.25, 1.25], wspace=0.25)
    map_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_p = fig.add_subplot(gs[0, 3])
    ax_r = fig.add_subplot(gs[0, 4])
    panel_label(map_axes[0], "F", x=-0.20, y=1.10)
    for ax, delta_map, seed in zip(map_axes, seed_deltas, [0, 1, 2]):
        show_grid(ax, delta_map, mask, f"Seed {seed}\nlow-PDE - default", "coolwarm", diverging=True)
    y = np.arange(len(df))[::-1]
    for yi, mean, sd in zip(y, df["source_pearson_delta_mean"], df["source_pearson_delta_sd"]):
        ax_p.plot([mean - sd, mean + sd], [yi, yi], color=PAPER_COLORS["teal"], linewidth=1.2)
        ax_p.scatter(mean, yi, s=26, color=PAPER_COLORS["teal"], edgecolor=PAPER_COLORS["dark"], linewidth=0.35)
    ax_p.axvline(0, color="#333333", linewidth=0.55)
    ax_p.set_yticks(y)
    ax_p.set_yticklabels(df["label"], fontsize=6.3)
    ax_p.set_xlabel("Mean delta Pearson", fontsize=6.7)
    ax_p.set_title("Seed-paired gain", fontsize=7.5)
    ax_p.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    for yi, mean, sd in zip(y, df["roughness_p95_delta_mean"], df["roughness_p95_delta_sd"]):
        ax_r.plot([mean - sd, mean + sd], [yi, yi], color=PAPER_COLORS["ochre"], linewidth=1.2)
        ax_r.scatter(mean, yi, s=26, color=PAPER_COLORS["ochre"], edgecolor=PAPER_COLORS["dark"], linewidth=0.35)
    ax_r.axvline(0, color="#333333", linewidth=0.55)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([])
    ax_r.set_xlabel("Mean delta gradient p95", fontsize=6.7)
    ax_r.set_title("Smoothness cost", fontsize=7.5)
    ax_r.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    fig.suptitle("Seed stability: repeated local field changes plus compact uncertainty", fontsize=8.8, fontweight="bold", y=1.03)
    save_panel(fig, "Fig4F_seed_stability")


def fig4g_targeted_extension_spatial_metrics() -> None:
    full = pd.read_csv(TARGETED_FULL).copy()
    brain = full[full["dataset"].eq("brain_aging_gse193107")].copy()
    sample_order = list(dict.fromkeys(brain["sample"]))
    sample_summary = (
        brain.groupby("sample", as_index=False)
        .agg(
            n_genes=("target_gene", "nunique"),
            source_mean=("spot_pearson_source", "mean"),
            source_min=("spot_pearson_source", "min"),
            source_max=("spot_pearson_source", "max"),
            rough_mean=("roughness_grad_p95", "mean"),
            rough_min=("roughness_grad_p95", "min"),
            rough_max=("roughness_grad_p95", "max"),
        )
        .set_index("sample")
        .loc[sample_order]
        .reset_index()
    )
    sample_summary["label"] = [brain_sample_label(sample) for sample in sample_summary["sample"]]
    examples = [
        ("Old A1\nApoe", "GSM5773457_Old_mouse_brain_A1-2", "Apoe_CNS_Myelin"),
        ("Old A1\nGfap", "GSM5773457_Old_mouse_brain_A1-2", "Gfap_CNS_Myelin"),
        ("Old A1\nCst3", "GSM5773457_Old_mouse_brain_A1-2", "Cst3_CNS_Myelin"),
        ("Old A1\nLpl", "GSM5773457_Old_mouse_brain_A1-2", "Lpl_CNS_Myelin"),
    ]

    fig = plt.figure(figsize=(9.85, 2.62))
    gs = gridspec.GridSpec(1, 6, figure=fig, width_ratios=[1, 1, 1, 0.95, 1.58, 1.22], wspace=0.58)
    map_axes = [fig.add_subplot(gs[0, i]) for i in range(4)]
    ax_p = fig.add_subplot(gs[0, 4])
    ax_r = fig.add_subplot(gs[0, 5])
    panel_label(map_axes[0], "G", x=-0.20, y=1.10)

    for ax, (title, sample, task) in zip(map_axes, examples):
        preflight = ROOT / "targeted_gene_extension" / "preflight" / "brain_aging_gse193107" / sample / task / "brightness"
        pinn = ROOT / "targeted_gene_extension" / "pinn" / "brain_aging_gse193107" / sample / task / "fourier_refined_low_pde_16g_gauss07"
        mask = np.load(preflight / "tissue_mask.npy") > 0
        field = np.load(pinn / "pinn_grid_prediction_postprocessed.npy")
        show_grid(ax, field, mask, title, "plasma")

    y = np.arange(len(sample_summary))[::-1]
    for yi, row in zip(y, sample_summary.itertuples(index=False)):
        color = PAPER_COLORS["blue"] if "Young" in row.label else PAPER_COLORS["teal"]
        ax_p.plot([row.source_min, row.source_max], [yi, yi], color=color, linewidth=2.0, alpha=0.24)
        ax_p.plot([0.55, row.source_mean], [yi, yi], color=color, linewidth=1.15, alpha=0.78)
        ax_p.scatter(row.source_mean, yi, s=26, color=color, edgecolor=PAPER_COLORS["dark"], linewidth=0.35, zorder=3)
        ax_r.plot([row.rough_min, row.rough_max], [yi, yi], color=PAPER_COLORS["ochre"], linewidth=2.0, alpha=0.26)
        ax_r.plot([0, row.rough_mean], [yi, yi], color=PAPER_COLORS["ochre"], linewidth=1.15, alpha=0.80)
        ax_r.scatter(row.rough_mean, yi, s=26, color=PAPER_COLORS["ochre"], edgecolor=PAPER_COLORS["dark"], linewidth=0.35, zorder=3)

    ax_p.axvline(0.55, color="#777777", linewidth=0.55, linestyle=":")
    ax_p.set_xlim(0.55, 0.84)
    ax_p.set_yticks(y)
    ax_p.set_yticklabels(sample_summary["label"], fontsize=5.3)
    ax_p.set_xlabel("Source Pearson", fontsize=6.4)
    ax_p.set_title("Section source fidelity\n8 genes per section", fontsize=7.5)
    ax_p.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    ax_p.tick_params(axis="x", labelsize=5.7)

    ax_r.set_xlim(0, 0.34)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([])
    ax_r.set_xlabel("Grad. p95", fontsize=6.4)
    ax_r.set_title("Section roughness\nmean and range", fontsize=7.5)
    ax_r.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    ax_r.tick_params(axis="x", labelsize=5.7)
    fig.suptitle("Primary brain targeted extension: aligned representative fields and section-level QC", fontsize=8.6, fontweight="bold", y=1.03)
    save_panel(fig, "Fig4G_targeted_extension_spatial_metrics")


def fig4f_profile_decision() -> None:
    rows = [
        ["Default profile", "Primary manuscript", "Conservative baseline;\nlowest interpretive risk"],
        ["Low-PDE profile", "Sensitivity/support", "Improves source fidelity;\nsmall smoothness cost for Gfap"],
        ["Histology prior swap", "Robustness check", "Brightness and hematoxylin\nproduce similar summaries"],
        ["Source clipping", "Preprocessing check", "Stable across 95-99.5%\nsource caps"],
    ]
    fig, ax = plt.subplots(figsize=(6.8, 1.82))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Component", "Figure role", "Interpretation"],
        cellLoc="left",
        colLoc="left",
        loc="upper center",
        bbox=[0.0, 0.02, 1.0, 0.78],
        colWidths=[0.24, 0.24, 0.52],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.4)
    table.scale(1, 1.08)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fa")
    ax.set_title("Recommended robustness narrative", fontsize=8.0, loc="left", pad=1)
    save_panel(fig, "Fig4H_profile_decision_table")


def rough_assembly() -> None:
    stems = [
        "Fig4A_spatial_histology_prior_comparison",
        "Fig4B_spatial_low_pde_profile_sensitivity",
        "Fig4C_targeted_extension_multigene_spatial_summary",
        "Fig4D_low_pde_profile_delta",
        "Fig4E_source_clipping_sensitivity",
        "Fig4F_seed_stability",
        "Fig4G_targeted_extension_spatial_metrics",
        "Fig4H_profile_decision_table",
    ]
    imgs = [Image.open(OUT_DIR / f"{stem}.png").convert("RGB") for stem in stems]
    thumb_w = 1750
    thumbs = []
    for img in imgs:
        scale = thumb_w / img.width
        thumbs.append(img.resize((thumb_w, int(img.height * scale)), Image.Resampling.LANCZOS))
    gutter = 110
    cols = 2
    rows = 4
    row_heights = [max(thumbs[i * cols + j].height for j in range(cols)) for i in range(rows)]
    canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * gutter, sum(row_heights) + (rows + 1) * gutter), "white")
    y = gutter
    for r in range(rows):
        x = gutter
        for c in range(cols):
            idx = r * cols + c
            canvas.paste(thumbs[idx], (x, y))
            x += thumb_w + gutter
        y += row_heights[r] + gutter
    canvas.save(OUT_DIR / "Figure4_rough_assembly.png", dpi=(EXPORT_DPI, EXPORT_DPI))
    canvas.save(OUT_DIR / "Figure4_rough_assembly.pdf", resolution=EXPORT_DPI)


def main() -> None:
    setup()
    ensure_dir(OUT_DIR)
    fig4a_spatial_prior_comparison()
    fig4b_spatial_profile_sensitivity()
    fig4c_targeted_extension_multigene()
    fig4b_low_pde_profile()
    fig4c_source_clipping()
    fig4d_seed_stability()
    fig4g_targeted_extension_spatial_metrics()
    fig4f_profile_decision()
    rough_assembly()
    print(f"Wrote Figure 4 panel assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
