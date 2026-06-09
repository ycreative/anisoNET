"""Generate separated Figure 4 panel assets for robustness and resource evidence."""

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
OUT_DIR = ROOT / "manuscript_figures" / "Figure4_robustness_reproducibility"

SPATIAL_ROOT = ROOT / "processed_visium" / "brain_aging_gse193107"
PREFLIGHT_ROOT = ROOT / "preflight" / "brain_aging_gse193107"
HISTO = ROOT / "histology_prior" / "brain_aging_gse193107" / "histology_prior_pinn_group_summary.csv"
LOW_PDE = ROOT / "loss_weight_sensitivity" / "brain_aging_gse193107" / "multi_section_low_pde" / "low_pde_profile_validation_group_summary.csv"
SEED = ROOT / "loss_weight_sensitivity" / "brain_aging_gse193107" / "low_pde_seed_stability" / "low_pde_seed_stability_paired_summary.csv"
CLIP = ROOT / "source_clipping_sensitivity" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "source_clipping_sensitivity_summary.csv"
RESOURCE = ROOT / "resource_profile" / "brain_aging_gse193107" / "GSM5773457_Old_mouse_brain_A1-2" / "anisonet_resource_profile_comparison.csv"

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


COLORS = {
    "Apoe": "#3f7f93",
    "Gfap": "#b35f5f",
    "brightness": "#4f7cac",
    "hematoxylin": "#9b6a9e",
    "masked": "#6a994e",
    "gauss07": "#c58c2b",
    "default": "#7d8597",
    "low_pde": "#2a9d8f",
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
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str, x: float = -0.13, y: float = 1.10) -> None:
    return None


def clean_field(value: str) -> str:
    return {"gauss07": "post", "masked": "mask"}.get(value, value)


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
        panel = crop_to_mask(normalized_panel(arr, mask, log=log), mask)
        ax.imshow(panel, cmap=cmap, interpolation="bilinear", origin="lower")
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
    color_left: str = "#4f7cac",
    color_right: str = "#9b6a9e",
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
    show_grid(axes[1], source, mask, "Apoe source", "magma")
    show_grid(axes[2], barrier, mask, "CNS-myelin barrier", "YlOrBr")
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
    show_grid(axes[0], source, mask, "Apoe source", "magma")
    show_grid(axes[1], barrier, mask, "CNS-myelin\nbarrier", "YlOrBr")
    show_grid(axes[2], default, mask, "Default\nfield", "plasma")
    show_grid(axes[3], low_pde, mask, "Low-PDE\nfield", "plasma")
    show_grid(axes[4], delta, mask, "Low-PDE -\ndefault", "coolwarm", diverging=True)
    show_grid(axes[5], grad_delta, mask, "Gradient\nchange", "coolwarm", diverging=True)
    fig.suptitle("Spatial profile sensitivity on a seed-paired section", fontsize=9, fontweight="bold", y=1.04)
    save_panel(fig, "Fig4B_spatial_low_pde_profile_sensitivity")


def fig4c_histology_prior() -> None:
    df = pd.read_csv(HISTO)
    summary = df[df["field_type"].eq("gauss07")].copy()
    bright = HISTO_REP_ROOT / "brightness"
    mask = np.load(bright / "tissue_mask.npy") > 0
    bright_field = np.load(HISTO_REP_ROOT / "brightness_pinn" / "pinn_grid_prediction_postprocessed.npy")
    hema_field = np.load(HISTO_REP_ROOT / "hematoxylin_pinn" / "pinn_grid_prediction_postprocessed.npy")
    delta = hema_field - bright_field

    fig = plt.figure(figsize=(8.7, 2.55))
    gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[1.05, 1.05, 1.05, 1.25, 1.25], wspace=0.22)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_p = fig.add_subplot(gs[0, 3])
    ax_r = fig.add_subplot(gs[0, 4])
    panel_label(axes[0], "C", x=-0.20, y=1.10)
    show_grid(axes[0], bright_field, mask, "Brightness-prior\nfield", "plasma")
    show_grid(axes[1], hema_field, mask, "Hematoxylin-prior\nfield", "plasma")
    show_grid(axes[2], delta, mask, "Field delta", "coolwarm", diverging=True)

    genes = ["Apoe", "Gfap"]
    b = summary[summary["histology_prior"].eq("brightness")].set_index("target_gene").loc[genes]
    h = summary[summary["histology_prior"].eq("hematoxylin")].set_index("target_gene").loc[genes]
    compact_slope_axis(
        ax_p,
        genes,
        b["spot_pearson_source_mean"].tolist(),
        h["spot_pearson_source_mean"].tolist(),
        "Bright",
        "Hema",
        "Pearson",
    )
    compact_slope_axis(
        ax_r,
        genes,
        b["roughness_grad_p95_mean"].tolist(),
        h["roughness_grad_p95_mean"].tolist(),
        "Bright",
        "Hema",
        "Gradient p95",
    )
    ax_p.set_title("Source fidelity", fontsize=7.5)
    ax_r.set_title("Roughness", fontsize=7.5)
    handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#4f7cac", markeredgecolor="#333333", markersize=4.5),
        mpl.lines.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#9b6a9e", markeredgecolor="#333333", markersize=4.5),
    ]
    fig.legend(handles, ["Brightness", "Hematoxylin"], frameon=False, ncol=2, fontsize=6.4, loc="upper center", bbox_to_anchor=(0.71, 0.93))
    fig.suptitle("Histology-prior robustness: spatial fields first, compact metrics second", fontsize=8.8, fontweight="bold", y=1.04)
    save_panel(fig, "Fig4C_histology_prior_metric_summary")


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

    fig = plt.figure(figsize=(8.8, 2.85))
    gs = gridspec.GridSpec(2, 6, figure=fig, width_ratios=[1, 1, 1, 1, 1.35, 1.35], wspace=0.18, hspace=0.18)
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
        0.0,
        0.72,
        "Local inset highlights\nwhere the profile change\nis spatially concentrated.",
        fontsize=6.8,
        color="#35424d",
        ha="left",
        va="top",
    )

    y = np.arange(len(delta))[::-1]
    colors = ["#2a9d8f" if v >= 0 else "#b56576" for v in delta["spot_pearson_source_mean"]]
    ax_metric.axvline(0, color="#333333", linewidth=0.55)
    for yi, value, color in zip(y, delta["spot_pearson_source_mean"], colors):
        ax_metric.plot([0, value], [yi, yi], color=color, linewidth=1.0)
        ax_metric.scatter(value, yi, s=24, color=color, edgecolor="#333333", linewidth=0.35)
    ax_metric.set_yticks(y)
    ax_metric.set_yticklabels(delta["label"], fontsize=6.2)
    ax_metric.set_xlabel("Delta Pearson", fontsize=6.7)
    ax_metric.set_title("Source-fit gain", fontsize=7.4)
    ax_metric.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)

    ax_trade.axvline(0, color="#333333", linewidth=0.55)
    for yi, mse, rough in zip(y, delta["spot_mse_source_mean"], delta["roughness_grad_p95_mean"]):
        ax_trade.scatter(mse, yi + 0.12, s=22, color="#7d8597", edgecolor="#333333", linewidth=0.30, label="Delta MSE" if yi == y[0] else None)
        ax_trade.scatter(rough, yi - 0.12, s=22, color="#c58c2b", edgecolor="#333333", linewidth=0.30, label="Delta rough" if yi == y[0] else None)
    ax_trade.set_yticks(y)
    ax_trade.set_yticklabels([])
    ax_trade.set_xlabel("Delta value", fontsize=6.7)
    ax_trade.set_title("Trade-off", fontsize=7.4)
    ax_trade.legend(frameon=False, fontsize=6.0, loc="lower right")
    ax_trade.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    fig.suptitle("Low-PDE profile: local spatial differences with compact metric deltas", fontsize=8.8, fontweight="bold", y=1.03)
    save_panel(fig, "Fig4D_low_pde_profile_delta")


def fig4c_source_clipping() -> None:
    df = pd.read_csv(CLIP)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.55), sharey=False)
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
        ax.set_title(gene, fontsize=8.2)
        ax.set_xlabel("Training source clipping percentile")
        ax.set_ylabel("Spot-source Pearson")
        ax.grid(color="#dddddd", linewidth=0.45, alpha=0.75)
    axes[0].legend(frameon=False, loc="lower left")
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
        ax_p.plot([mean - sd, mean + sd], [yi, yi], color="#2a9d8f", linewidth=1.2)
        ax_p.scatter(mean, yi, s=26, color="#2a9d8f", edgecolor="#333333", linewidth=0.35)
    ax_p.axvline(0, color="#333333", linewidth=0.55)
    ax_p.set_yticks(y)
    ax_p.set_yticklabels(df["label"], fontsize=6.3)
    ax_p.set_xlabel("Mean delta Pearson", fontsize=6.7)
    ax_p.set_title("Seed-paired gain", fontsize=7.5)
    ax_p.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    for yi, mean, sd in zip(y, df["roughness_p95_delta_mean"], df["roughness_p95_delta_sd"]):
        ax_r.plot([mean - sd, mean + sd], [yi, yi], color="#c58c2b", linewidth=1.2)
        ax_r.scatter(mean, yi, s=26, color="#c58c2b", edgecolor="#333333", linewidth=0.35)
    ax_r.axvline(0, color="#333333", linewidth=0.55)
    ax_r.set_yticks(y)
    ax_r.set_yticklabels([])
    ax_r.set_xlabel("Mean delta gradient p95", fontsize=6.7)
    ax_r.set_title("Smoothness cost", fontsize=7.5)
    ax_r.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
    fig.suptitle("Seed stability: repeated local field changes plus compact uncertainty", fontsize=8.8, fontweight="bold", y=1.03)
    save_panel(fig, "Fig4F_seed_stability")


def fig4e_resource_profile() -> None:
    df = pd.read_csv(RESOURCE)
    labels = df["profile"].str.replace("fourier_refined_", "", regex=False).str.replace("_16g", "", regex=False)
    x = np.arange(len(df))
    preflight = PREFLIGHT_ROOT / HISTO_REP_SAMPLE / HISTO_REP_TASK
    mask = np.load(preflight / "tissue_mask.npy") > 0
    coords = np.load(preflight / "coords_norm.npy")
    source = np.load(preflight / "source_grid.npy")

    fig = plt.figure(figsize=(8.2, 2.45))
    gs = gridspec.GridSpec(1, 5, figure=fig, width_ratios=[1.05, 1.05, 1.2, 1.15, 1.15], wspace=0.30)
    ax_he = fig.add_subplot(gs[0, 0])
    ax_source = fig.add_subplot(gs[0, 1])
    ax_surface = fig.add_subplot(gs[0, 2], projection="3d")
    ax_time = fig.add_subplot(gs[0, 3])
    ax_mem = fig.add_subplot(gs[0, 4])
    panel_label(ax_he, "G", x=-0.22, y=1.10)
    show_he_with_spots(ax_he, HISTO_REP_SAMPLE, coords, "Profiled\nsection")
    show_grid(ax_source, source, mask, "Apoe source\nfield object", "magma")
    show_surface(ax_surface, source, mask, "3D source\nheight")

    colors = ["#7d8597", "#2a9d8f"]
    for ax, metric, title, xlabel in [
        (ax_time, "mean_elapsed_seconds", "Runtime", "Seconds"),
        (ax_mem, "mean_peak_cuda_reserved_gb", "Memory", "CUDA GB"),
    ]:
        y = np.arange(len(df))[::-1]
        vals = df[metric].to_numpy()
        ax.barh(y, vals, height=0.36, color=colors, edgecolor="#333333", linewidth=0.35)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.2)
        ax.set_xlabel(xlabel, fontsize=6.7)
        ax.set_title(title, fontsize=7.5)
        ax.grid(axis="x", color="#dddddd", linewidth=0.45, alpha=0.75)
        ax.tick_params(axis="x", labelsize=6.2)
    fig.suptitle("Resource profile tied to the representative spatial inference object", fontsize=8.8, fontweight="bold", y=1.03)
    save_panel(fig, "Fig4G_resource_profile")


def fig4f_profile_decision() -> None:
    rows = [
        ["Default profile", "Primary manuscript", "Conservative baseline;\nlowest interpretive risk"],
        ["Low-PDE profile", "Sensitivity/support", "Improves source fidelity;\nsmall smoothness cost for Gfap"],
        ["Histology prior swap", "Robustness check", "Brightness and hematoxylin\nproduce similar summaries"],
        ["Source clipping", "Preprocessing check", "Stable across 95-99.5%\nsource caps"],
    ]
    fig, ax = plt.subplots(figsize=(7.4, 2.7))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Component", "Figure role", "Interpretation"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.25, 0.24, 0.51],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.42)
    for (row, col), cell in table.get_celld().items():
        cell.set_linewidth(0.45)
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#e9ecef")
            cell.set_text_props(weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8f9fa")
    ax.text(-0.04, 1.05, "H", transform=ax.transAxes, fontsize=12, fontweight="bold", ha="left", va="top")
    ax.set_title("Recommended robustness narrative", fontsize=8.6, loc="left", pad=10)
    save_panel(fig, "Fig4H_profile_decision_table")


def rough_assembly() -> None:
    stems = [
        "Fig4A_spatial_histology_prior_comparison",
        "Fig4B_spatial_low_pde_profile_sensitivity",
        "Fig4C_histology_prior_metric_summary",
        "Fig4D_low_pde_profile_delta",
        "Fig4E_source_clipping_sensitivity",
        "Fig4F_seed_stability",
        "Fig4G_resource_profile",
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
    canvas.save(OUT_DIR / "Figure4_rough_assembly.png", dpi=(300, 300))
    canvas.save(OUT_DIR / "Figure4_rough_assembly.pdf", resolution=300)


def main() -> None:
    setup()
    ensure_dir(OUT_DIR)
    fig4a_spatial_prior_comparison()
    fig4b_spatial_profile_sensitivity()
    fig4c_histology_prior()
    fig4b_low_pde_profile()
    fig4c_source_clipping()
    fig4d_seed_stability()
    fig4e_resource_profile()
    fig4f_profile_decision()
    rough_assembly()
    print(f"Wrote Figure 4 panel assets to {OUT_DIR}")


if __name__ == "__main__":
    main()
