"""Generate separated Figure 1 panel assets for the GPB revision."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Polygon, Rectangle
from PIL import Image


PROJECT_ROOT = Path(os.environ.get("ANISONET_PROJECT_ROOT", Path(__file__).resolve().parents[2]))
ROOT = Path(os.environ.get("ANISONET_ANALYSIS_ROOT", PROJECT_ROOT / "codexAnalysis"))
OUT_DIR = ROOT / "manuscript_figures" / "Figure1_method_overview"
SOURCE_ASSET_DIR = OUT_DIR / "source_assets"
D_SCHEMATIC_ASSET = "Fig1D_barrier_constrained_schematic_v2.png"

SAMPLE = "GSM5773457_Old_mouse_brain_A1-2"
TASK = "Apoe_CNS_Myelin"
PINN_RUN = "fourier_refined_16g_gauss07_batch"

PREFLIGHT_DIR = ROOT / "preflight" / "brain_aging_gse193107" / SAMPLE / TASK
PINN_DIR = ROOT / "pinn" / "brain_aging_gse193107" / SAMPLE / TASK / PINN_RUN
SPATIAL_DIR = ROOT / "processed_visium" / "brain_aging_gse193107" / SAMPLE / "spatial"
BATCH_METRICS = ROOT / "batch" / "brain_aging_gse193107" / TASK / "batch_metrics_summary.csv"


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


def panel_label(ax: plt.Axes, label: str, x: float = -0.08, y: float = 1.08) -> None:
    return None


def load_array(name: str) -> np.ndarray:
    return np.asarray(np.load(PREFLIGHT_DIR / name), dtype=float)


def load_inputs() -> dict[str, np.ndarray]:
    return {
        "source": load_array("source_grid.npy"),
        "barrier": load_array("barrier_grid.npy"),
        "diffusion": load_array("diffusion_grid.npy"),
        "resistance": load_array("resistance_grid.npy"),
        "mask": load_array("tissue_mask.npy") > 0,
        "coords": load_array("coords_norm.npy"),
        "field_raw": np.asarray(np.load(PINN_DIR / "pinn_grid_prediction_norm.npy"), dtype=float),
        "field_masked": np.asarray(np.load(PINN_DIR / "pinn_grid_prediction_clean_tissue_masked.npy"), dtype=float),
        "field_post": np.asarray(np.load(PINN_DIR / "pinn_grid_prediction_postprocessed.npy"), dtype=float),
    }


def norm_panel(arr: np.ndarray, mask: np.ndarray | None = None, *, log: bool = False) -> np.ndarray:
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


def crop_slices(mask: np.ndarray, pad: int = 6) -> tuple[slice, slice]:
    yy, xx = np.where(mask)
    if yy.size == 0:
        return slice(0, mask.shape[0]), slice(0, mask.shape[1])
    y0 = max(int(yy.min()) - pad, 0)
    y1 = min(int(yy.max()) + pad + 1, mask.shape[0])
    x0 = max(int(xx.min()) - pad, 0)
    x1 = min(int(xx.max()) + pad + 1, mask.shape[1])
    return slice(y0, y1), slice(x0, x1)


def crop_to_mask(arr: np.ndarray, mask: np.ndarray, pad: int = 6) -> np.ndarray:
    ys, xs = crop_slices(mask, pad=pad)
    return arr[ys, xs]


def show_grid(
    ax: plt.Axes,
    arr: np.ndarray,
    mask: np.ndarray,
    title: str,
    cmap: str,
    *,
    log: bool = False,
    rectangle: tuple[int, int, int, int] | None = None,
    title_fontsize: float = 7.2,
    title_pad: float = 2,
) -> None:
    panel = crop_to_mask(norm_panel(arr, mask, log=log), mask)
    ax.imshow(panel, cmap=cmap, interpolation="bilinear", origin="lower")
    if rectangle is not None:
        ys, xs = crop_slices(mask)
        y0, y1, x0, x1 = rectangle
        ax.add_patch(
            Rectangle(
                (x0 - xs.start, y0 - ys.start),
                x1 - x0,
                y1 - y0,
                fill=False,
                edgecolor="#34a853",
                linewidth=1.1,
            )
        )
    ax.set_title(title, fontsize=title_fontsize, pad=title_pad)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def high_score_bounds(data: dict[str, np.ndarray], window: int = 42) -> tuple[int, int, int, int]:
    mask = data["mask"]
    field = norm_panel(data["field_post"], mask)
    barrier = norm_panel(data["barrier"], mask)
    source = norm_panel(data["source"], mask)
    gy, gx = np.gradient(np.nan_to_num(barrier, nan=0.0))
    edge = np.sqrt(gx * gx + gy * gy)
    yy, xx = np.indices(mask.shape)
    tissue_y, tissue_x = np.where(mask)
    if tissue_y.size:
        y_min, y_max = tissue_y.min(), tissue_y.max()
        x_min, x_max = tissue_x.min(), tissue_x.max()
        dist_to_box = np.minimum.reduce([yy - y_min, y_max - yy, xx - x_min, x_max - xx])
        interior = np.clip(dist_to_box / float(window), 0, 1)
    else:
        interior = np.ones_like(field)
    score = np.where(mask, (0.45 * field + 0.35 * source + 0.20 * barrier) * (0.4 + edge) * (0.25 + interior), np.nan)
    if np.all(~np.isfinite(score)):
        yy, xx = np.where(mask)
        cy = int(np.mean(yy)) if yy.size else field.shape[0] // 2
        cx = int(np.mean(xx)) if xx.size else field.shape[1] // 2
    else:
        cy, cx = np.unravel_index(np.nanargmax(score), score.shape)
    half = window // 2
    y0 = max(int(cy) - half, 0)
    y1 = min(y0 + window, field.shape[0])
    x0 = max(int(cx) - half, 0)
    x1 = min(x0 + window, field.shape[1])
    return y0, y1, x0, x1


def weighted_region_ellipse(
    arr: np.ndarray,
    mask: np.ndarray,
    image_width: int,
    image_height: int,
    *,
    percentile: float,
    color: str,
    label: str,
) -> Ellipse:
    data = norm_panel(arr, mask)
    valid = mask & np.isfinite(data)
    threshold = np.nanpercentile(data[valid], percentile) if np.any(valid) else 1.0
    pick = valid & (data >= threshold)
    if np.count_nonzero(pick) < 5:
        pick = valid
    y, x = np.where(pick)
    weights = data[pick]
    if weights.size == 0 or np.nansum(weights) <= 0:
        cy, cx = image_height / 2, image_width / 2
        width, height, angle = image_width * 0.25, image_height * 0.15, 0
    else:
        gx = x / max(arr.shape[1] - 1, 1) * image_width
        gy = (1 - y / max(arr.shape[0] - 1, 1)) * image_height
        coords = np.column_stack([gx, gy])
        center = np.average(coords, axis=0, weights=weights)
        cov = np.cov((coords - center).T, aweights=weights)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals = np.maximum(vals[order], 1.0)
        vecs = vecs[:, order]
        width = float(np.sqrt(vals[0]) * 2.1)
        height = float(np.sqrt(vals[1]) * 2.1)
        width = float(np.clip(width, image_width * 0.08, image_width * 0.42))
        height = float(np.clip(height, image_height * 0.06, image_height * 0.30))
        angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
        cx, cy = center
    ellipse = Ellipse((cx, cy), width, height, angle=angle, facecolor="none", edgecolor=color, linewidth=1.8, label=label)
    return ellipse


def show_zoom(ax: plt.Axes, arr: np.ndarray, mask: np.ndarray, bounds: tuple[int, int, int, int], title: str, cmap: str) -> None:
    y0, y1, x0, x1 = bounds
    sub_mask = mask[y0:y1, x0:x1]
    sub = norm_panel(arr[y0:y1, x0:x1], sub_mask)
    ax.imshow(sub, cmap=cmap, interpolation="nearest", origin="lower")
    ax.set_title(title, fontsize=6.9, pad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#34a853")


def fig1a_data_and_task_definition(data: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(7.25, 2.35))
    gs = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1.05, 0.88, 0.92, 1.00], wspace=0.18)
    ax = fig.add_subplot(gs[0, 0])
    panel_label(ax, "A", x=-0.03, y=1.05)

    image = Image.open(SPATIAL_DIR / "tissue_hires_image.png").convert("RGB")
    image.thumbnail((900, 900), Image.Resampling.LANCZOS)
    arr = np.asarray(image)
    ax.imshow(arr)
    h, w = arr.shape[:2]
    coords = data["coords"]
    x = np.clip(coords[:, 0], 0, 1) * w
    y = (1 - np.clip(coords[:, 1], 0, 1)) * h
    ax.scatter(x, y, s=2.0, facecolor="none", edgecolor="#6f7682", linewidth=0.16, alpha=0.26)
    source_ellipse = weighted_region_ellipse(data["source"], data["mask"], w, h, percentile=97.0, color="#d94841", label="Apoe source-enriched region")
    barrier_ellipse = weighted_region_ellipse(data["barrier"], data["mask"], w, h, percentile=94.0, color="#2266aa", label="CNS-myelin barrier-enriched region")
    ax.add_patch(source_ellipse)
    ax.add_patch(barrier_ellipse)
    ax.set_title("H&E + Visium spots", fontsize=8.2, fontweight="bold", pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.04, 0.06, "source/barrier\nregions", transform=ax.transAxes, fontsize=5.7, color="#263238", bbox={"facecolor": "white", "alpha": 0.72, "edgecolor": "none", "pad": 1.5})

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    ax2.set_title("Measured inputs", fontsize=8.2, fontweight="bold", pad=3)
    rng = np.random.default_rng(7)
    expr = rng.gamma(shape=1.5, scale=1.0, size=(7, 9))
    expr[rng.random(expr.shape) < 0.48] *= 0.15
    ax2.imshow(expr, cmap="YlOrBr", aspect="auto", extent=(0.08, 0.92, 0.55, 0.92))
    ax2.text(0.08, 0.96, "gene expression matrix", fontsize=5.8, color="#263238")
    xs = np.linspace(0.14, 0.86, 6)
    ys = np.linspace(0.16, 0.42, 4)
    for j, yy in enumerate(ys):
        offset = 0.06 if j % 2 else 0
        ax2.scatter(xs + offset, np.full_like(xs, yy), s=18, facecolor="white", edgecolor="#263238", linewidth=0.7)
    ax2.text(0.08, 0.47, "spatial locations", fontsize=5.8, color="#263238")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis("off")
    ax3.set_title("Task-specific priors", fontsize=8.2, fontweight="bold", pad=3)
    cards = [
        (0.10, 0.72, "Source prior", "example: Apoe", "#d94841"),
        (0.10, 0.48, "Barrier module", "example: CNS myelin", "#2266aa"),
        (0.10, 0.24, "Structural prior", "H&E tissue support", "#6a994e"),
    ]
    for x0, y0, title, value, color in cards:
        ax3.add_patch(Rectangle((x0, y0), 0.80, 0.16, facecolor="#fbfbf8", edgecolor=color, linewidth=0.8))
        ax3.text(x0 + 0.04, y0 + 0.105, title, fontsize=5.6, fontweight="bold", color=color)
        ax3.text(x0 + 0.04, y0 + 0.040, value, fontsize=5.7, color="#263238")
    ax3.text(0.10, 0.07, "Gene/module choices are\ndeclared per analysis task.", fontsize=5.8, color="#263238")

    ax4 = fig.add_subplot(gs[0, 3])
    ax4.axis("off")
    ax4.set_title("Barrier-constrained task", fontsize=8.2, fontweight="bold", pad=3)
    ax4.add_patch(Rectangle((0.10, 0.18), 0.80, 0.66, facecolor="#f7f7f5", edgecolor="#b6b6b6", linewidth=0.5))
    ax4.plot([0.22, 0.78], [0.42, 0.42], color="#2266aa", linewidth=5, solid_capstyle="round")
    ax4.scatter([0.28, 0.72], [0.62, 0.26], s=52, c=["#d94841", "#7c8a99"], edgecolors="white", linewidths=0.6)
    ax4.add_patch(FancyArrowPatch((0.30, 0.60), (0.70, 0.28), arrowstyle="-|>", mutation_scale=9, linewidth=0.8, color="#7c8a99", linestyle="--"))
    ax4.text(0.15, 0.08, "Geometric proximity can conflict\nwith anatomical resistance.", fontsize=6.1, color="#263238")
    save_panel(fig, "Fig1A_data_and_task_definition")


def fig1b_field_construction(data: dict[str, np.ndarray]) -> None:
    fig = plt.figure(figsize=(7.25, 2.35))
    gs = gridspec.GridSpec(2, 5, figure=fig, height_ratios=[0.42, 1.0], wspace=0.10, hspace=0.22)
    ax_flow = fig.add_subplot(gs[0, :])
    ax_flow.axis("off")
    panel_label(ax_flow, "B", x=-0.02, y=1.10)
    steps = [
        ("target expression", r"$S(x,y)$", "#d94841"),
        ("barrier markers", r"$B(x,y)$", "#2266aa"),
        ("H&E morphology", r"$H(x,y)$", "#6a994e"),
        ("scalar diffusion", r"$D=D_H e^{-\alpha B}$", "#7b5ea7"),
        ("resistance view", r"$R=1/D$", "#8a6f3d"),
    ]
    for i, (top, bottom, color) in enumerate(steps):
        x = 0.03 + i * 0.19
        ax_flow.add_patch(Rectangle((x, 0.20), 0.15, 0.58, facecolor="#fbfbf8", edgecolor=color, linewidth=0.7))
        ax_flow.text(x + 0.075, 0.58, top, ha="center", va="center", fontsize=5.6, color="#263238")
        ax_flow.text(x + 0.075, 0.35, bottom, ha="center", va="center", fontsize=7.0, color=color)
        if i < len(steps) - 1:
            ax_flow.annotate("", xy=(x + 0.175, 0.49), xytext=(x + 0.152, 0.49), arrowprops={"arrowstyle": "-|>", "lw": 0.65, "color": "#263238"})
    axes = [fig.add_subplot(gs[1, i]) for i in range(5)]
    show_grid(axes[0], data["source"], data["mask"], "source S\nApoe", "magma", title_fontsize=6.4, title_pad=1)
    show_grid(axes[1], data["barrier"], data["mask"], "barrier B\nMbp/Plp1/Mobp", "YlOrBr", title_fontsize=6.4, title_pad=1)
    show_grid(axes[2], data["resistance"], data["mask"], "resistance\nR=1/D", "viridis", log=True, title_fontsize=6.4, title_pad=1)
    show_grid(axes[3], data["diffusion"], data["mask"], "scalar diffusion\nD(x,y)", "cividis", title_fontsize=6.4, title_pad=1)
    field_context = 0.66 * norm_panel(data["field_post"], data["mask"]) + 0.34 * norm_panel(data["barrier"], data["mask"])
    show_grid(axes[4], field_context, data["mask"], "field-barrier\ncontext", "plasma", title_fontsize=6.4, title_pad=1)
    fig.suptitle("Field construction from expression, morphology, and marker priors", fontsize=8.8, fontweight="bold", y=1.01)
    save_panel(fig, "Fig1B_field_construction")


def fig1c_scalar_pinn_architecture() -> None:
    fig = plt.figure(figsize=(7.25, 2.65))
    ax = fig.add_subplot(111)
    ax.axis("off")
    panel_label(ax, "C", x=-0.04, y=1.04)
    ax.set_title("Scalar PINN architecture and loss terms", fontsize=9.0, fontweight="bold", pad=3)

    # Architecture path
    boxes = [
        (0.05, 0.58, 0.10, 0.18, "(x,y)\ncoords", "#eef4f1"),
        (0.20, 0.58, 0.14, 0.18, "Fourier\nfeatures", "#eef4f1"),
        (0.40, 0.53, 0.17, 0.28, "MLP\nC_theta(x,y)", "#e8eef8"),
        (0.64, 0.58, 0.13, 0.18, "field\nprediction", "#eef4f1"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="#6e8794", linewidth=0.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.5, color="#263238")
    for x0, x1 in [(0.15, 0.20), (0.34, 0.40), (0.57, 0.64)]:
        ax.annotate("", xy=(x1, 0.67), xytext=(x0, 0.67), arrowprops={"arrowstyle": "-|>", "lw": 0.75, "color": "#263238"})

    ax.add_patch(Rectangle((0.04, 0.10), 0.42, 0.30, facecolor="#f6f1e7", edgecolor="#263238", linewidth=0.55))
    ax.text(0.07, 0.34, "Scalar PDE residual", fontsize=7.2, fontweight="bold", color="#263238")
    ax.text(0.07, 0.24, r"$D(x,y)(C_{xx}+C_{yy}) - kC + S(x,y)=0$", fontsize=8.2, color="#263238")
    ax.text(0.07, 0.15, r"$D(x,y)=D_H(x,y)\exp[-\alpha B(x,y)]$", fontsize=7.3, color="#263238")

    loss_boxes = [
        (0.53, 0.32, "data loss\nmeasured spots"),
        (0.70, 0.32, "PDE residual\ntissue domain"),
        (0.53, 0.12, "boundary +\nbackground"),
        (0.70, 0.12, "smoothness\nregularizer"),
    ]
    for x, y, text in loss_boxes:
        ax.add_patch(Rectangle((x, y), 0.14, 0.13, facecolor="#fbfbf8", edgecolor="#b8b8b8", linewidth=0.5))
        ax.text(x + 0.07, y + 0.065, text, ha="center", va="center", fontsize=5.7, color="#263238")
    ax.text(0.53, 0.47, "Training objective", fontsize=7.2, fontweight="bold", color="#263238")
    ax.text(0.04, 0.03, "Current implementation: scalar coefficient and scalar Laplacian residual.", fontsize=5.9, color="#5b5b5b")
    save_panel(fig, "Fig1C_scalar_pinn_architecture")


def fig1d_barrier_mechanism_explanation() -> None:
    schematic_path = SOURCE_ASSET_DIR / D_SCHEMATIC_ASSET
    if not schematic_path.exists():
        schematic_path = PROJECT_ROOT / "reproducibility" / "assets" / D_SCHEMATIC_ASSET
    if not schematic_path.exists():
        raise FileNotFoundError(f"Missing Figure 1D schematic asset: {schematic_path}")
    fig = plt.figure(figsize=(7.25, 2.25))
    ax = fig.add_subplot(111)
    image = Image.open(schematic_path).convert("RGB")
    ax.imshow(image)
    ax.set_title("Barrier-constrained attenuation", fontsize=9.0, fontweight="bold", pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_panel(fig, "Fig1D_barrier_mechanism_explanation")


def fig1e_output_and_local_zoom(data: dict[str, np.ndarray]) -> None:
    bounds = high_score_bounds(data)
    fig = plt.figure(figsize=(7.25, 2.75))
    gs = gridspec.GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1.05], wspace=0.12, hspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    panel_label(ax1, "E", x=-0.15, y=1.12)
    show_grid(ax1, data["source"], data["mask"], "Measured source", "magma", rectangle=bounds)
    ax2 = fig.add_subplot(gs[0, 1])
    show_grid(ax2, data["field_raw"], data["mask"], "PINN field", "viridis", rectangle=bounds)
    ax3 = fig.add_subplot(gs[0, 2])
    show_grid(ax3, data["field_masked"], data["mask"], "Tissue-masked field", "viridis", rectangle=bounds)
    ax4 = fig.add_subplot(gs[0, 3])
    show_grid(ax4, data["barrier"], data["mask"], "Barrier context", "YlOrBr", rectangle=bounds)
    z1 = fig.add_subplot(gs[1, 0])
    show_zoom(z1, data["source"], data["mask"], bounds, "Zoom: source", "magma")
    z2 = fig.add_subplot(gs[1, 1])
    show_zoom(z2, data["field_post"], data["mask"], bounds, "Zoom: inferred field", "viridis")
    z3 = fig.add_subplot(gs[1, 2])
    show_zoom(z3, data["barrier"], data["mask"], bounds, "Zoom: barrier", "YlOrBr")
    ax_note = fig.add_subplot(gs[1, 3])
    ax_note.axis("off")
    ax_note.add_patch(Rectangle((0.04, 0.14), 0.92, 0.72, facecolor="#f7f7f5", edgecolor="#b8b8b8", linewidth=0.45))
    ax_note.text(0.10, 0.76, "Local zoom QC", fontsize=7.0, fontweight="bold", color="#263238", va="top")
    ax_note.text(0.10, 0.52, "Same tissue\nneighborhood", fontsize=5.8, color="#263238", linespacing=1.10, va="top")
    ax_note.text(0.10, 0.30, "Tests: Figures 2-5", fontsize=5.6, color="#5b5b5b", va="top")
    fig.suptitle("Representative output and barrier-neighborhood zoom", fontsize=8.8, fontweight="bold", y=1.02)
    save_panel(fig, "Fig1E_output_and_local_zoom")


def fig1_metric_ribbon(data: dict[str, np.ndarray]) -> None:
    metrics = pd.read_csv(BATCH_METRICS)
    row = metrics[(metrics["sample"] == SAMPLE) & (metrics["field_type"] == "gauss07")].iloc[0]
    preflight = json.loads((PREFLIGHT_DIR / "preflight_metrics.json").read_text(encoding="utf-8"))
    cards = [
        ("Source fit", "Pearson", float(row["spot_pearson_source"]), 1.0, "#3f7f93"),
        ("Smoothness", "roughness p95", float(row["roughness_grad_p95"]), 0.18, "#c58c2b"),
        ("Tissue support", "leakage ratio", float(row["background_to_tissue_ratio"]), 0.08, "#6a994e"),
        ("Barrier scale", "R ratio", float(preflight["resistance_ratio_in_tissue"]), 360.0, "#7b5ea7"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(6.1, 1.55), gridspec_kw={"wspace": 0.14})
    panel_label(axes[0], "E", x=-0.18, y=1.08)
    for ax, (title, label, value, scale, color) in zip(axes, cards):
        ax.axis("off")
        ax.add_patch(Rectangle((0.04, 0.12), 0.92, 0.76, facecolor="#fbfbf8", edgecolor="#b8b8b8", linewidth=0.45))
        ax.text(0.10, 0.75, title, fontsize=7.7, fontweight="bold", color="#263238")
        ax.text(0.10, 0.58, label, fontsize=6.3, color="#5b5b5b")
        frac = min(max(value / scale, 0), 1)
        ax.add_patch(Rectangle((0.10, 0.30), 0.76, 0.12, facecolor="#e5e5e5", edgecolor="none"))
        ax.add_patch(Rectangle((0.10, 0.30), 0.76 * frac, 0.12, facecolor=color, edgecolor="none"))
        fmt = f"{value:.3f}" if value < 10 else f"{value:.0f}x"
        ax.text(0.10, 0.17, fmt, fontsize=7.6, fontweight="bold", color=color)
    fig.suptitle("Metric families used throughout the revision", fontsize=8.8, fontweight="bold", y=1.05)
    save_panel(fig, "Fig1_metric_ribbon")


def rough_assembly() -> None:
    panel_files = [
        ("A", "Fig1A_data_and_task_definition.png"),
        ("B", "Fig1B_field_construction.png"),
        ("C", "Fig1C_scalar_pinn_architecture.png"),
        ("D", "Fig1D_barrier_mechanism_explanation.png"),
        ("E", "Fig1E_output_and_local_zoom.png"),
    ]
    fig = plt.figure(figsize=(7.4, 7.25))
    gs = gridspec.GridSpec(4, 2, figure=fig, height_ratios=[0.95, 0.90, 0.90, 1.12], wspace=0.08, hspace=0.08)
    layout = {
        "A": gs[0, :],
        "B": gs[1, :],
        "C": gs[2, 0],
        "D": gs[2, 1],
        "E": gs[3, :],
    }
    for label, filename in panel_files:
        ax = fig.add_subplot(layout[label])
        image = Image.open(OUT_DIR / filename).convert("RGB")
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Figure 1 rough assembly: anisoNET method overview", fontsize=11, fontweight="bold", y=0.985)
    save_panel(fig, "Figure1_rough_assembly")


def write_current_panel_set() -> None:
    text = """# Current Figure 1 Panel Set

Use these panels for the current GPB-style Figure 1 method overview:

- `Fig1A_data_and_task_definition.pdf/png`
- `Fig1B_field_construction.pdf/png`
- `Fig1C_scalar_pinn_architecture.pdf/png`
- `Fig1D_barrier_mechanism_explanation.pdf/png`
- `Fig1E_output_and_local_zoom.pdf/png`
- `Figure1_rough_assembly.pdf/png`

Current revision note:

- Figure 1 now uses real GSE193107 old A1 Apoe/CNS-myelin data as the visual anchor.
- The figure defines data inputs, task-specific priors, field construction, scalar PINN architecture, barrier mechanism, representative output, and local zoom QC.
- Panel letters are intentionally not embedded in panel assets; add final A-E labels manually during Inkscape assembly.
- The rough assembly is for orientation only; final journal spacing and typography should be refined manually.
"""
    (OUT_DIR / "CURRENT_PANEL_SET.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup()
    ensure_dir(OUT_DIR)
    data = load_inputs()
    fig1a_data_and_task_definition(data)
    fig1b_field_construction(data)
    fig1c_scalar_pinn_architecture()
    fig1d_barrier_mechanism_explanation()
    fig1e_output_and_local_zoom(data)
    rough_assembly()
    write_current_panel_set()
    print(f"Wrote Figure 1 panels to {OUT_DIR}")


if __name__ == "__main__":
    main()
