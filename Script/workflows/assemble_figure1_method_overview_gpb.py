"""Assemble a data-backed GPB-style Figure 1 method overview draft."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle
from PIL import Image


PROJECT_ROOT = Path(r"K:\YC\experiment\STagent")
SAMPLE = "GSM5773457_Old_mouse_brain_A1-2"
TASK = "Apoe_CNS_Myelin"
PRELIGHT_DIR = PROJECT_ROOT / "codexAnalysis" / "preflight" / "brain_aging_gse193107" / SAMPLE / TASK
PINN_DIR = (
    PROJECT_ROOT
    / "codexAnalysis"
    / "pinn"
    / "brain_aging_gse193107"
    / SAMPLE
    / TASK
    / "fourier_refined_16g_gauss07_batch"
)
SPATIAL_DIR = PROJECT_ROOT / "codexAnalysis" / "processed_visium" / "brain_aging_gse193107" / SAMPLE / "spatial"
OUTPUT_DIR = PROJECT_ROOT / "codexAnalysis" / "manuscript_figures" / "main"


def setup() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 7.2,
            "axes.linewidth": 0.45,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_array(name: str) -> np.ndarray:
    return np.asarray(np.load(PRELIGHT_DIR / name), dtype=float)


def load_inputs() -> dict[str, np.ndarray]:
    return {
        "source": load_array("source_grid.npy"),
        "barrier": load_array("barrier_grid.npy"),
        "diffusion": load_array("diffusion_grid.npy"),
        "resistance": load_array("resistance_grid.npy"),
        "mask": load_array("tissue_mask.npy") > 0,
        "coords": load_array("coords_norm.npy"),
        "field": np.asarray(np.load(PINN_DIR / "pinn_grid_prediction_postprocessed.npy"), dtype=float),
    }


def norm_panel(arr: np.ndarray, mask: np.ndarray | None = None, *, log: bool = False) -> np.ndarray:
    data = np.asarray(arr, dtype=float)
    if log:
        data = np.log10(np.maximum(data, np.nanpercentile(data[np.isfinite(data)], 1) * 0.1))
    valid = np.isfinite(data)
    if mask is not None:
        valid &= mask
    if not np.any(valid):
        return np.zeros_like(data)
    lo, hi = np.nanpercentile(data[valid], [2, 98])
    if hi <= lo:
        hi = lo + 1e-6
    out = (data - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    if mask is not None:
        out = np.where(mask, out, np.nan)
    return out


def crop_to_mask(arr: np.ndarray, mask: np.ndarray, pad: int = 7) -> np.ndarray:
    yy, xx = np.where(mask)
    if len(xx) == 0:
        return arr
    y0, y1 = max(int(yy.min()) - pad, 0), min(int(yy.max()) + pad + 1, arr.shape[0])
    x0, x1 = max(int(xx.min()) - pad, 0), min(int(xx.max()) + pad + 1, arr.shape[1])
    return arr[y0:y1, x0:x1]


def show_grid(ax, arr: np.ndarray, mask: np.ndarray, title: str, cmap: str, *, log: bool = False) -> None:
    panel = crop_to_mask(norm_panel(arr, mask, log=log), mask)
    ax.imshow(panel, cmap=cmap, interpolation="bilinear")
    ax.set_title(title, fontsize=7.4, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_he(ax, coords: np.ndarray) -> None:
    image = Image.open(SPATIAL_DIR / "tissue_hires_image.png").convert("RGB")
    image.thumbnail((900, 900), Image.Resampling.LANCZOS)
    ax.imshow(image)
    # Normalized coordinates are plotted as a spatial sampling cue only; exact spot-pixel
    # registration is handled in preprocessing and shown in downstream field panels.
    h, w = np.asarray(image).shape[:2]
    x = np.clip(coords[:, 0], 0, 1) * w
    y = (1 - np.clip(coords[:, 1], 0, 1)) * h
    ax.scatter(x, y, s=3.2, facecolor="#f08a4b", edgecolor="white", linewidth=0.12, alpha=0.72)
    ax.set_title("H&E + Visium spots", fontsize=7.4, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_label(fig, ax, label: str) -> None:
    box = ax.get_position()
    fig.text(box.x0 - 0.014, box.y1 + 0.006, label, fontsize=10, weight="bold", ha="left", va="bottom")


def arrow(fig, ax_from, ax_to) -> None:
    start_box = ax_from.get_position()
    end_box = ax_to.get_position()
    start = (start_box.x1 + 0.006, (start_box.y0 + start_box.y1) / 2)
    end = (end_box.x0 - 0.006, (end_box.y0 + end_box.y1) / 2)
    fig.patches.append(
        FancyArrowPatch(
            start,
            end,
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.7,
            color="#263238",
        )
    )


def draw_equation_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(Rectangle((0.03, 0.08), 0.94, 0.84, facecolor="#f6f1e7", edgecolor="#263238", linewidth=0.55))
    ax.text(0.08, 0.82, "Scalar PINN field model", fontsize=8.6, weight="bold", color="#263238")
    ax.text(0.08, 0.64, r"$D(x,y)(C_{xx}+C_{yy}) - kC + S(x,y)=0$", fontsize=10, color="#263238")
    ax.text(0.08, 0.45, r"$D(x,y)=D_H(x,y)\exp[-\alpha B(x,y)]$", fontsize=8.3, color="#263238")
    ax.text(0.08, 0.27, "Loss: data + PDE residual + boundary/background + smoothness", fontsize=7.2, color="#263238")
    ax.text(0.08, 0.14, "Scalar coefficient; not tensor or full divergence-form diffusion.", fontsize=6.8, color="#8a1c1c")


def draw_workflow_box(ax) -> None:
    ax.axis("off")
    steps = [
        ("1", "Preflight\nfields"),
        ("2", "PINN\nfit"),
        ("3", "Masked\nfield"),
        ("4", "Metrics\nand QC"),
    ]
    x0, y0, w, h = 0.03, 0.26, 0.20, 0.46
    for i, (idx, text) in enumerate(steps):
        x = x0 + i * 0.235
        ax.add_patch(Rectangle((x, y0), w, h, facecolor="#eef4f1", edgecolor="#263238", linewidth=0.5))
        ax.text(x + 0.025, y0 + h - 0.10, idx, fontsize=8.5, weight="bold", color="#2a6f62")
        ax.text(x + 0.065, y0 + h - 0.12, text, fontsize=7.0, va="top", color="#263238")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + 0.035, y0 + 0.25), xytext=(x + w + 0.005, y0 + 0.25), arrowprops={"arrowstyle": "-|>", "lw": 0.7, "color": "#263238"})
    ax.text(0.03, 0.86, "Reproducible analysis workflow", fontsize=8.6, weight="bold", color="#263238")
    ax.text(0.03, 0.10, "Field-modeling workflow; benchmark limits are reported in later figures.", fontsize=7.0, color="#263238")


def draw_parameter_box(ax) -> None:
    ax.axis("off")
    ax.add_patch(Rectangle((0.04, 0.14), 0.92, 0.72, facecolor="#f7f7f5", edgecolor="#b8b8b8", linewidth=0.45))
    ax.text(0.08, 0.76, "Figure 1 anchors", fontsize=8.4, weight="bold", color="#263238")
    rows = [
        ("Data", "GSE193107 old A1"),
        ("Target", "Apoe"),
        ("Barrier", "Mbp/Plp1/Mobp"),
        ("Grid", "200 x 200"),
        ("Readout", "masked field"),
    ]
    y = 0.64
    for key, value in rows:
        ax.text(0.09, y, key, fontsize=7.0, weight="bold", color="#263238")
        ax.text(0.42, y, value, fontsize=7.0, color="#263238")
        y -= 0.092
    ax.text(0.08, 0.18, "No universal interpolation-superiority claim.", fontsize=6.2, color="#8a1c1c")


def main() -> None:
    setup()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    mask = data["mask"]

    fig = plt.figure(figsize=(7.4, 5.25), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        6,
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=0.055,
        wspace=0.25,
        hspace=0.40,
        height_ratios=[1.06, 1.0, 0.88],
    )

    fig.suptitle(
        "anisoNET: tissue-constrained scalar field modeling with histology and transcriptomic barriers",
        fontsize=10.2,
        weight="bold",
        y=0.968,
    )

    ax_he = fig.add_subplot(gs[0, 0:2])
    show_he(ax_he, data["coords"])
    panel_label(fig, ax_he, "A")

    ax_source = fig.add_subplot(gs[0, 2])
    show_grid(ax_source, data["source"], mask, "Source S(x,y)\nApoe", "magma")
    panel_label(fig, ax_source, "B")

    ax_barrier = fig.add_subplot(gs[0, 3])
    show_grid(ax_barrier, data["barrier"], mask, "Barrier B(x,y)\nCNS myelin", "YlOrBr")

    ax_res = fig.add_subplot(gs[0, 4])
    show_grid(ax_res, data["resistance"], mask, "Resistance R=1/D", "viridis", log=True)

    ax_diff = fig.add_subplot(gs[0, 5])
    show_grid(ax_diff, data["diffusion"], mask, "Diffusion D(x,y)", "cividis")

    ax_eq = fig.add_subplot(gs[1, 0:3])
    draw_equation_box(ax_eq)
    panel_label(fig, ax_eq, "C")

    ax_raw = fig.add_subplot(gs[1, 3])
    show_grid(ax_raw, data["field"], np.ones_like(mask, dtype=bool), "PINN field", "mako" if "mako" in plt.colormaps() else "viridis")
    panel_label(fig, ax_raw, "D")

    ax_masked = fig.add_subplot(gs[1, 4])
    show_grid(ax_masked, data["field"], mask, "Tissue-masked\nfield", "mako" if "mako" in plt.colormaps() else "viridis")

    ax_context = fig.add_subplot(gs[1, 5])
    context = norm_panel(data["field"], mask) * 0.72 + norm_panel(data["resistance"], mask, log=True) * 0.28
    show_grid(ax_context, context, mask, "Field + barrier\ncontext", "rocket" if "rocket" in plt.colormaps() else "plasma")

    ax_workflow = fig.add_subplot(gs[2, 0:4])
    draw_workflow_box(ax_workflow)
    panel_label(fig, ax_workflow, "E")

    ax_params = fig.add_subplot(gs[2, 4:6])
    draw_parameter_box(ax_params)
    panel_label(fig, ax_params, "F")

    arrow(fig, ax_he, ax_source)
    arrow(fig, ax_eq, ax_raw)

    out_pdf = OUTPUT_DIR / "Figure1_method_overview_gpb_rebuild_draft.pdf"
    out_png = OUTPUT_DIR / "Figure1_method_overview_gpb_rebuild_draft.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
